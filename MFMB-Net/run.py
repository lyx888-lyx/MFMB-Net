import os
import gc
import time
import random
import logging
import torch
import pynvml
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool

from models.AMIO import AMIO
from trains.ATIO import ATIO
from data.load_data import MMDataLoader
from config.config_regression import ConfigRegression

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run(args):

    '''
    feature_dims(768, 5, 20)
    '''
    save_tag = getattr(args, 'exp_tag', '') or ('mide' if getattr(args, 'mide_enable', False) else 'baseline')
    missing_tag = f"m{getattr(args, 'missing_rate', (0,))[0]:.1f}" if hasattr(args, 'missing_rate') else 'm0.0'
    args.model_save_dir = os.path.join(args.model_save_dir, save_tag, missing_tag)
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    ckpt_name = f'{args.modelName}-{args.datasetName}-{args.train_mode}-{save_tag}-{missing_tag}-seed{args.seed}.pth'
    args.model_save_path = os.path.join(args.model_save_dir, ckpt_name)
    # indicate used gpu
    if len(args.gpu_ids) == 0 and torch.cuda.is_available():
        # load free-most gpu
        pynvml.nvmlInit()
        device_count=pynvml.nvmlDeviceGetCount();
        dst_gpu_id, min_mem_used = 0, 1e16
        #for g_id in [0, 1]:
        for g_id in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(g_id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used = meminfo.used
            if mem_used < min_mem_used:
                min_mem_used = mem_used
                dst_gpu_id = g_id
        print(f'Find gpu: {dst_gpu_id}, use memory: {min_mem_used}!')
        logger.info(f'Find gpu: {dst_gpu_id}, with memory: {min_mem_used} left!')
        # When CUDA_VISIBLE_DEVICES remaps GPUs, always use logical device 0.
        if os.environ.get('CUDA_VISIBLE_DEVICES', '').strip() != '':
            args.gpu_ids.append(0)
        else:
            args.gpu_ids.append(dst_gpu_id)
    # device
    using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
    logger.info("Let's use %d GPUs!" % len(args.gpu_ids))
    device = torch.device('cuda:%d' % int(args.gpu_ids[0]) if using_cuda else 'cpu')
    args.device = device
    # add tmp tensor to increase the temporary consumption of GPU
    #tmp_tensor = torch.zeros((100, 100)).to(args.device)
    # load data and models
    dataloader = MMDataLoader(args)
    model = AMIO(args).to(device)

    #del tmp_tensor

    def count_parameters(model):
        answer = 0
        for p in model.parameters():
            if p.requires_grad:
                answer += p.numel()
                # print(p)
        return answer
    logger.info(f'The model has {count_parameters(model)} trainable parameters')
    atio = ATIO().getTrain(args)
    # do train
    atio.do_train(model, dataloader)
    # load pretrained model
    if not os.path.exists(args.model_save_path):
        raise FileNotFoundError(f'No checkpoint saved at {args.model_save_path}')
    model.load_state_dict(torch.load(args.model_save_path, map_location=device))
    model.to(device)
    # do test
    if args.is_tune:
        # using valid dataset to tune hyper parameters
        results = atio.do_test(model, dataloader['test'], mode="TEST")
    else:
        results = atio.do_test(model, dataloader['test'], mode="TEST")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(5)
 
    return results

def run_normal(args):
    subdir = args.exp_tag if args.exp_tag else ('mide' if getattr(args, 'mide_enable', False) else 'baseline')
    args.res_save_dir = os.path.join(args.res_save_dir, subdir)
    init_args = args
    model_results = []
    seeds = args.seeds
    missing_rate = 0.0
    # run results
    for i, seed in enumerate(seeds):
        args = init_args
        # load config
        config = ConfigRegression(args)
        args = config.get_config()
        if getattr(init_args, 'mide_enable', False):
            args.mide_enable = True
        if getattr(init_args, 'use_amp', False):
            args.use_amp = True
        if i == 0 and args.data_missing:
            missing_rate = str(args.missing_rate[0])
        setup_seed(seed)
        args.seed = seed
        logger.info('Start running %s...' %(args.modelName))
        logger.info(args)
        # runnning
        args.cur_time = i+1
        test_results = run(args)
        # restore results
        model_results.append(test_results)
    criterions = list(model_results[0].keys())
    # load other results
    save_path = os.path.join(args.res_save_dir, \
                        f'{args.datasetName}-{args.train_mode}-{missing_rate}.csv')
    if not os.path.exists(args.res_save_dir):
        os.makedirs(args.res_save_dir)
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame(columns=["Model"] + criterions)
    # save results
    res = [args.modelName]
    for c in criterions:
        values = [r[c] for r in model_results]
        mean = round(np.mean(values)*100, 2)
        std = round(np.std(values)*100, 2)
        res.append((mean, std))
    df.loc[len(df)] = res
    df.to_csv(save_path, index=None)
    logger.info('Results are added to %s...' %(save_path))

def set_log(args):
    exp_tag = getattr(args, 'exp_tag', '') or ('mide' if getattr(args, 'mide_enable', False) else 'baseline')
    missing = getattr(args, 'missing', 0.0)
    log_dir = os.path.join('results', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f'{exp_tag}_m{missing:.1f}.log')
    # set logging
    logger = logging.getLogger() 
    logger.setLevel(logging.DEBUG)

    for ph in logger.handlers:
        logger.removeHandler(ph)
    # add FileHandler to log file
    formatter_file = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter_file)
    logger.addHandler(sh)
    return logger

def _add_mide_cli(parser):
    parser.add_argument('--mide_variant', type=str, default=None,
                        choices=['old', 'split_aup', 'split_aup_plus'])
    mide_floats = [
        'mide_tau_uni', 'mide_tau_loo', 'mide_tau_p', 'mide_beta_uni', 'mide_beta_loo',
        'mide_margin', 'mide_contrib_eps', 'mide_pos_threshold', 'mide_neg_threshold',
        'mide_avail_high', 'mide_avail_low', 'mide_u_low', 'mide_keep_density_target',
        'mide_d_floor_min', 'mide_lambda_uni', 'mide_lambda_util_bce', 'mide_lambda_rank',
        'mide_lambda_poll_bce', 'mide_lambda_noinfo', 'mide_lambda_poll',
        'mide_lambda_sparse', 'mide_lambda_sep', 'mide_head_hidden', 'mide_head_dropout',
        'mide_sep_eps', 'mide_sep_margin',
    ]
    for name in mide_floats:
        parser.add_argument(f'--{name}', type=float, default=None)
    mide_ints = ['mide_aux_start_epoch', 'mide_full_start_epoch', 'mide_gate_start_epoch']
    for name in mide_ints:
        parser.add_argument(f'--{name}', type=int, default=None)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mode', type=str, default="regression",
                        help='regression')
    parser.add_argument('--modelName', type=str, default='mfmb_net',
                        help='support mfmb_net')
    parser.add_argument('--datasetName', type=str, default='mosi',
                        help='support mosi/mosei')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num workers of loading data')
    parser.add_argument('--model_save_dir', type=str, default='results/models',
                        help='path to save results.')
    parser.add_argument('--res_save_dir', type=str, default='results/results',
                        help='path to save results.')
    parser.add_argument('--gpu_ids', type=list, default=[],
                        help='indicates the gpus will be used. If none, the most-free gpu will be used!')
    parser.add_argument('--missing', type=float, default=0.0)
    parser.add_argument('--data_root', type=str, default=None,
                        help='Dataset root directory. Overrides MFMB_DATA_ROOT env and default path.')
    parser.add_argument('--mide_enable', action='store_true', default=False,
                        help='Enable MIDE module.')
    parser.add_argument('--exp_tag', type=str, default='',
                        help='Experiment tag for result/model subdirectories.')
    parser.add_argument('--batch_size_override', type=int, default=None)
    parser.add_argument('--lr_other_override', type=float, default=None)
    parser.add_argument('--lr_bert_override', type=float, default=None)
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Enable automatic mixed precision training.')
    parser.add_argument('--save_metric', type=str, default='loss',
                        choices=['loss', 'corr', 'has0'],
                        help='Primary checkpoint metric (default loss for comparability).')
    _add_mide_cli(parser)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    args.missing_rate = tuple([args.missing, args.missing, args.missing])
    args.use_amp = args.amp
    global logger; logger = set_log(args)
    args.seeds = [111, 1111, 11111]
    run_normal(args)
