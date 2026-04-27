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


def _fusion_center_slug(args):
    """Isolate logs / results / checkpoints between micro-fusion and local-encoder settings."""
    fc = getattr(args, 'fusion_center_modality', 'text')
    lte = getattr(args, 'local_temporal_encoder_type', 'legacy')
    if lte == 'legacy':
        return fc
    return '%s_lte_%s' % (fc, lte)


def _apply_missing_rates(args):
    """Resolve per-modality missing rates; set args.missing_rate tuple.

    Rules:
    - Each modality uses --text_missing_rate / --audio_missing_rate / --vision_missing_rate when given.
    - Otherwise falls back to --missing.
    - All resolved values must lie in [0, 1].
    """
    m = float(getattr(args, 'missing', 0.0))
    if not (0.0 <= m <= 1.0):
        raise ValueError('--missing must be in [0, 1], got %s' % (m,))

    def _one(flag_name, raw):
        if raw is None:
            return m
        v = float(raw)
        if not (0.0 <= v <= 1.0):
            raise ValueError('%s must be in [0, 1], got %s' % (flag_name, v))
        return v

    args.text_missing_rate = _one(
        '--text_missing_rate', getattr(args, 'text_missing_rate', None)
    )
    args.audio_missing_rate = _one(
        '--audio_missing_rate', getattr(args, 'audio_missing_rate', None)
    )
    args.vision_missing_rate = _one(
        '--vision_missing_rate', getattr(args, 'vision_missing_rate', None)
    )
    args.missing_rate = (args.text_missing_rate, args.audio_missing_rate, args.vision_missing_rate)


def _missing_results_slug(args):
    """CSV filename segment: single number if symmetric, else t/a/v tag (asymmetric)."""
    t, a, v = args.text_missing_rate, args.audio_missing_rate, args.vision_missing_rate
    if t == a == v:
        return str(t)
    return 't%s_a%s_v%s' % (t, a, v)


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
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    fc = _fusion_center_slug(args)
    args.model_save_path = os.path.join(
        args.model_save_dir,
        f'{args.modelName}-{args.datasetName}-{args.train_mode}-fc_{fc}.pth',
    )
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
    assert os.path.exists(args.model_save_path)
    model.load_state_dict(torch.load(args.model_save_path))
    model.to(device)
    # do test
    if args.is_tune:
        # using valid dataset to tune hyper parameters
        results = atio.do_test(model, dataloader['test'], mode="TEST")
    else:
        results = atio.do_test(model, dataloader['test'], mode="TEST")

    # 如果不需要长期保留 checkpoint，测试完就删掉
    if (not args.keep_ckpt) and os.path.exists(args.model_save_path):
        try:
            os.remove(args.model_save_path)
            logger.info(f"Removed checkpoint: {args.model_save_path}")
        except Exception as e:
            logger.warning(f"Failed to remove checkpoint {args.model_save_path}: {e}")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(5)
 
    return results

def run_normal(args):
    fc = _fusion_center_slug(args)
    args.res_save_dir = os.path.join(args.res_save_dir, 'normals', fc)
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
        if i == 0 and args.data_missing:
            missing_rate = _missing_results_slug(args)
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
    os.makedirs('logs', exist_ok=True)
    fc = _fusion_center_slug(args)
    log_file_path = os.path.join('logs', f'{args.modelName}-{args.datasetName}-fc_{fc}.log')
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
    return logger

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
    parser.add_argument(
        '--missing',
        type=float,
        default=0.0,
        help='Default missing rate for all modalities [0,1]. Used when a per-modality rate is not set.',
    )
    parser.add_argument(
        '--text_missing_rate',
        type=float,
        default=None,
        help='Text missing rate [0,1]. Default: same as --missing.',
    )
    parser.add_argument(
        '--audio_missing_rate',
        type=float,
        default=None,
        help='Audio missing rate [0,1]. Default: same as --missing.',
    )
    parser.add_argument(
        '--vision_missing_rate',
        type=float,
        default=None,
        help='Vision missing rate [0,1]. Default: same as --missing.',
    )
    parser.add_argument(
        '--fusion_center_modality',
        type=str,
        default='text',
        choices=['text', 'audio', 'vision', 'dynamic_missing'],
        help="GATE_F micro-fusion: 'text'/'audio'/'vision' fixed stacks, or 'dynamic_missing' (soft blend of three centers; missing-only weights). Default=text.",
    )
    parser.add_argument(
        '--dynamic_anchor_prior_text',
        type=float,
        default=0.30,
        help='Soft dynamic anchor: logit prior for text center (before beta*r). Typical: > prior_audio.',
    )
    parser.add_argument(
        '--dynamic_anchor_prior_audio',
        type=float,
        default=0.24,
        help='Soft dynamic anchor: logit prior for audio center.',
    )
    parser.add_argument(
        '--dynamic_anchor_prior_vision',
        type=float,
        default=0.22,
        help='Soft dynamic anchor: logit prior for vision center.',
    )
    parser.add_argument(
        '--dynamic_anchor_beta',
        type=float,
        default=2.0,
        help='Soft dynamic anchor: scale on valid_ratio in logits (prior + beta * r).',
    )
    parser.add_argument(
        '--dynamic_anchor_min_valid',
        type=float,
        default=1e-3,
        help='Soft dynamic anchor: modalities with valid_ratio below this get near-zero weight.',
    )
    parser.add_argument(
        '--local_temporal_encoder_type',
        type=str,
        default='legacy',
        choices=['legacy', 'mstcn'],
        help="GATE_F local/micro temporal encoder: 'legacy' (Bi-GRU+C_GATE) or 'mstcn' (light multi-scale Conv1d). Default=legacy.",
    )
    parser.add_argument("--keep_ckpt", action="store_true", help="whether to keep checkpoint files after test")
    parser.add_argument(
        '--debug_data_inspect',
        action='store_true',
        help='Log actual dataPath, pickle path, split sizes, shapes, missing=0.0 checks, first batch (once).',
    )
    parser.add_argument(
        '--export_test_predictions',
        action='store_true',
        help='After TEST, save predictions CSV (+ meta json) under export_pred_dir; one file per seed.',
    )
    parser.add_argument(
        '--export_pred_dir',
        type=str,
        default='results/predictions',
        help='Directory for exported test prediction CSVs.',
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    _apply_missing_rates(args)
    global logger; logger = set_log(args)
    args.seeds = [111, 1111, 11111]
    run_normal(args)
