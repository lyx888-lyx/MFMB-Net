import os
import gc
import time
import random
import logging
import argparse
from typing import List

import torch
import pynvml
import numpy as np
import pandas as pd

from models.AMIO import AMIO
from trains.ATIO import ATIO
from data.load_data import MMDataLoader
from config.config_regression import ConfigRegression

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_gpu_ids(raw) -> List[int]:
    if isinstance(raw, list):
        return raw
    if raw is None:
        return []
    text = str(raw).strip()
    if text == "":
        return []
    return [int(x) for x in text.split(',') if x.strip() != ""]


def parse_seeds(raw) -> List[int]:
    if isinstance(raw, list):
        return [int(x) for x in raw]
    return [int(x.strip()) for x in str(raw).split(',') if x.strip()]


def fmt_rate(x: float) -> str:
    text = f"{float(x):.4f}".rstrip('0').rstrip('.')
    return text if text else '0'


def make_missing_tag(missing_rate) -> str:
    t, a, v = missing_rate
    return f"t{fmt_rate(t)}_a{fmt_rate(a)}_v{fmt_rate(v)}"


def resolve_missing_rate(args):
    t = args.missing if args.missing_t is None else args.missing_t
    a = args.missing if args.missing_a is None else args.missing_a
    v = args.missing if args.missing_v is None else args.missing_v
    return (float(t), float(a), float(v))


def build_run_name(args):
    base = f"{args.modelName}-{args.datasetName}-{args.train_mode}-{make_missing_tag(args.missing_rate)}"
    if args.run_tag:
        base = f"{base}-{args.run_tag}"
    return base


def run(args):
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir, exist_ok=True)

    if len(args.gpu_ids) == 0 and torch.cuda.is_available():
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        dst_gpu_id, min_mem_used = 0, 1e16
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

    using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
    logger.info("Let's use %d GPUs!" % len(args.gpu_ids))
    device = torch.device('cuda:%d' % int(args.gpu_ids[0]) if using_cuda else 'cpu')
    args.device = device

    dataloader = MMDataLoader(args)
    model = AMIO(args).to(device)

    def count_parameters(model):
        answer = 0
        for p in model.parameters():
            if p.requires_grad:
                answer += p.numel()
        return answer

    logger.info(f'The model has {count_parameters(model)} trainable parameters')
    atio = ATIO().getTrain(args)
    atio.do_train(model, dataloader)

    assert os.path.exists(args.model_save_path), f"Model checkpoint not found: {args.model_save_path}"
    model.load_state_dict(torch.load(args.model_save_path, map_location=device))
    model.to(device)
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
    time.sleep(2)
    return results


def run_normal(args):
    args.res_save_dir = os.path.join(args.res_save_dir, 'normals')
    init_args = args
    model_results = []
    seeds = args.seeds

    for i, seed in enumerate(seeds):
        args = init_args
        config = ConfigRegression(args)
        args = config.get_config()
        args.missing_rate = init_args.missing_rate
        args.gpu_ids = list(init_args.gpu_ids)
        args.seeds = init_args.seeds
        args.run_tag = init_args.run_tag
        args.seed = seed
        args.cur_time = i + 1
        args.experiment_name = build_run_name(args)
        args.model_save_path = os.path.join(args.model_save_dir, f'{args.experiment_name}.pth')
        setup_seed(seed)
        logger.info('Start running %s...' % (args.modelName))
        logger.info(args)
        test_results = run(args)
        model_results.append(test_results)

    criterions = list(model_results[0].keys())
    if not os.path.exists(args.res_save_dir):
        os.makedirs(args.res_save_dir, exist_ok=True)

    csv_name = f'{args.datasetName}-{args.train_mode}-{make_missing_tag(args.missing_rate)}'
    if args.run_tag:
        csv_name += f'-{args.run_tag}'
    csv_name += '.csv'
    save_path = os.path.join(args.res_save_dir, csv_name)

    df = pd.DataFrame(columns=["Model", "MissingText", "MissingAudio", "MissingVision"] + criterions)
    res = [args.modelName, args.missing_rate[0], args.missing_rate[1], args.missing_rate[2]]
    for c in criterions:
        values = [r[c] for r in model_results]
        mean = round(np.mean(values) * 100, 2)
        std = round(np.std(values) * 100, 2)
        res.append((mean, std))
    df.loc[len(df)] = res
    df.to_csv(save_path, index=None)
    logger.info('Results are written to %s...' % (save_path))


def set_log(args):
    os.makedirs('logs', exist_ok=True)
    log_file_name = f'{args.modelName}-{args.datasetName}-{make_missing_tag(args.missing_rate)}'
    if args.run_tag:
        log_file_name += f'-{args.run_tag}'
    log_file_path = os.path.join('logs', f'{log_file_name}.log')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    for ph in logger.handlers:
        logger.removeHandler(ph)

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mode', type=str, default='regression', help='regression')
    parser.add_argument('--modelName', type=str, default='mfmb_net', help='support mfmb_net')
    parser.add_argument('--datasetName', type=str, default='mosi', help='support mosi/mosei')
    parser.add_argument('--num_workers', type=int, default=0, help='num workers of loading data')
    parser.add_argument('--model_save_dir', type=str, default='results/models', help='path to save models.')
    parser.add_argument('--res_save_dir', type=str, default='results/results', help='path to save results.')
    parser.add_argument('--gpu_ids', type=str, default='', help='comma-separated GPU ids. Empty means auto-select.')
    parser.add_argument('--missing', type=float, default=0.0, help='legacy shared missing rate for all 3 modalities')
    parser.add_argument('--missing_t', type=float, default=None, help='text missing rate')
    parser.add_argument('--missing_a', type=float, default=None, help='audio missing rate')
    parser.add_argument('--missing_v', type=float, default=None, help='vision missing rate')
    parser.add_argument('--seeds', type=str, default='111,1111,11111', help='comma-separated seeds')
    parser.add_argument('--run_tag', type=str, default='', help='optional suffix for grouping a batch of runs')
    parser.add_argument("--keep_ckpt", action="store_true", help="whether to keep checkpoint files after test")
    args = parser.parse_args()
    args.gpu_ids = parse_gpu_ids(args.gpu_ids)
    args.seeds = parse_seeds(args.seeds)
    args.missing_rate = resolve_missing_rate(args)
    return args


if __name__ == '__main__':
    args = parse_args()
    global logger
    logger = set_log(args)
    run_normal(args)
