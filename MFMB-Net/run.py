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

from models.AMIO import AMIO
from trains.ATIO import ATIO
from data.load_data import MMDataLoader
from config.config_regression import ConfigRegression

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

"""
总入口：
    负责：
    解析命令行参数
    加载配置
    自动选 GPU
    构建 dataloader
    构建模型
    调 trainer 训练
    训练完加载最佳权重
    在 test 上评估
    多个 seed 跑完后，把均值和方差写进 csv

核心链路：
    run.py
    -> config
    -> data/load_data.py
    -> models/AMIO.py
    -> models/missingTask/MFMB_NET/model.py
    -> alignment + generator + fusion
    -> trains/missingTask/MFMB_NET.py
    -> metricsTop.py

    for m in 0.0 0.1 0.2 0.3 0.4 0.5
    do
    CUDA_VISIBLE_DEVICES=0 python run.py \
        --datasetName mosei \
        --missing $m \
        --gpu_ids 0 \
        --run_tag half1
    done
"""


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def format_missing(missing: float) -> str:
    # 统一文件命名格式，比如 0.0 / 0.5 / 1.0
    return f"{missing:.1f}"


def build_run_name(args, seed=None):
    """
    给每个实验一个独立名字，用于：
    - log 文件
    - checkpoint 文件
    """
    miss = format_missing(args.missing)
    parts = [args.modelName, args.datasetName, args.train_mode, f"m{miss}"]
    if args.run_tag:
        parts.append(args.run_tag)
    if seed is not None:
        parts.append(f"seed{seed}")
    return "-".join(parts)


def pick_gpu_if_needed(args):
    """
    如果没有显式指定 --gpu_ids，就自动选当前最空闲的 GPU
    """
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

        print(f"Find gpu: {dst_gpu_id}, used memory: {min_mem_used}")
        logger.info(f"Find gpu: {dst_gpu_id}, used memory: {min_mem_used}")
        args.gpu_ids = [dst_gpu_id]


def run(args):
    # 每个 seed 单独一个 checkpoint，避免并行时互相覆盖
    os.makedirs(args.model_save_dir, exist_ok=True)
    ckpt_name = f"{build_run_name(args, seed=args.seed)}.pth"
    args.model_save_path = os.path.join(args.model_save_dir, ckpt_name)

    pick_gpu_if_needed(args)

    using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
    logger.info("Let's use %d GPUs!" % len(args.gpu_ids))
    device = torch.device(f"cuda:{int(args.gpu_ids[0])}" if using_cuda else "cpu")
    args.device = device

    # load data and model
    dataloader = MMDataLoader(args)
    model = AMIO(args).to(device)

    def count_parameters(model):
        total = 0
        for p in model.parameters():
            if p.requires_grad:
                total += p.numel()
        return total

    logger.info(f"The model has {count_parameters(model)} trainable parameters")

    atio = ATIO().getTrain(args)

    # train
    atio.do_train(model, dataloader)

    # load best ckpt
    assert os.path.exists(args.model_save_path), f"Checkpoint not found: {args.model_save_path}"
    model.load_state_dict(torch.load(args.model_save_path, map_location=device))
    model.to(device)

    # do test
    results = atio.do_test(model, dataloader["test"], mode="TEST")

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
    time.sleep(3)

    return results


def run_normal(args):
    args.res_save_dir = os.path.join(args.res_save_dir, "normals")
    os.makedirs(args.res_save_dir, exist_ok=True)

    init_args = args
    model_results = []
    seeds = args.seeds
    missing_rate = 0.0

    for i, seed in enumerate(seeds):
        args = init_args

        # load config
        config = ConfigRegression(args)
        args = config.get_config()

        if i == 0 and args.data_missing:
            missing_rate = str(args.missing_rate[0])

        setup_seed(seed)
        args.seed = seed
        args.cur_time = i + 1

        logger.info("Start running %s..." % args.modelName)
        logger.info(args)

        test_results = run(args)
        model_results.append(test_results)

    criterions = list(model_results[0].keys())

    # 这里保持原来的结果文件命名，不影响你后面 AUILC 统计脚本
    save_path = os.path.join(
        args.res_save_dir,
        f"{args.datasetName}-{args.train_mode}-{missing_rate}.csv"
    )

    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame(columns=["Model"] + criterions)

    res = [args.modelName]
    for c in criterions:
        values = [r[c] for r in model_results]
        mean = round(np.mean(values) * 100, 2)
        std = round(np.std(values) * 100, 2)
        res.append((mean, std))

    df.loc[len(df)] = res
    df.to_csv(save_path, index=None)
    logger.info("Results are added to %s..." % save_path)


def set_log(args):
    os.makedirs("logs", exist_ok=True)

    log_name = f"{build_run_name(args)}.log"
    log_file_path = os.path.join("logs", log_name)

    # 直接配置 root logger，这样 trainer 里的 logging.getLogger() 也能写进来
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # 清掉旧 handler，避免重复输出
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    formatter = logging.Formatter(
        "%(asctime)s:%(levelname)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 文件日志
    fh = logging.FileHandler(log_file_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)

    # 终端日志
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    root_logger.addHandler(sh)

    return root_logger


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_mode", type=str, default="regression", help="regression")
    parser.add_argument("--modelName", type=str, default="mfmb_net", help="support mfmb_net")
    parser.add_argument("--datasetName", type=str, default="mosi", help="support mosi/mosei")
    parser.add_argument("--num_workers", type=int, default=0, help="num workers of loading data")
    parser.add_argument("--model_save_dir", type=str, default="results/models", help="path to save models")
    parser.add_argument("--res_save_dir", type=str, default="results/results", help="path to save csv results")
    parser.add_argument("--keep_ckpt", action="store_true", help="whether to keep checkpoint files after test")

    # 改成 CLI 友好的写法：--gpu_ids 0
    parser.add_argument(
        "--gpu_ids",
        type=int,
        nargs="*",
        default=[],
        help="GPU ids, e.g. --gpu_ids 0"
    )

    parser.add_argument("--missing", type=float, default=0.0)

    # 新增：给并行终端用的标记，避免日志/ckpt撞名
    parser.add_argument(
        "--run_tag",
        type=str,
        default="",
        help="optional tag for parallel runs, e.g. half1 / half2"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.missing_rate = tuple([args.missing, args.missing, args.missing])

    global logger
    logger = set_log(args)

    args.seeds = [111, 1111, 11111]
    run_normal(args)