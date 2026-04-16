import os
import gc
import time
import random
import logging
import torch
try:
    import pynvml
except ImportError:
    pynvml = None
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


def _sanitize_log_tag(s: str) -> str:
    out = []
    for c in (s or '').strip():
        if c.isalnum() or c in '-_.':
            out.append(c)
        else:
            out.append('_')
    t = ''.join(out).strip('_')[:200]
    return t if t else 'run'


def _ensure_run_slug(args):
    """与日志、checkpoint、结果 CSV 共用的运行标识；便于多任务对照。"""
    tag = (getattr(args, 'log_tag', '') or '').strip()
    if tag:
        args.run_slug = _sanitize_log_tag(tag)
    else:
        args.run_slug = f'{time.strftime("%Y%m%d-%H%M%S")}_{os.getpid()}'


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
    fc = getattr(args, 'fusion_center_modality', 'text')
    tct = getattr(args, 'text_corrupt_train', 0.0)
    tce = getattr(args, 'text_corrupt_eval', 0.0)
    mr = '-'.join(str(round(x, 4)) for x in getattr(args, 'missing_rate', (0.0, 0.0, 0.0)))
    rs = getattr(args, 'run_slug', '') or ''
    rs_part = f'-{rs}' if rs else ''
    args.model_save_path = os.path.join(
        args.model_save_dir,
        f'{args.modelName}-{args.datasetName}-{args.train_mode}-m{mr}-fc{fc}-tc{tct}-te{tce}{rs_part}.pth',
    )
    # indicate used gpu
    if len(args.gpu_ids) == 0 and torch.cuda.is_available() and pynvml is not None:
        # load free-most gpu
        pynvml.nvmlInit()
        device_count=pynvml.nvmlDeviceGetCount();
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
    args.res_save_dir = os.path.join(args.res_save_dir, 'normals')
    init_args = args
    model_results = []
    seeds = args.seeds
    missing_rate = '0-0-0'
    # run results
    for i, seed in enumerate(seeds):
        args = init_args
        # load config
        config = ConfigRegression(args)
        args = config.get_config()
        if i == 0 and args.data_missing:
            missing_rate = '-'.join(str(round(x, 4)) for x in args.missing_rate)
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
    corrupt_tag = '-tc{}-te{}-{}'.format(
        getattr(args, 'text_corrupt_train', 0.0),
        getattr(args, 'text_corrupt_eval', 0.0),
        getattr(args, 'text_corrupt_mode', 'mix')
    )
    fc = getattr(args, 'fusion_center_modality', 'text')
    rs = getattr(args, 'run_slug', '') or ''
    rs_part = f'-{rs}' if rs else ''
    save_path = os.path.join(
        args.res_save_dir,
        f'{args.datasetName}-{args.train_mode}-{missing_rate}-fc{fc}{corrupt_tag}{rs_part}.csv',
    )
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
    tag = (getattr(args, 'log_tag', '') or '').strip()
    slug = getattr(args, 'run_slug', None) or f'{time.strftime("%Y%m%d-%H%M%S")}_{os.getpid()}'
    if tag:
        stem = f'{args.modelName}-{args.datasetName}-{slug}'
    else:
        stem = f'{args.modelName}-{args.datasetName}_{slug}'
    log_file_path = os.path.join('logs', f'{stem}.log')
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
    args.log_file_path = os.path.abspath(log_file_path)
    logger.info('Log file: %s', args.log_file_path)
    return logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mode', type=str, default="regression",
                        help='regression')
    parser.add_argument('--modelName', type=str, default='mfmb_net',
                        help='support mfmb_net')
    parser.add_argument('--datasetName', type=str, default='mosi',
                        help='support mosi/mosei')
    parser.add_argument(
        '--log_tag',
        type=str,
        default='',
        help='运行标识：写入日志 / checkpoint / results/normals 汇总 CSV 的文件名；为空则自动 时间戳_PID，多任务互不覆盖。例: --log_tag dynamic_te04',
    )
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num workers of loading data')
    parser.add_argument('--model_save_dir', type=str, default='results/models',
                        help='path to save results.')
    parser.add_argument('--res_save_dir', type=str, default='results/results',
                        help='path to save results.')
    parser.add_argument('--gpu_ids', type=list, default=[],
                        help='indicates the gpus will be used. If none, the most-free gpu will be used!')
    parser.add_argument('--missing', type=float, default=0.0,
                        help='若未单独指定 missing_t/a/v，则三模态均使用该缺失率')
    parser.add_argument('--missing_t', type=float, default=None, help='文本缺失率，默认与 --missing 相同')
    parser.add_argument('--missing_a', type=float, default=None, help='音频缺失率')
    parser.add_argument('--missing_v', type=float, default=None, help='视频缺失率')
    parser.add_argument('--text_corrupt_train', type=float, default=0.0,
                        help='token corruption rate for train split text input')
    parser.add_argument('--text_corrupt_eval', type=float, default=0.0,
                        help='token corruption rate for valid/test split text input')
    parser.add_argument('--text_corrupt_mode', type=str, default='mix', choices=['none', 'token', 'span', 'mix'],
                        help='text corruption mode applied on input ids after missing simulation')
    parser.add_argument('--text_corrupt_span_frac', type=float, default=0.4,
                        help='fraction of corrupt tokens allocated to contiguous spans when using span/mix mode')
    parser.add_argument('--text_corrupt_seed', type=int, default=2026,
                        help='base seed for text corruption benchmark')
    parser.add_argument("--keep_ckpt", action="store_true", help="whether to keep checkpoint files after test")
    parser.add_argument(
        '--fusion_center_modality',
        type=str,
        default='text',
        choices=['text', 'audio', 'vision', 'dynamic'],
        help='Fusion hub: text=固定文本锚点(avt)；dynamic=按完整度/能量学习打分选锚点。',
    )
    parser.add_argument(
        '--use_distill',
        type=int,
        default=0,
        help='1=启用 clean-teacher / corrupted-student 蒸馏（与缺失+text_m 扰动配合）；0=关闭，保持旧实验。',
    )
    parser.add_argument(
        '--distill_teacher_detach',
        type=int,
        default=1,
        help='1=教师前向 torch.no_grad（默认）；0=允许梯度经共享权重回传教师分支（慎用）。',
    )
    parser.add_argument(
        '--distill_lambda_logit',
        type=float,
        default=0.1,
        help='最终预测 logit / 回归值蒸馏项权重（回归默认 MSE）；保守默认 0.1。',
    )
    parser.add_argument(
        '--distill_lambda_feat',
        type=float,
        default=0.1,
        help='融合表示 fused_rep 蒸馏（SmoothL1）权重；保守默认 0.1。',
    )
    parser.add_argument(
        '--distill_lambda_rel',
        type=float,
        default=0.0,
        help='batch 关系矩阵蒸馏权重；默认 0 关闭，仅保留接口。',
    )
    parser.add_argument(
        '--distill_temperature',
        type=float,
        default=2.0,
        help='分类任务 logit 蒸馏温度 T（回归忽略）；用于 KL 扩展。',
    )
    parser.add_argument(
        '--distill_clean_weight',
        type=float,
        default=0.0,
        help='全模态齐且文本无扰动样本的蒸馏权重；默认 0（仅困难样本参与蒸馏）。',
    )
    parser.add_argument(
        '--use_kd_logit',
        type=int,
        default=1,
        help='1=总损失中加入 logit 蒸馏项（仍受 distill_lambda_logit 缩放）；0=关闭。',
    )
    parser.add_argument(
        '--use_kd_feat',
        type=int,
        default=1,
        help='1=总损失中加入 fused_rep 蒸馏；0=关闭。',
    )
    parser.add_argument(
        '--use_kd_rel',
        type=int,
        default=0,
        help='1=总损失中加入关系矩阵蒸馏；默认 0；需同时设 distill_lambda_rel>0 才有实际梯度。',
    )
    parser.add_argument(
        '--online_text_corrupt',
        type=int,
        default=0,
        help='1=仅在 model 内对 text_m 做在线扰动（dataset 侧会跳过 corruption）；0=默认用 dataset 侧一次扰动，不在 model 内二次扰动。',
    )
    parser.add_argument(
        '--strict_original_loss',
        type=int,
        default=0,
        help='1=从第 1 个 epoch 起 pred+gen+kd 全上；0=默认 warmup：第 1 个 epoch 仅 pred+gen，第 2 个 epoch 起加 kd。',
    )
    parser.add_argument(
        '--return_fusion_aux',
        type=int,
        default=0,
        help='1=valid/test 前向额外返回 fusion aux（不写 loss）；用于调试。训练阶段由蒸馏自动取 aux。',
    )
    parser.add_argument(
        '--save_anchor_analysis',
        type=int,
        default=0,
        help='1=在指定 split 的 eval/test 上导出 teacher/student anchor 对齐 CSV（dynamic 诊断）；0=关闭且无额外开销。',
    )
    parser.add_argument(
        '--anchor_analysis_split',
        type=str,
        default='test',
        choices=['test', 'valid', 'both', 'all'],
        help='anchor 导出所跑的 dataloader：test / valid / both(=test+valid) / all(同 both)。',
    )
    parser.add_argument(
        '--anchor_analysis_dir',
        type=str,
        default='results/anchor_analysis',
        help='anchor analysis 样本级与汇总 CSV 的输出目录。',
    )
    parser.add_argument(
        '--distill_teacher_center_modality',
        type=str,
        default='same_as_student',
        choices=['same_as_student', 'text', 'audio', 'vision', 'dynamic'],
        help='蒸馏教师 fusion hub：same_as_student=与 fusion_center_modality 一致；否则仅教师前向临时覆盖（实验2：教师固定锚点+学生 dynamic）。',
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    mt = args.missing_t if args.missing_t is not None else args.missing
    ma = args.missing_a if args.missing_a is not None else args.missing
    mv = args.missing_v if args.missing_v is not None else args.missing
    args.missing_rate = tuple([mt, ma, mv])
    _ensure_run_slug(args)
    global logger; logger = set_log(args)
    args.seeds = [111, 1111, 11111]
    run_normal(args)
