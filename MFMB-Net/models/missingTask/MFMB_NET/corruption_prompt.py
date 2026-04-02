"""
Corruption / missing condition embeddings（离散标签 + mask 推断统计量）。

================================================================================
1) missing-type（per-sample id 0..4，名称固定）
================================================================================
  id 0: none   — 三模态缺失率均不超过 missing_pattern_thresh（默认 0.05）
  id 1: text   — 仅文本缺失率超阈
  id 2: audio  — 仅音频缺失率超阈
  id 3: vision — 仅视频缺失率超阈
  id 4: multi  — 至少两路超阈

================================================================================
2) missing-rate bucket（由 r_max = max(r_text, r_audio, r_vision) 映射，阈值写死）
================================================================================
  id 0: none — r_max <= RATE_EPS（默认 1e-6，近似无缺失）
  id 1: low  — (RATE_EPS, 0.3]
  id 2: mid  — (0.3, 0.7]
  id 3: high — (0.7, 1.0]

================================================================================
3) noise-type（id 0..6；音频/视频噪声未接入时用 none/unknown 占位，便于后续扩展）
================================================================================
  id 0: none
  id 1: occlusion（与 span 类扰动对齐）
  id 2: temporal_gap（预留）
  id 3: white_noise（预留）
  id 4: token_unk（与 token 类扰动对齐）
  id 5: mix
  id 6: unknown（未识别 mode 或占位）

text_corrupt_mode（utils/text_corrupt.py）→ noise_type_id 见 TEXT_CORRUPT_TO_ID。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any

# ---------------------------------------------------------------------------
# 固定词汇与阈值（实验记录时请引用本节）
# ---------------------------------------------------------------------------
MISSING_TYPE_NAMES: Tuple[str, ...] = ("none", "text", "audio", "vision", "multi")

# r_t,r_a,r_v 超过该阈值则视为「该模态存在缺失」用于 missing-type 判定
MISSING_PATTERN_THRESH = 0.05

RATE_EPS = 1e-6
RATE_LOW_BOUND = 0.3
RATE_MID_BOUND = 0.7

RATE_BUCKET_NAMES: Tuple[str, ...] = ("none", "low", "mid", "high")

NOISE_TYPE_NAMES: Tuple[str, ...] = (
    "none",
    "occlusion",
    "temporal_gap",
    "white_noise",
    "token_unk",
    "mix",
    "unknown",
)

NUM_MISSING_TYPE = len(MISSING_TYPE_NAMES)
NUM_RATE_BUCKET = len(RATE_BUCKET_NAMES)
NUM_NOISE_TYPE = len(NOISE_TYPE_NAMES)

# 与 argparse text_corrupt_mode choices：none | token | span | mix 对齐；其余为扩展名
TEXT_CORRUPT_TO_ID = {
    "none": 0,
    "token": 4,
    "span": 1,
    "mix": 5,
    "occlusion": 1,
    "temporal_gap": 2,
    "white_noise": 3,
    "token_unk": 4,
    "unknown": 6,
}


def infer_missing_rates(text_mm, audio_mm, vision_mm, text_mask, audio_mask, vision_mask):
    """各模态缺失比例 [B]；missing_mask 中 1=保留，0=缺失。"""
    eps = 1e-6

    def _rate(mm, vm):
        valid = vm.float().clamp(0, 1)
        present = (mm.float() * valid).sum(dim=1)
        denom = valid.sum(dim=1).clamp(min=eps)
        return 1.0 - (present / denom)

    return _rate(text_mm, text_mask), _rate(audio_mm, audio_mask), _rate(vision_mm, vision_mask)


def rates_to_bucket_id(rate: torch.Tensor, eps: float = RATE_EPS) -> torch.Tensor:
    """
    将标量缺失率 r（此处为 r_max）映射到 bucket id 0..3，规则见模块文档。
    """
    bid = torch.zeros_like(rate, dtype=torch.long)
    bid = torch.where(rate > eps, torch.ones_like(bid), bid)
    bid = torch.where(rate > RATE_LOW_BOUND, torch.full_like(bid, 2), bid)
    bid = torch.where(rate > RATE_MID_BOUND, torch.full_like(bid, 3), bid)
    return bid


def infer_missing_type_id(
    rt: torch.Tensor,
    ra: torch.Tensor,
    rv: torch.Tensor,
    thresh: float = MISSING_PATTERN_THRESH,
) -> torch.Tensor:
    t = (rt > thresh).long()
    a = (ra > thresh).long()
    v = (rv > thresh).long()
    cnt = t + a + v
    out = torch.zeros_like(cnt)
    out = torch.where(cnt >= 2, torch.full_like(out, 4), out)
    single = cnt == 1
    out = torch.where(single & (t == 1), torch.ones_like(out), out)
    out = torch.where(single & (a == 1), torch.full_like(out, 2), out)
    out = torch.where(single & (v == 1), torch.full_like(out, 3), out)
    return out


def _id_list_to_names(ids: torch.Tensor, table: Tuple[str, ...]) -> List[str]:
    return [table[int(i)] for i in ids.detach().cpu().view(-1).tolist()]


def build_prompt_debug_dict(
    mt: torch.Tensor,
    rb: torch.Tensor,
    noise_ids: torch.Tensor,
    rt: torch.Tensor,
    ra: torch.Tensor,
    rv: torch.Tensor,
    rmax: torch.Tensor,
) -> Dict[str, Any]:
    """供 fusion return_aux / 日志使用；含 id 与可读 name。"""
    return {
        "missing_type_id": mt,
        "missing_type_name": _id_list_to_names(mt, MISSING_TYPE_NAMES),
        "rate_bucket_id": rb,
        "rate_bucket_name": _id_list_to_names(rb, RATE_BUCKET_NAMES),
        "noise_type_id": noise_ids,
        "noise_type_name": _id_list_to_names(noise_ids, NOISE_TYPE_NAMES),
        "missing_rate_text": rt,
        "missing_rate_audio": ra,
        "missing_rate_vision": rv,
        "missing_rate_max": rmax,
    }


class CorruptionPromptEncoder(nn.Module):
    def __init__(self, out_dim, dropout=0.0):
        super().__init__()
        self.emb_mt = nn.Embedding(NUM_MISSING_TYPE, out_dim)
        self.emb_rb = nn.Embedding(NUM_RATE_BUCKET, out_dim)
        self.emb_nt = nn.Embedding(NUM_NOISE_TYPE, out_dim)
        self.fuse = nn.Sequential(
            nn.Linear(out_dim * 3, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward_from_tensors(
        self,
        text_mm,
        audio_mm,
        vision_mm,
        text_mask,
        audio_mask,
        vision_mask,
        noise_type_ids,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        rt, ra, rv = infer_missing_rates(text_mm, audio_mm, vision_mm, text_mask, audio_mask, vision_mask)
        rmax = torch.max(torch.max(rt, ra), rv)
        rb = rates_to_bucket_id(rmax)
        mt = infer_missing_type_id(rt, ra, rv)
        e = torch.cat(
            [
                self.emb_mt(mt),
                self.emb_rb(rb),
                self.emb_nt(noise_type_ids.clamp(0, NUM_NOISE_TYPE - 1)),
            ],
            dim=-1,
        )
        dbg = build_prompt_debug_dict(mt, rb, noise_type_ids, rt, ra, rv, rmax)
        return self.fuse(e), dbg

    @staticmethod
    def noise_id_from_args(args) -> int:
        mode = args.get("text_corrupt_mode", "none") if hasattr(args, "get") else getattr(args, "text_corrupt_mode", "none")
        if not isinstance(mode, str):
            return int(mode) if mode is not None else TEXT_CORRUPT_TO_ID["unknown"]
        return TEXT_CORRUPT_TO_ID.get(mode.lower().strip(), TEXT_CORRUPT_TO_ID["unknown"])
