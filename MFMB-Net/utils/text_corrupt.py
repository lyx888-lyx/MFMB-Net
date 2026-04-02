"""
BERT 输入级文本扰动：在送入 BertTextEncoder 之前改写 input_ids（[MASK] / 随机词 / 连续 span）。

用于与「固定文本锚点 vs 动态锚点」对比，在文本不可靠时观察融合是否更鲁棒。
"""
import torch
from typing import Tuple

# bert-base-uncased 常用 id（与 MMSA 预训练 BERT 一致）
MASK_ID = 103
CLS_ID = 101
SEP_ID = 102


def _valid_positions(ids_row: torch.Tensor, attn_row: torch.Tensor) -> list[int]:
    """可扰动位置：非 padding，且跳过 CLS、SEP。"""
    valid = (attn_row > 0).nonzero(as_tuple=True)[0]
    out = []
    for i in valid.tolist():
        tid = int(ids_row[i].item())
        if i == 0 and tid == CLS_ID:
            continue
        if tid in (CLS_ID, SEP_ID):
            continue
        out.append(i)
    return out


def _corrupt_one_row(
    ids: torch.Tensor,
    attn: torch.Tensor,
    rate: float,
    mode: str,
    span_frac: float,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    """ids: [L] long, attn: [L] float"""
    out = ids.clone()
    pos = _valid_positions(out, attn)
    if len(pos) < 2:
        return out
    n = max(1, int(len(pos) * rate))
    n = min(n, len(pos))

    if mode == "token":
        pick = torch.randperm(len(pos), device=device)[:n]
        idx = torch.tensor([pos[i] for i in pick.tolist()], device=device, dtype=torch.long)
        rnd = torch.randint(0, vocab_size, (idx.numel(),), device=device)
        out[idx] = rnd
    elif mode == "span":
        span_len = max(1, min(n, len(pos)))
        start = torch.randint(0, max(1, len(pos) - span_len + 1), (1,), device=device).item()
        for k in range(span_len):
            j = pos[start + k]
            out[j] = MASK_ID
    else:  # mix: span 段 + 随机 token，预算各一半（由 span_frac 微调）
        n_span = max(0, int(n * span_frac))
        n_tok = max(0, n - n_span)
        used = set()
        if n_span > 0 and len(pos) >= 2:
            span_len = max(1, min(n_span, len(pos)))
            start = torch.randint(0, max(1, len(pos) - span_len + 1), (1,), device=device).item()
            for k in range(span_len):
                j = pos[start + k]
                out[j] = MASK_ID
                used.add(j)
        if n_tok > 0:
            remaining = [p for p in pos if p not in used]
            if remaining:
                m = min(n_tok, len(remaining))
                pick = torch.randperm(len(remaining), device=device)[:m]
                idx = torch.tensor([remaining[i] for i in pick.tolist()], device=device, dtype=torch.long)
                rnd = torch.randint(0, vocab_size, (idx.numel(),), device=device)
                out[idx] = rnd
    return out


def apply_text_corruption(
    text: torch.Tensor,
    *,
    corrupt_rate: float,
    mode: str,
    span_frac: float,
    vocab_size: int = 30522,
) -> torch.Tensor:
    """
    text: [B, 3, L] — input_ids / attention_mask / segment_ids（与 BertTextEncoder 一致）
    corrupt_rate: 扰动强度（约等于被替换 token 比例）
    """
    if corrupt_rate <= 0 or mode == "none":
        return text
    dtype = text.dtype
    device = text.device
    B, _, L = text.shape
    out = text.clone()
    ids = text[:, 0, :].long()
    attn = text[:, 1, :].float()
    seg = text[:, 2, :]

    for b in range(B):
        out[b, 0] = _corrupt_one_row(
            ids[b], attn[b], corrupt_rate, mode, span_frac, vocab_size, device
        ).float()

    out[:, 1] = attn
    out[:, 2] = seg.float() if seg.dtype != torch.float else seg
    return out


def maybe_corrupt_text_pair(
    text: torch.Tensor,
    text_m: torch.Tensor,
    *,
    training: bool,
    text_corrupt_train: float,
    text_corrupt_eval: float,
    text_corrupt_mode: str,
    text_corrupt_span_frac: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """训练用 text_corrupt_train，验证/测试用 text_corrupt_eval；**同时**扰动 clean 与 missing 支路。"""
    rate = text_corrupt_train if training else text_corrupt_eval
    if rate <= 0 or text_corrupt_mode == "none":
        return text, text_m
    text = apply_text_corruption(
        text, corrupt_rate=rate, mode=text_corrupt_mode, span_frac=text_corrupt_span_frac
    )
    text_m = apply_text_corruption(
        text_m, corrupt_rate=rate, mode=text_corrupt_mode, span_frac=text_corrupt_span_frac
    )
    return text, text_m


def maybe_corrupt_text_m_only(
    text: torch.Tensor,
    text_m: torch.Tensor,
    *,
    training: bool,
    text_corrupt_train: float,
    text_corrupt_eval: float,
    text_corrupt_mode: str,
    text_corrupt_span_frac: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    仅对 **text_m**（缺失/学生支路）做输入级扰动，**text** 保持数据集原始 clean，供蒸馏教师与文本重建目标一致。
    """
    rate = text_corrupt_train if training else text_corrupt_eval
    if rate <= 0 or text_corrupt_mode == "none":
        return text, text_m
    text_m = apply_text_corruption(
        text_m, corrupt_rate=rate, mode=text_corrupt_mode, span_frac=text_corrupt_span_frac
    )
    return text, text_m
