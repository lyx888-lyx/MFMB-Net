"""
Save test-set predictions to CSV for offline metric recomputation.
Does not change model forward or loss.
"""
import os
import json
import logging

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger('MSA')

_DOM_NAMES = ('text', 'audio', 'vision')


def _dominant_labels_from_weights(weights_np):
    """weights_np: (N, 3) -> list[str] length N."""
    idx = np.argmax(weights_np, axis=1).astype(np.int64)
    return [_DOM_NAMES[i] for i in idx.reshape(-1)]


def save_anchor_center_summary(args, weights_np, valid_ratios_np, mode):
    """Run-level TEST summary: mean/std weights, dominant counts, mean valid ratios."""
    if mode != "TEST":
        return None
    if weights_np is None or len(weights_np) == 0:
        return None

    fc = getattr(args, 'fusion_center_modality', 'text')
    seed = getattr(args, 'seed', -1)
    ds = getattr(args, 'datasetName', 'data')
    out_dir = getattr(args, 'export_pred_dir', 'results/predictions')
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f'center_summary_{ds}_fc_{fc}_seed{seed}_test')

    n = int(weights_np.shape[0])
    dom_idx = np.argmax(weights_np, axis=1)
    c0, c1, c2 = int(np.sum(dom_idx == 0)), int(np.sum(dom_idx == 1)), int(np.sum(dom_idx == 2))

    summary = {
        'datasetName': ds,
        'fusion_center_modality': fc,
        'seed': int(seed) if seed is not None else None,
        'missing': float(getattr(args, 'missing', 0.0)),
        'text_missing_rate': float(getattr(args, 'text_missing_rate', getattr(args, 'missing', 0.0))),
        'audio_missing_rate': float(getattr(args, 'audio_missing_rate', getattr(args, 'missing', 0.0))),
        'vision_missing_rate': float(getattr(args, 'vision_missing_rate', getattr(args, 'missing', 0.0))),
        'n_test_samples': n,
        'mean_w_text': float(np.mean(weights_np[:, 0])),
        'mean_w_audio': float(np.mean(weights_np[:, 1])),
        'mean_w_vision': float(np.mean(weights_np[:, 2])),
        'std_w_text': float(np.std(weights_np[:, 0])),
        'std_w_audio': float(np.std(weights_np[:, 1])),
        'std_w_vision': float(np.std(weights_np[:, 2])),
        'dominant_text_count': c0,
        'dominant_audio_count': c1,
        'dominant_vision_count': c2,
        'dominant_text_ratio': float(c0 / n) if n else 0.0,
        'dominant_audio_ratio': float(c1 / n) if n else 0.0,
        'dominant_vision_ratio': float(c2 / n) if n else 0.0,
        'mean_valid_ratio_text': float(np.nanmean(valid_ratios_np[:, 0])),
        'mean_valid_ratio_audio': float(np.nanmean(valid_ratios_np[:, 1])),
        'mean_valid_ratio_vision': float(np.nanmean(valid_ratios_np[:, 2])),
    }
    if fc == 'dynamic_missing':
        summary['anchor_router_type'] = getattr(args, 'anchor_router_type', 'rule')
        summary['anchor_router_hidden'] = int(getattr(args, 'anchor_router_hidden', 16))
        summary['anchor_router_use_prior'] = int(getattr(args, 'anchor_router_use_prior', 1))
        summary['anchor_router_temperature'] = float(getattr(args, 'anchor_router_temperature', 1.0))

    json_path = stem + '.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    logger.info('Saved anchor center summary (json) to %s', json_path)

    csv_path = stem + '.csv'
    pd.DataFrame([summary]).to_csv(csv_path, index=False)
    logger.info('Saved anchor center summary (csv) to %s', csv_path)

    logger.info(
        '[ANCHOR_SUMMARY][TEST] mean soft weights: text=%.4f audio=%.4f vision=%.4f | '
        'dominant center ratio: text=%.4f audio=%.4f vision=%.4f',
        summary['mean_w_text'],
        summary['mean_w_audio'],
        summary['mean_w_vision'],
        summary['dominant_text_ratio'],
        summary['dominant_audio_ratio'],
        summary['dominant_vision_ratio'],
    )
    return json_path


def save_test_predictions_csv(
    args,
    pred_1d,
    true_1d,
    indices,
    ids,
    mode,
    anchor_weights=None,
    anchor_valid_ratios=None,
):
    """pred/true: 1D numpy or torch; indices: per-row dataset index; ids: same length.

    anchor_weights: optional (N, 3) float32 — w_text, w_audio, w_vision per row (same order as pred).
    anchor_valid_ratios: optional (N, 3) for valid_ratio_* columns.
    """
    if mode != "TEST":
        return None
    if not getattr(args, 'export_test_predictions', False):
        return None

    fc = getattr(args, 'fusion_center_modality', 'text')
    seed = getattr(args, 'seed', -1)
    ds = getattr(args, 'datasetName', 'data')

    if isinstance(pred_1d, torch.Tensor):
        pred_1d = pred_1d.detach().cpu().numpy().reshape(-1)
    else:
        pred_1d = np.asarray(pred_1d).reshape(-1)
    if isinstance(true_1d, torch.Tensor):
        true_1d = true_1d.detach().cpu().numpy().reshape(-1)
    else:
        true_1d = np.asarray(true_1d).reshape(-1)

    n = len(pred_1d)
    idx_arr = np.asarray(indices).reshape(-1)
    if len(idx_arr) != n or len(ids) != n:
        logger.warning(
            'export length mismatch: pred=%d indices=%d ids=%d — skip export',
            n,
            len(idx_arr),
            len(ids),
        )
        return None

    sort_order = np.argsort(idx_arr, kind='mergesort')
    pred_s = pred_1d[sort_order]
    true_s = true_1d[sort_order]
    idx_s = idx_arr[sort_order]
    ids_seq = list(ids) if not isinstance(ids, np.ndarray) else ids.reshape(-1).tolist()
    id_s = [str(ids_seq[sort_order[j]]) for j in range(n)]

    pred_clip3 = np.clip(pred_s, -3.0, 3.0)
    true_clip3 = np.clip(true_s, -3.0, 3.0)
    pred_clip2 = np.clip(pred_s, -2.0, 2.0)
    true_clip2 = np.clip(true_s, -2.0, 2.0)

    out_dir = getattr(args, 'export_pred_dir', 'results/predictions')
    os.makedirs(out_dir, exist_ok=True)
    fname = f"predictions_{ds}_fc_{fc}_seed{seed}_test_full_drop_last_false.csv"
    path = os.path.join(out_dir, fname)

    df_dict = {
        'row_in_export_order': np.arange(n),
        'dataset_index': idx_s,
        'sample_id': id_s,
        'truth': true_s,
        'pred_raw': pred_s,
        'pred_clip_m3_p3': pred_clip3,
        'truth_clip_m3_p3': true_clip3,
        'pred_clip_m2_p2': pred_clip2,
        'truth_clip_m2_p2': true_clip2,
        'pred_round_acc7': np.round(pred_clip3),
        'truth_round_acc7': np.round(true_clip3),
        'pred_round_acc5': np.round(pred_clip2),
        'truth_round_acc5': np.round(true_clip2),
        'seed': seed,
        'datasetName': ds,
        'fusion_center_modality': fc,
    }

    anchor_cols_ok = False
    if anchor_weights is not None:
        aw = np.asarray(anchor_weights, dtype=np.float64)
        if aw.shape[0] != n or aw.shape[1] != 3:
            logger.warning(
                'anchor_weights shape %s incompatible with n=%d — skip anchor columns',
                aw.shape,
                n,
            )
        else:
            aw = aw[sort_order]
            df_dict['w_text'] = aw[:, 0]
            df_dict['w_audio'] = aw[:, 1]
            df_dict['w_vision'] = aw[:, 2]
            df_dict['dominant_center'] = _dominant_labels_from_weights(aw.astype(np.float64))
            anchor_cols_ok = True

    if anchor_cols_ok and anchor_valid_ratios is not None:
        vr = np.asarray(anchor_valid_ratios, dtype=np.float64)
        if vr.shape[0] == n and vr.shape[1] == 3:
            vr = vr[sort_order]
            df_dict['valid_ratio_text'] = vr[:, 0]
            df_dict['valid_ratio_audio'] = vr[:, 1]
            df_dict['valid_ratio_vision'] = vr[:, 2]

    df = pd.DataFrame(df_dict)
    df.to_csv(path, index=False)
    logger.info('Exported test predictions to %s (n=%d)', path, len(df))

    meta = {
        'csv_path': os.path.abspath(path),
        'n_samples': int(len(df)),
        'datasetName': ds,
        'seed': int(seed) if seed is not None else None,
        'fusion_center_modality': fc,
        'missing': float(getattr(args, 'missing', 0.0)),
        'text_missing_rate': float(getattr(args, 'text_missing_rate', getattr(args, 'missing', 0.0))),
        'audio_missing_rate': float(getattr(args, 'audio_missing_rate', getattr(args, 'missing', 0.0))),
        'vision_missing_rate': float(getattr(args, 'vision_missing_rate', getattr(args, 'missing', 0.0))),
        'dataPath_runtime': getattr(args, 'dataPath', ''),
        'dataloader_note': (
            'Test DataLoader: shuffle=False, drop_last=False — full test split in metrics/export.'
        ),
        'export_tag': 'full_drop_last_false',
        'anchor_router_type': (
            getattr(args, 'anchor_router_type', 'rule') if fc == 'dynamic_missing' else None
        ),
        'anchor_columns': [
            c
            for c in (
                'w_text',
                'w_audio',
                'w_vision',
                'dominant_center',
                'valid_ratio_text',
                'valid_ratio_audio',
                'valid_ratio_vision',
            )
            if c in df_dict
        ],
    }
    meta_path = path.replace('.csv', '_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    return path
