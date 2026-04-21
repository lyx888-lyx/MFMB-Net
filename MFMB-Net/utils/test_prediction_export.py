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


def save_test_predictions_csv(args, pred_1d, true_1d, indices, ids, mode):
    """pred/true: 1D numpy or torch; indices: per-row dataset index; ids: same length."""
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
    if len(indices) != n or len(ids) != n:
        logger.warning(
            'export length mismatch: pred=%d indices=%d ids=%d — skip export',
            n, len(indices), len(ids),
        )
        return None

    pred_clip3 = np.clip(pred_1d, -3.0, 3.0)
    true_clip3 = np.clip(true_1d, -3.0, 3.0)
    pred_clip2 = np.clip(pred_1d, -2.0, 2.0)
    true_clip2 = np.clip(true_1d, -2.0, 2.0)

    out_dir = getattr(args, 'export_pred_dir', 'results/predictions')
    os.makedirs(out_dir, exist_ok=True)
    # Distinct from legacy exports when test used drop_last=True (fewer rows).
    fname = f"predictions_{ds}_fc_{fc}_seed{seed}_test_full_drop_last_false.csv"
    path = os.path.join(out_dir, fname)

    # ids may be numpy scalars or str
    id_col = []
    for i in range(len(pred_1d)):
        try:
            id_col.append(str(ids[i]))
        except Exception:
            id_col.append('')

    df = pd.DataFrame({
        'row_in_export_order': np.arange(len(pred_1d)),
        'dataset_index': np.asarray(indices).reshape(-1),
        'sample_id': id_col,
        'truth': true_1d,
        'pred_raw': pred_1d,
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
    })
    # Stable order: sort by dataset index (DataLoader may shuffle test batches)
    df = df.sort_values('dataset_index').reset_index(drop=True)
    df.to_csv(path, index=False)
    logger.info('Exported test predictions to %s (n=%d)', path, len(df))

    meta = {
        'csv_path': os.path.abspath(path),
        'n_samples': int(len(df)),
        'datasetName': ds,
        'seed': int(seed) if seed is not None else None,
        'fusion_center_modality': fc,
        'dataPath_runtime': getattr(args, 'dataPath', ''),
        'dataloader_note': (
            'Test DataLoader: shuffle=False, drop_last=False — full test split in metrics/export.'
        ),
        'export_tag': 'full_drop_last_false',
    }
    meta_path = path.replace('.csv', '_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    return path

