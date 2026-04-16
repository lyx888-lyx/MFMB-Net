import os
import csv
import time
import logging
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.functions import dict_to_str
from utils.metricsTop import MetricsTop

logger = logging.getLogger('MSA')

_ANCHOR_IDX_TO_NAME = ('text', 'audio', 'vision')


def _anchor_logits_row_str(anchor_logits: torch.Tensor, batch_i: int) -> str:
    if anchor_logits is None:
        return ''
    row = anchor_logits[batch_i].detach().float().cpu().numpy().reshape(-1)
    return ','.join(f'{float(x):.8g}' for x in row)


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2 or y.size < 2:
        return float('nan')
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


class MFMB_NET():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.datasetName)

    def do_train(self, model, dataloader):
        # 1. 暴力拦截：不管前面谁把 device 改成了 cpu，在这里统统强制改回 cuda！
        self.args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 加上这一行，双保险：确保模型先去 GPU！
        model = model.to(self.args.device)
        # 3. 打印确认
        # print(f"============ 暴力修改后，当前使用的设备是: {self.args.device} ============")
        if self.args.use_bert_finetune:
            bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            bert_params = list(model.Model.text_model.named_parameters())

            bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
            bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
            model_params_other = [p for n, p in list(model.named_parameters()) if 'text_model' not in n]

            optimizer_grouped_parameters = [
                {'params': bert_params_decay, 'weight_decay': self.args.weight_decay_bert, 'lr': self.args.learning_rate_bert},
                {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.args.learning_rate_bert},
                {'params': model_params_other, 'weight_decay': self.args.weight_decay_other, 'lr': self.args.learning_rate_other}
            ]
            optimizer = optim.Adam(optimizer_grouped_parameters)
        else:
            optimizer = optim.Adam(model.parameters(), lr=self.args.learning_rate_other, weight_decay=self.args.weight_decay_other)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, verbose=True, patience=self.args.patience)

        ll = float(getattr(self.args, 'distill_lambda_logit', 0.0))
        lf = float(getattr(self.args, 'distill_lambda_feat', 0.0))
        lr = float(getattr(self.args, 'distill_lambda_rel', 0.0))
        uk_logit = int(getattr(self.args, 'use_kd_logit', 0)) != 0
        uk_feat = int(getattr(self.args, 'use_kd_feat', 0)) != 0
        uk_rel = int(getattr(self.args, 'use_kd_rel', 0)) != 0
        strict_orig = int(getattr(self.args, 'strict_original_loss', 0)) != 0

        epochs, best_epoch = 0, 0
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        while True:  
            epochs += 1
            y_pred, y_true = [], []
            losses = []
            model.train()
            train_loss, predict_loss, generate_loss = 0.0, 0.0, 0.0
            kd_logit_raw_acc = kd_feat_raw_acc = kd_rel_raw_acc = 0.0
            kd_logit_w_acc = kd_feat_w_acc = kd_rel_w_acc = 0.0
            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()

                    text = batch_data['text'].to(self.args.device)
                    text_m = batch_data['text_m'].to(self.args.device)
                    text_missing_mask = batch_data['text_missing_mask'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    audio_m = batch_data['audio_m'].to(self.args.device)
                    audio_mask = batch_data['audio_mask'].to(self.args.device)
                    audio_missing_mask = batch_data['audio_missing_mask'].to(self.args.device)
                    vision = batch_data['vision'].to(self.args.device)
                    vision_m = batch_data['vision_m'].to(self.args.device)
                    vision_mask = batch_data['vision_mask'].to(self.args.device)
                    vision_missing_mask = batch_data['vision_missing_mask'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    text_corrupt_mask = batch_data.get('text_corrupt_mask')
                    if text_corrupt_mask is not None:
                        text_corrupt_mask = text_corrupt_mask.to(self.args.device)

                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)

                    model_out = model(
                        (text, text_m, text_missing_mask),
                        (audio, audio_m, audio_mask, audio_missing_mask),
                        (vision, vision_m, vision_mask, vision_missing_mask),
                        text_corrupt_mask=text_corrupt_mask,
                    )
                    if len(model_out) == 3:
                        prediction, gen_loss, extra = model_out
                        kd_logit = extra.get('kd_logit', torch.zeros((), device=prediction.device))
                        kd_feat = extra.get('kd_feat', torch.zeros((), device=prediction.device))
                        kd_rel = extra.get('kd_rel', torch.zeros((), device=prediction.device))
                    else:
                        prediction, gen_loss = model_out
                        kd_logit = kd_feat = kd_rel = torch.zeros((), device=prediction.device)

                    pred_loss = self.criterion(prediction, labels)
                    base_loss = pred_loss + gen_loss
                    kd_logit_w = (ll * kd_logit) if uk_logit else torch.zeros((), device=prediction.device, dtype=pred_loss.dtype)
                    kd_feat_w = (lf * kd_feat) if uk_feat else torch.zeros((), device=prediction.device, dtype=pred_loss.dtype)
                    kd_rel_w = (lr * kd_rel) if uk_rel else torch.zeros((), device=prediction.device, dtype=pred_loss.dtype)
                    kd_total = kd_logit_w + kd_feat_w + kd_rel_w
                    # strict_original_loss=1：首 epoch 即加 kd；默认 warmup：epoch 1 仅 pred+gen
                    use_kd_this_epoch = strict_orig or (epochs > 1)
                    loss = base_loss + (kd_total if use_kd_this_epoch else torch.zeros((), device=prediction.device, dtype=pred_loss.dtype))

                    loss.backward()

                    if self.args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], self.args.grad_clip)

                    left_epochs -= 1
                    if left_epochs == 0:
                        optimizer.step()
                        left_epochs = self.args.update_epochs

                    train_loss += loss.item()
                    predict_loss += pred_loss.item()
                    generate_loss += gen_loss.item()
                    kd_logit_raw_acc += float(kd_logit.detach().item())
                    kd_feat_raw_acc += float(kd_feat.detach().item())
                    kd_rel_raw_acc += float(kd_rel.detach().item())
                    kd_logit_w_acc += float(kd_logit_w.detach().item())
                    kd_feat_w_acc += float(kd_feat_w.detach().item())
                    kd_rel_w_acc += float(kd_rel_w.detach().item())

                    y_pred.append(prediction.cpu())
                    y_true.append(labels.cpu())

                if left_epochs != self.args.update_epochs:
                    optimizer.step()

            n_batch = len(dataloader['train'])
            train_loss = train_loss / n_batch
            predict_loss = predict_loss / n_batch
            generate_loss = generate_loss / n_batch
            kd_logit_raw_acc /= n_batch
            kd_feat_raw_acc /= n_batch
            kd_rel_raw_acc /= n_batch
            kd_logit_w_acc /= n_batch
            kd_feat_w_acc /= n_batch
            kd_rel_w_acc /= n_batch
            
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                "TRAIN-(%s) (%d/%d/%d)>> total_loss: %.4f | pred_loss: %.4f | gen_loss: %.4f | "
                "kd_logit_raw: %.6f | kd_feat_raw: %.6f | kd_rel_raw: %.6f | "
                "kd_logit_weighted: %.6f | kd_feat_weighted: %.6f | kd_rel_weighted: %.6f | %s"
                % (
                    self.args.modelName,
                    epochs - best_epoch, epochs, self.args.cur_time,
                    train_loss, predict_loss, generate_loss,
                    kd_logit_raw_acc, kd_feat_raw_acc, kd_rel_raw_acc,
                    kd_logit_w_acc, kd_feat_w_acc, kd_rel_w_acc,
                    dict_to_str(train_results),
                )
            )
            
            val_results = self.do_test(model, dataloader['valid'], mode="VAL", anchor_epoch=epochs)
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])

            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)
                
            if epochs - best_epoch >= self.args.early_stop:
                return

    def do_test(self, model, dataloader, mode="VAL", anchor_epoch=None):
        # 1. 暴力拦截：不管前面谁把 device 改成了 cpu，在这里统统强制改回 cuda！
        self.args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 加上这一行，双保险：确保模型先去 GPU！
        model = model.to(self.args.device)
        # 3. 打印确认
        # print(f"============ 暴力修改后，当前使用的设备是: {self.args.device} ============")
        model.eval()
        y_pred, y_true = [], []
        eval_loss, predict_loss, generate_loss = 0.0, 0.0, 0.0
        _aa = int(getattr(self.args, 'save_anchor_analysis', 0)) != 0
        _split = str(getattr(self.args, 'anchor_analysis_split', 'test')).lower()
        run_anchor = _aa and (
            _split in ('all', 'both')
            or (_split == 'test' and mode == 'TEST')
            or (_split == 'valid' and mode == 'VAL')
        )
        return_fusion_aux = bool(int(getattr(self.args, 'return_fusion_aux', 0))) or run_anchor
        split_tag = 'test' if mode == 'TEST' else 'valid' if mode == 'VAL' else str(mode).lower()
        anchor_rows = [] if run_anchor else None
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:

                    text = batch_data['text'].to(self.args.device)
                    text_m = batch_data['text_m'].to(self.args.device)
                    text_missing_mask = batch_data['text_missing_mask'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    audio_m = batch_data['audio_m'].to(self.args.device)
                    audio_mask = batch_data['audio_mask'].to(self.args.device)
                    audio_missing_mask = batch_data['audio_missing_mask'].to(self.args.device)
                    vision = batch_data['vision'].to(self.args.device)
                    vision_m = batch_data['vision_m'].to(self.args.device)
                    vision_mask = batch_data['vision_mask'].to(self.args.device)
                    vision_missing_mask = batch_data['vision_missing_mask'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    text_corrupt_mask = batch_data.get('text_corrupt_mask')
                    if text_corrupt_mask is not None:
                        text_corrupt_mask = text_corrupt_mask.to(self.args.device)

                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)

                    model_out = model(
                        (text, text_m, text_missing_mask),
                        (audio, audio_m, audio_mask, audio_missing_mask),
                        (vision, vision_m, vision_mask, vision_missing_mask),
                        text_corrupt_mask=text_corrupt_mask,
                        return_fusion_aux=return_fusion_aux,
                        return_anchor_analysis=run_anchor,
                    )
                    if len(model_out) == 3:
                        outputs, gen_loss, _extra = model_out
                    else:
                        outputs, gen_loss = model_out
                        _extra = {}

                    pred_loss = self.criterion(outputs, labels)
                    total_loss = pred_loss + gen_loss
                    loss = pred_loss

                    eval_loss += loss.item()
                    predict_loss += pred_loss.item()
                    generate_loss += gen_loss.item()

                    y_pred.append(outputs.cpu())
                    y_true.append(labels.cpu())

                    if run_anchor and _extra:
                        t_aux = _extra.get('teacher_fusion_aux') or {}
                        s_aux = _extra.get('student_fusion_aux') or {}
                        t_pred = _extra.get('teacher_pred')
                        t_eff_center = str(_extra.get('teacher_effective_center', ''))
                        if t_pred is None or not s_aux:
                            continue
                        ta_idx_t = t_aux.get('anchor_idx')
                        sa_idx_t = s_aux.get('anchor_idx')
                        ta_log = t_aux.get('anchor_logits')
                        sa_log = s_aux.get('anchor_logits')
                        bsz = outputs.size(0)
                        raw_ids = batch_data.get('id')
                        if raw_ids is not None and torch.is_tensor(raw_ids):
                            raw_ids = raw_ids.detach().cpu()
                        fc_student = str(getattr(self.args, 'fusion_center_modality', ''))
                        dtc_arg = str(getattr(self.args, 'distill_teacher_center_modality', 'same_as_student'))
                        seed_v = getattr(self.args, 'seed', '')
                        tct = getattr(self.args, 'text_corrupt_train', 0.0)
                        tce = getattr(self.args, 'text_corrupt_eval', 0.0)
                        for j in range(bsz):
                            ti = int(ta_idx_t[j].item()) if ta_idx_t is not None else -1
                            si = int(sa_idx_t[j].item()) if sa_idx_t is not None else -1
                            tn = _ANCHOR_IDX_TO_NAME[ti] if 0 <= ti < len(_ANCHOR_IDX_TO_NAME) else ''
                            sn = _ANCHOR_IDX_TO_NAME[si] if 0 <= si < len(_ANCHOR_IDX_TO_NAME) else ''
                            sp = float(outputs[j].detach().float().cpu().reshape(-1)[0].item())
                            tp = float(t_pred[j].detach().float().cpu().reshape(-1)[0].item())
                            lb = float(labels[j].detach().float().cpu().reshape(-1)[0].item())
                            if raw_ids is None:
                                sid = j
                            elif isinstance(raw_ids, torch.Tensor):
                                sid = raw_ids[j].item()
                            else:
                                sid = raw_ids[j]
                            anchor_rows.append({
                                'sample_id': sid,
                                'split': split_tag,
                                'seed': seed_v,
                                'text_corrupt_train': tct,
                                'text_corrupt_eval': tce,
                                'fusion_center_modality': fc_student,
                                'teacher_center_modality': t_eff_center,
                                'teacher_anchor_idx': ti,
                                'student_anchor_idx': si,
                                'teacher_anchor_name': tn,
                                'student_anchor_name': sn,
                                'anchor_match': int(ti == si) if ti >= 0 and si >= 0 else 0,
                                'teacher_pred': tp,
                                'student_pred': sp,
                                'label': lb,
                                'abs_error_student': abs(sp - lb),
                                'teacher_anchor_logits': _anchor_logits_row_str(ta_log, j) if ta_log is not None else '',
                                'student_anchor_logits': _anchor_logits_row_str(sa_log, j) if sa_log is not None else '',
                            })
        eval_loss = eval_loss / len(dataloader)

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)

        logger.info("%s-(%s) >> %s" % (mode, self.args.modelName, dict_to_str(eval_results)))

        if run_anchor and anchor_rows:
            out_dir = str(getattr(self.args, 'anchor_analysis_dir', 'results/anchor_analysis'))
            os.makedirs(out_dir, exist_ok=True)
            ds = str(getattr(self.args, 'datasetName', 'data'))
            fc = str(getattr(self.args, 'fusion_center_modality', 'text'))
            dtc = str(getattr(self.args, 'distill_teacher_center_modality', 'same_as_student')).replace('/', '_')
            tct = getattr(self.args, 'text_corrupt_train', 0.0)
            tce = getattr(self.args, 'text_corrupt_eval', 0.0)
            seed_v = getattr(self.args, 'seed', '')
            ep_tag = f'_ep{int(anchor_epoch)}' if anchor_epoch is not None else ''
            rs = str(getattr(self.args, 'run_slug', '') or '').strip()
            rs_seg = f'_{rs}' if rs else ''
            base = f'{ds}_fc{fc}_dtc{dtc}_tc{tct}_te{tce}_seed{seed_v}_{split_tag}{ep_tag}{rs_seg}_anchor'
            sample_path = os.path.join(out_dir, f'{base}_samples.csv')
            summary_path = os.path.join(out_dir, f'{base}_summary.csv')
            fieldnames = [
                'sample_id', 'split', 'seed', 'text_corrupt_train', 'text_corrupt_eval',
                'fusion_center_modality', 'teacher_center_modality',
                'teacher_anchor_idx', 'student_anchor_idx', 'teacher_anchor_name', 'student_anchor_name',
                'anchor_match', 'teacher_pred', 'student_pred', 'label', 'abs_error_student',
                'teacher_anchor_logits', 'student_anchor_logits',
            ]
            with open(sample_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(anchor_rows)
            n = len(anchor_rows)
            am = np.array([r['anchor_match'] for r in anchor_rows], dtype=np.float64)
            agree = float(am.mean()) if n else float('nan')
            ta = np.array([r['teacher_anchor_idx'] for r in anchor_rows], dtype=np.int64)
            sa = np.array([r['student_anchor_idx'] for r in anchor_rows], dtype=np.int64)
            ae = np.array([r['abs_error_student'] for r in anchor_rows], dtype=np.float64)
            sp = np.array([r['student_pred'] for r in anchor_rows], dtype=np.float64)
            lb = np.array([r['label'] for r in anchor_rows], dtype=np.float64)

            def _ratio(idx_arr, k):
                return float((idx_arr == k).mean()) if n else float('nan')

            m_ok = am > 0.5
            m_bad = am < 0.5
            mae_m = float(ae[m_ok].mean()) if m_ok.any() else float('nan')
            mae_mm = float(ae[m_bad].mean()) if m_bad.any() else float('nan')
            cr_m = _safe_pearson(sp[m_ok], lb[m_ok]) if m_ok.sum() >= 2 else float('nan')
            cr_mm = _safe_pearson(sp[m_bad], lb[m_bad]) if m_bad.sum() >= 2 else float('nan')
            with open(summary_path, 'w', newline='') as sf:
                sw = csv.writer(sf)
                sw.writerow([
                    'total_samples', 'agreement_rate', 'disagreement_rate',
                    'teacher_text_ratio', 'teacher_audio_ratio', 'teacher_vision_ratio',
                    'student_text_ratio', 'student_audio_ratio', 'student_vision_ratio',
                    'mae_when_match', 'mae_when_mismatch', 'corr_when_match', 'corr_when_mismatch',
                    'dataset', 'fusion_center_modality', 'distill_teacher_center_modality',
                    'text_corrupt_train', 'text_corrupt_eval', 'seed', 'split',
                ])
                sw.writerow([
                    n, agree, 1.0 - agree if n else float('nan'),
                    _ratio(ta, 0), _ratio(ta, 1), _ratio(ta, 2),
                    _ratio(sa, 0), _ratio(sa, 1), _ratio(sa, 2),
                    mae_m, mae_mm, cr_m, cr_mm,
                    ds, fc, dtc, tct, tce, seed_v, split_tag,
                ])
            logger.info('Anchor analysis written to %s and %s', sample_path, summary_path)

        return eval_results
