import os
import json
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


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj


class MFMB_NET():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.datasetName)
        self.epoch_stats_log = []
        self.use_amp = getattr(args, 'use_amp', False) and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _forward_batch(self, model, batch_data, epoch, training):
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

        if self.args.train_mode == 'classification':
            labels = labels.view(-1).long()
        else:
            labels = labels.view(-1, 1)

        out = model(
            (text, text_m, text_missing_mask),
            (audio, audio_m, audio_mask, audio_missing_mask),
            (vision, vision_m, vision_mask, vision_missing_mask),
            labels=labels,
            epoch=epoch,
            training=training,
        )
        if len(out) == 3:
            prediction, gen_loss, mide_aux = out
        else:
            prediction, gen_loss = out
            mide_aux = None
        return prediction, gen_loss, mide_aux, labels

    def _early_stop_active(self, epoch):
        if getattr(self.args, 'mide_enable', False):
            return epoch > getattr(self.args, 'mide_full_start_epoch', 4)
        return epoch > 1

    def _save_epoch_stats(self, epoch_record):
        self.epoch_stats_log.append(epoch_record)
        exp_tag = getattr(self.args, 'exp_tag', '') or (
            'mide' if getattr(self.args, 'mide_enable', False) else 'baseline'
        )
        missing = getattr(self.args, 'missing_rate', (0.0,))[0]
        seed = getattr(self.args, 'seed', 0)
        out_dir = os.path.join('results', 'epoch_stats')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{exp_tag}_m{missing:.1f}_seed{seed}.json')
        with open(out_path, 'w') as f:
            json.dump(_json_safe(self.epoch_stats_log), f, indent=2)

    def do_train(self, model, dataloader):
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

        epochs, best_epoch = 0, 0
        best_valid_loss = 1e8
        best_valid_corr = -1e8
        best_valid_has0 = -1e8
        best_epoch_by_loss = 0
        best_epoch_by_corr = 0
        best_epoch_by_has0 = 0
        epochs_since_best = 0
        saved_any = False

        while True:
            epochs += 1
            y_pred, y_true = [], []
            train_loss, predict_loss, generate_loss, mide_loss_total = 0.0, 0.0, 0.0, 0.0
            mide_loss_parts = {}
            epoch_mide_stats = []
            model.train()
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    optimizer.zero_grad(set_to_none=True)

                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        prediction, gen_loss, mide_aux, labels = self._forward_batch(
                            model, batch_data, epochs, training=True
                        )
                        pred_loss = self.criterion(prediction, labels)

                        if epochs > 1:
                            loss = pred_loss + gen_loss
                        else:
                            loss = pred_loss

                        if mide_aux is not None and mide_aux.get('losses') is not None:
                            mide_total = mide_aux['losses'].get('total_mide', 0.0)
                            if getattr(self.args, 'mide_enable', False):
                                loss = loss + mide_total
                            for k, v in mide_aux['losses'].items():
                                mide_loss_parts[k] = mide_loss_parts.get(k, 0.0) + float(v.item())
                            mide_loss_total += float(mide_total.item() if torch.is_tensor(mide_total) else mide_total)
                            if mide_aux.get('stats'):
                                epoch_mide_stats.append(mide_aux['stats'])

                    if not torch.isfinite(loss):
                        logger.warning('Non-finite loss at epoch %d; skipping batch', epochs)
                        continue

                    self.scaler.scale(loss).backward()

                    if self.args.grad_clip != -1.0:
                        self.scaler.unscale_(optimizer)
                        nn.utils.clip_grad_value_(
                            [param for param in model.parameters() if param.requires_grad],
                            self.args.grad_clip,
                        )

                    self.scaler.step(optimizer)
                    self.scaler.update()

                    train_loss += loss.item()
                    predict_loss += pred_loss.item()
                    generate_loss += gen_loss.item() if torch.is_tensor(gen_loss) else float(gen_loss)

                    y_pred.append(prediction.detach().float().cpu())
                    y_true.append(labels.detach().float().cpu())

            n_batches = max(len(dataloader['train']), 1)
            train_loss /= n_batches
            predict_loss /= n_batches
            generate_loss /= n_batches
            mide_loss_total /= n_batches

            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)

            avg_stats = {}
            if epoch_mide_stats:
                for key in epoch_mide_stats[0].keys():
                    if key == 'phase':
                        avg_stats[key] = epoch_mide_stats[-1][key]
                        continue
                    vals = torch.stack([s[key] for s in epoch_mide_stats], dim=0).mean(dim=0)
                    avg_stats[key] = vals.tolist() if torch.is_tensor(vals) else vals

            mide_log = ''
            if mide_loss_parts:
                mide_log = ' mide: ' + ', '.join(
                    f'{k}={mide_loss_parts[k] / n_batches:.4f}' for k in sorted(mide_loss_parts.keys())
                )
            if avg_stats:
                mide_log += f' D_mean={avg_stats.get("D_mean")} C_mean={avg_stats.get("C_mean")}'

            logger.info(
                "TRAIN-(%s) (%d/%d/%d)>> loss: %.4f(pred: %.4f; gen: %.4f;%s) %s"
                % (
                    self.args.modelName,
                    epochs_since_best,
                    epochs,
                    self.args.cur_time,
                    train_loss,
                    predict_loss,
                    generate_loss,
                    f' mide_total={mide_loss_total:.4f}' if getattr(self.args, 'mide_enable', False) else '',
                    dict_to_str(train_results),
                )
            )
            if mide_log:
                logger.info("TRAIN-MIDE >>%s" % mide_log)

            val_results = self.do_test(model, dataloader['valid'], mode="VAL", epoch=epochs)
            cur_loss = val_results['Loss']
            cur_corr = val_results.get('Corr', 0.0)
            cur_has0 = val_results.get('Has0_acc_2', 0.0)
            scheduler.step(cur_loss)

            improved_loss = cur_loss <= (best_valid_loss - 1e-6)
            tie_loss = abs(cur_loss - best_valid_loss) <= 1e-6
            tie_better_corr = tie_loss and cur_corr > best_valid_corr

            if improved_loss or tie_better_corr:
                best_valid_loss = cur_loss
                best_epoch_by_loss = epochs
                best_epoch = epochs
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)
                saved_any = True
                epochs_since_best = 0
                logger.info(
                    'Saved best_by_loss checkpoint epoch=%d Loss=%.4f Corr=%.4f',
                    epochs, cur_loss, cur_corr,
                )
            else:
                if self._early_stop_active(epochs):
                    epochs_since_best += 1

            if cur_corr > best_valid_corr + 1e-6:
                best_valid_corr = cur_corr
                best_epoch_by_corr = epochs
            if cur_has0 > best_valid_has0 + 1e-6:
                best_valid_has0 = cur_has0
                best_epoch_by_has0 = epochs

            epoch_record = {
                'epoch': epochs,
                'train_loss': round(train_loss, 6),
                'pred_loss': round(predict_loss, 6),
                'gen_loss': round(generate_loss, 6),
                'mide_total': round(mide_loss_total, 6),
                'valid': val_results,
                'best_by_loss': {'epoch': best_epoch_by_loss, 'Loss': best_valid_loss},
                'best_by_corr': {'epoch': best_epoch_by_corr, 'Corr': best_valid_corr},
                'best_by_has0': {'epoch': best_epoch_by_has0, 'Has0_acc_2': best_valid_has0},
            }
            for k, v in mide_loss_parts.items():
                epoch_record[f'mide_{k}'] = round(v / n_batches, 6)
            for k, v in avg_stats.items():
                epoch_record[k] = v
            self._save_epoch_stats(epoch_record)

            if self._early_stop_active(epochs) and epochs_since_best >= self.args.early_stop:
                logger.info(
                    'Early stop at epoch %d | best_by_loss ep=%d Loss=%.4f | best_by_corr ep=%d Corr=%.4f | best_by_has0 ep=%d Has0=%.4f',
                    epochs, best_epoch_by_loss, best_valid_loss,
                    best_epoch_by_corr, best_valid_corr,
                    best_epoch_by_has0, best_valid_has0,
                )
                if not saved_any:
                    torch.save(model.cpu().state_dict(), self.args.model_save_path)
                    model.to(self.args.device)
                return

    def do_test(self, model, dataloader, mode="VAL", epoch=1):
        model.eval()
        y_pred, y_true = [], []
        eval_loss, predict_loss, generate_loss = 0.0, 0.0, 0.0
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        prediction, gen_loss, mide_aux, labels = self._forward_batch(
                            model, batch_data, epoch, training=False
                        )
                        pred_loss = self.criterion(prediction, labels)
                        loss = pred_loss

                    eval_loss += loss.item()
                    predict_loss += pred_loss.item()
                    generate_loss += gen_loss.item() if torch.is_tensor(gen_loss) else float(gen_loss)

                    y_pred.append(prediction.float().cpu())
                    y_true.append(labels.float().cpu())
        n_batches = max(len(dataloader), 1)
        eval_loss = eval_loss / n_batches

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)

        logger.info("%s-(%s) >> %s" % (mode, self.args.modelName, dict_to_str(eval_results)))
        return eval_results
