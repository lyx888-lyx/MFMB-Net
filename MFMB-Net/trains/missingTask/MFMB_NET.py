import os
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


class MFMB_NET():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.datasetName)
        self.mide_epoch_stats = []

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
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        while True:
            epochs += 1
            y_pred, y_true = [], []
            train_loss, predict_loss, generate_loss, mide_loss_total = 0.0, 0.0, 0.0, 0.0
            mide_loss_parts = {}
            epoch_mide_stats = []
            model.train()
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    optimizer.zero_grad()

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
                        if epochs > getattr(self.args, 'mide_warmup_epochs', 1):
                            loss = loss + mide_total
                        for k, v in mide_aux['losses'].items():
                            mide_loss_parts[k] = mide_loss_parts.get(k, 0.0) + float(v.item())
                        mide_loss_total += float(mide_total.item() if torch.is_tensor(mide_total) else mide_total)
                        if mide_aux.get('stats'):
                            epoch_mide_stats.append(mide_aux['stats'])

                    loss.backward()

                    if self.args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_(
                            [param for param in model.parameters() if param.requires_grad],
                            self.args.grad_clip,
                        )

                    optimizer.step()

                    train_loss += loss.item()
                    predict_loss += pred_loss.item()
                    generate_loss += gen_loss.item() if torch.is_tensor(gen_loss) else float(gen_loss)

                    y_pred.append(prediction.detach().cpu())
                    y_true.append(labels.detach().cpu())

            n_batches = len(dataloader['train'])
            train_loss /= n_batches
            predict_loss /= n_batches
            generate_loss /= n_batches
            mide_loss_total /= n_batches

            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)

            mide_log = ''
            if mide_loss_parts:
                mide_log = ' mide: ' + ', '.join(
                    f'{k}={mide_loss_parts[k] / n_batches:.4f}' for k in sorted(mide_loss_parts.keys())
                )
            if epoch_mide_stats:
                avg_stats = {}
                for key in epoch_mide_stats[0].keys():
                    vals = torch.stack([s[key] for s in epoch_mide_stats], dim=0).mean(dim=0)
                    avg_stats[key] = vals.tolist()
                self.mide_epoch_stats.append({'epoch': epochs, 'stats': avg_stats})
                mide_log += f' D_mean={avg_stats.get("D_mean")} C_mean={avg_stats.get("C_mean")}'

            logger.info(
                "TRAIN-(%s) (%d/%d/%d)>> loss: %.4f(pred: %.4f; gen: %.4f;%s) %s"
                % (
                    self.args.modelName,
                    epochs - best_epoch,
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
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])

            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)

            if epochs - best_epoch >= self.args.early_stop:
                return

    def do_test(self, model, dataloader, mode="VAL", epoch=1):
        model.eval()
        y_pred, y_true = [], []
        eval_loss, predict_loss, generate_loss = 0.0, 0.0, 0.0
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    prediction, gen_loss, mide_aux, labels = self._forward_batch(
                        model, batch_data, epoch, training=False
                    )

                    pred_loss = self.criterion(prediction, labels)
                    loss = pred_loss

                    eval_loss += loss.item()
                    predict_loss += pred_loss.item()
                    generate_loss += gen_loss.item() if torch.is_tensor(gen_loss) else float(gen_loss)

                    y_pred.append(prediction.cpu())
                    y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader)

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)

        logger.info("%s-(%s) >> %s" % (mode, self.args.modelName, dict_to_str(eval_results)))
        return eval_results
