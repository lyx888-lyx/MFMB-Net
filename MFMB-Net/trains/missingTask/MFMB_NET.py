import os
import time
import logging
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
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

        epochs, best_epoch = 0, 0
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        while True:  
            epochs += 1
            y_pred, y_true = [], []
            losses = []
            model.train()
            train_loss, predict_loss, generate_loss = 0.0, 0.0, 0.0
            train_kd_logit, train_kd_feat = 0.0, 0.0
            use_distill = self.args.get('use_distill', False)
            w_l = float(self.args.get('distill_logit_weight', 0.0))
            w_f = float(self.args.get('distill_feat_weight', 0.0))
            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    
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

                    if use_distill:
                        prediction, gen_loss, dpack = model(
                            (text, text_m, text_missing_mask),
                            (audio, audio_m, audio_mask, audio_missing_mask),
                            (vision, vision_m, vision_mask, vision_missing_mask),
                            return_distill=True,
                        )
                        kd_fn = F.smooth_l1_loss if str(self.args.get('distill_loss_type', 'mse')).lower() == 'smoothl1' else F.mse_loss
                        kd_logit = kd_fn(prediction, dpack['teacher_pred'], reduction='mean')
                        kd_feat = kd_fn(dpack['student_fused_rep'], dpack['teacher_fused_rep'], reduction='mean')
                        dpack['kd_logit_loss'] = kd_logit.item()
                        dpack['kd_feat_loss'] = kd_feat.item()
                        train_kd_logit += dpack['kd_logit_loss']
                        train_kd_feat += dpack['kd_feat_loss']
                    else:
                        prediction, gen_loss = model(
                            (text, text_m, text_missing_mask),
                            (audio, audio_m, audio_mask, audio_missing_mask),
                            (vision, vision_m, vision_mask, vision_missing_mask),
                        )
                        kd_logit = prediction.new_zeros(())
                        kd_feat = prediction.new_zeros(())

                    pred_loss = self.criterion(prediction, labels)
                    if epochs > 1:
                        loss = pred_loss + gen_loss
                    else:
                        loss = pred_loss
                    if use_distill:
                        loss = loss + w_l * kd_logit + w_f * kd_feat
                    loss.backward()
                    
                    if self.args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], self.args.grad_clip)

                    optimizer.step()
                    train_loss += loss.item()
                    predict_loss += pred_loss.item()
                    generate_loss += gen_loss.item()

                    y_pred.append(prediction.cpu())
                    y_true.append(labels.cpu())
                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    optimizer.step()
            train_loss = train_loss / len(dataloader['train'])
            predict_loss = predict_loss / len(dataloader['train'])
            generate_loss = generate_loss / len(dataloader['train'])
            n_batch = len(dataloader['train'])
            mean_kd_l = train_kd_logit / n_batch if use_distill else 0.0
            mean_kd_f = train_kd_feat / n_batch if use_distill else 0.0
            
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            if use_distill:
                logger.info(
                    "TRAIN-(%s) (%d/%d/%d)>> total: %.4f (pred: %.4f; gen: %.4f; kd_logit: %.4f; kd_feat: %.4f) %s"
                    % (
                        self.args.modelName,
                        epochs - best_epoch,
                        epochs,
                        self.args.cur_time,
                        train_loss,
                        predict_loss,
                        generate_loss,
                        mean_kd_l,
                        mean_kd_f,
                        dict_to_str(train_results),
                    )
                )
            else:
                logger.info("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f(pred: %.4f; gen: %.4f) %s" % (self.args.modelName, \
                            epochs - best_epoch, epochs, self.args.cur_time, train_loss, predict_loss, generate_loss, dict_to_str(train_results)))
            
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])

            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)
                
            if epochs - best_epoch >= self.args.early_stop:
                return

    def do_test(self, model, dataloader, mode="VAL"):
        # 1. 暴力拦截：不管前面谁把 device 改成了 cpu，在这里统统强制改回 cuda！
        self.args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 加上这一行，双保险：确保模型先去 GPU！
        model = model.to(self.args.device)
        # 3. 打印确认
        # print(f"============ 暴力修改后，当前使用的设备是: {self.args.device} ============")
        model.eval()
        y_pred, y_true = [], []
        eval_loss, predict_loss, generate_loss = 0.0, 0.0, 0.0
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

                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)

                    outputs, gen_loss = model((text, text_m, text_missing_mask), (audio, audio_m, audio_mask, audio_missing_mask), (vision, vision_m, vision_mask, vision_missing_mask))

                    pred_loss = self.criterion(outputs, labels)
                    total_loss = pred_loss + gen_loss
                    loss = pred_loss

                    eval_loss += loss.item()
                    predict_loss += pred_loss.item()
                    generate_loss += gen_loss.item()

                    y_pred.append(outputs.cpu())
                    y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader)

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)

        logger.info("%s-(%s) >> %s" % (mode, self.args.modelName, dict_to_str(eval_results)))
        return eval_results
