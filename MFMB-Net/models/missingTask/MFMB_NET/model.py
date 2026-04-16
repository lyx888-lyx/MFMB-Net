import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from models.missingTask.MFMB_NET.alignment_1 import Alignment
from models.missingTask.MFMB_NET.generator import Generator
from models.subNets.BertTextEncoder import BertTextEncoder

from models.missingTask.MFMB_NET.fusion_599 import Fusion
from utils.text_corrupt import maybe_corrupt_text_pair, maybe_corrupt_text_m_only

# CMD Loss
class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments=3):
        x1 = x1.view(-1, x1.shape[-1])
        x2 = x2.view(-1, x2.shape[-1])
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        b = torch.max(x2, dim=0)[0]
        a = torch.min(x2, dim=0)[0]
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = (summed+1e-12)**(0.5)
        return sqrt

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)
class RECLoss(nn.Module):
    def __init__(self, args):
        super(RECLoss, self).__init__()

        self.eps = torch.FloatTensor([1e-4]).to(args.device)
        self.args = args

        if args.recloss_type == 'SmoothL1Loss':
            self.loss = nn.SmoothL1Loss(reduction='sum')
        elif args.recloss_type == 'MSELoss':
            self.loss = nn.MSELoss(reduction='sum')
        elif args.recloss_type == 'cmd':
            self.loss = CMD()
        elif args.recloss_type == 'combine':
            self.loss = nn.SmoothL1Loss(reduction='sum')
            self.loss_cmd = CMD()

    def forward(self, pred, target, mask):
        """
            pred, target -> batch, seq_len, d
            mask -> batch, seq_len
        """
        mask = mask.unsqueeze(-1).expand(pred.shape[0], pred.shape[1], pred.shape[2])

        # 在第 70 行前面加一句，强制让 self.eps 去找 mask 所在的设备
        eps = self.eps.to(mask.device) if isinstance(self.eps, torch.Tensor) else self.eps
        loss = self.loss(pred*mask, target*mask) / (torch.sum(mask) + eps)

        if self.args.recloss_type == 'combine' and self.args.weight_sim_loss!=0:
            loss += (self.args.weight_sim_loss * self.loss_cmd(pred*mask, target*mask) / (torch.sum(mask) + self.eps))
        return loss


def _distill_sample_weights(
    missing_mask_t: torch.Tensor,
    text_mask: torch.Tensor,
    missing_mask_a: torch.Tensor,
    audio_mask: torch.Tensor,
    missing_mask_v: torch.Tensor,
    vision_mask: torch.Tensor,
    text_corrupt_mask: torch.Tensor,
    distill_clean_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """样本级蒸馏权重；困难样本（任一模态有缺失或文本被污染）权重为 1，否则为 distill_clean_weight。"""
    t_miss = ((text_mask > 0) & (missing_mask_t < 0.5)).any(dim=1)
    a_miss = ((audio_mask > 0) & (missing_mask_a < 0.5)).any(dim=1)
    v_miss = ((vision_mask > 0) & (missing_mask_v < 0.5)).any(dim=1)
    corrupt = (text_corrupt_mask.sum(dim=1) > 0)
    hard = t_miss | a_miss | v_miss | corrupt
    B = missing_mask_t.size(0)
    device, dtype = missing_mask_t.device, missing_mask_t.dtype
    w = torch.where(
        hard,
        torch.ones(B, device=device, dtype=dtype),
        torch.full((B,), float(distill_clean_weight), device=device, dtype=dtype),
    )
    return w, hard


class MFMB_NET(nn.Module):
    def __init__(self, args):
        super(MFMB_NET, self).__init__()
        self.args = args
        
        self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_bert_finetune)

        self.align_subnet = Alignment(args)

        if not args.without_generator:

            self.generator_t = Generator(args, modality='text')
            self.generator_a = Generator(args, modality='audio')
            self.generator_v = Generator(args, modality='vision')

            self.gen_loss = RECLoss(args)
   
        args.fusion_t_in = args.fusion_a_in = args.fusion_v_in = args.dst_feature_dim_nheads[0] * 3

        self.fusion_subnet = Fusion(args)

    def _teacher_forward_center_modality(self) -> Optional[str]:
        """蒸馏教师分支 hub；None 表示不覆盖（与 student 的 fusion_center_modality 一致）。"""
        dtc = getattr(self.args, 'distill_teacher_center_modality', 'same_as_student')
        if dtc == 'same_as_student':
            return None
        hub = str(dtc)
        fusion = self.fusion_subnet.Model
        if hub == 'dynamic' and getattr(fusion, 'anchor_scorer', None) is None:
            return None
        return hub

    def _effective_teacher_center_name(self) -> str:
        """用于日志 / 导出：教师实际使用的 hub 名称。"""
        ov = self._teacher_forward_center_modality()
        if ov is None:
            return str(getattr(self.fusion_subnet.Model, 'fusion_center_modality', getattr(self.args, 'fusion_center_modality', 'text')))
        return ov

    def _distill_teacher_forward(
        self,
        text_enc: torch.Tensor,
        audio: torch.Tensor,
        vision: torch.Tensor,
        text_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        vision_mask: torch.Tensor,
        center_modality: Optional[str] = None,
    ):
        """
        Clean-teacher：完整干净三模态，复用 text_model（已编码的 text_enc）/ align / fusion。
        distill_teacher_detach=True 时用 no_grad；False 时允许梯度穿过教师路径（共享权重）。
        center_modality 非 None 时临时覆盖 fusion hub（仅本前向），不影响 student。
        """
        args = self.args
        _dt = getattr(args, 'distill_teacher_detach', True)
        detach_teacher = _dt if isinstance(_dt, bool) else (int(_dt) != 0)
        train_was = self.training
        self.eval()

        missing_t = torch.ones_like(text_mask)
        missing_a = torch.ones_like(audio_mask)
        missing_v = torch.ones_like(vision_mask)

        fusion = self.fusion_subnet.Model
        prev_hub = getattr(fusion, 'fusion_center_modality', None)
        if center_modality is not None:
            fusion.fusion_center_modality = center_modality

        def _body():
            th, ah, vh, _, _, _ = self.align_subnet(text_enc, audio, vision)
            return self.fusion_subnet(
                (th, text_mask, missing_t),
                (ah, audio_mask, missing_a),
                (vh, vision_mask, missing_v),
                return_aux=True,
            )

        try:
            if detach_teacher:
                with torch.no_grad():
                    logits, aux = _body()
            else:
                logits, aux = _body()
            return logits, aux
        finally:
            if center_modality is not None and prev_hub is not None:
                fusion.fusion_center_modality = prev_hub
            if train_was:
                self.train()

    def _compute_distill_losses(
        self,
        student_pred: torch.Tensor,
        teacher_pred: torch.Tensor,
        student_aux: Dict,
        teacher_aux: Dict,
        w: torch.Tensor,
        hard: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """logit / fused_rep / batch-relation 三类蒸馏（标量）。"""
        args = self.args
        device = student_pred.device
        dtype = student_pred.dtype
        eps = 1e-8
        w_sum = w.sum().clamp(min=eps)

        # 1) logit：回归用 MSE；分类可扩展为 temperature-KL
        if getattr(args, 'train_mode', 'regression') == 'regression':
            s = student_pred.view(w.size(0), -1).squeeze(-1)
            t = teacher_pred.view(w.size(0), -1).squeeze(-1).detach()
            per = (s - t) ** 2
            kd_logit = (w * per).sum() / w_sum
        else:
            T = float(getattr(args, 'distill_temperature', 2.0))
            T = max(T, 1e-6)
            s_log = F.log_softmax(student_pred / T, dim=-1)
            t_prob = F.softmax(teacher_pred.detach() / T, dim=-1)
            per = (F.kl_div(s_log, t_prob, reduction='none').sum(dim=-1)) * (T * T)
            kd_logit = (w * per).sum() / w_sum

        # 2) 融合表示：SmoothL1，按维均值后样本加权
        z_s = student_aux['fused_rep']
        z_t = teacher_aux['fused_rep'].detach()
        per_feat = F.smooth_l1_loss(z_s, z_t, reduction='none').mean(dim=-1)
        kd_feat = (w * per_feat).sum() / w_sum

        # 3) batch 关系矩阵：仅当 batch 内存在困难样本时启用
        if hard.any():
            zs = F.normalize(student_aux['fused_rep'], dim=-1, eps=1e-8)
            zt = F.normalize(teacher_aux['fused_rep'].detach(), dim=-1, eps=1e-8)
            rs = zs @ zs.t()
            rt = zt @ zt.t()
            kd_rel = F.mse_loss(rs, rt)
        else:
            kd_rel = torch.zeros((), device=device, dtype=dtype)

        return kd_logit, kd_feat, kd_rel

    def forward(self, text, audio, vision, text_corrupt_mask=None, return_fusion_aux=False, return_anchor_analysis=False):
        text, text_m, missing_mask_t = text
    
        audio, audio_m, audio_mask, missing_mask_a = audio
       

        vision, vision_m, vision_mask, missing_mask_v = vision
       
        text_mask = text[:,1,:]

        _ud = getattr(self.args, 'use_distill', 0)
        use_distill = _ud if isinstance(_ud, bool) else (int(_ud) != 0)
        online_tc = int(getattr(self.args, 'online_text_corrupt', 0)) != 0
        uk_logit = int(getattr(self.args, 'use_kd_logit', 0)) != 0
        uk_feat = int(getattr(self.args, 'use_kd_feat', 0)) != 0
        uk_rel = int(getattr(self.args, 'use_kd_rel', 0)) != 0
        ll_a = float(getattr(self.args, 'distill_lambda_logit', 0.0))
        lf_a = float(getattr(self.args, 'distill_lambda_feat', 0.0))
        lr_a = float(getattr(self.args, 'distill_lambda_rel', 0.0))
        # 仅当某项 kd 既打开开关又有正 λ 时才跑 teacher（λ=0 的消融不浪费前向）
        any_kd_enabled = use_distill and self.training and (
            (uk_logit and ll_a > 0) or (uk_feat and lf_a > 0) or (uk_rel and lr_a > 0)
        )

        runtime_corrupt = None
        # use_distill=0：与旧版一致，pair 同时扰动 text/text_m（可与 dataloader 侧再叠加，保持兼容）
        if not use_distill:
            text, text_m = maybe_corrupt_text_pair(
                text, text_m,
                training=self.training,
                text_corrupt_train=getattr(self.args, 'text_corrupt_train', 0.0),
                text_corrupt_eval=getattr(self.args, 'text_corrupt_eval', 0.0),
                text_corrupt_mode=getattr(self.args, 'text_corrupt_mode', 'mix'),
                text_corrupt_span_frac=getattr(self.args, 'text_corrupt_span_frac', 0.4),
            )
        elif self.training and online_tc:
            # 与 dataloader 二选一：dataset 在 online_tc=1 时已跳过 corruption，此处只做一次在线扰动
            text, text_m, runtime_corrupt = maybe_corrupt_text_m_only(
                text, text_m,
                training=self.training,
                text_corrupt_train=getattr(self.args, 'text_corrupt_train', 0.0),
                text_corrupt_eval=getattr(self.args, 'text_corrupt_eval', 0.0),
                text_corrupt_mode=getattr(self.args, 'text_corrupt_mode', 'mix'),
                text_corrupt_span_frac=getattr(self.args, 'text_corrupt_span_frac', 0.4),
                return_corrupt_mask=True,
            )
        # else: use_distill=1 且 online_tc=0 — text/text_m 沿用 batch（teacher 用干净 text，student 用 dataloader 已扰动的 text_m）

        # 合并 dataloader 侧 text_corrupt_mask 与在线扰动掩码（仅 online_tc=1 时有 runtime）
        if text_corrupt_mask is not None:
            tcm = text_corrupt_mask.to(text.device, dtype=text_mask.dtype)
            if runtime_corrupt is not None:
                tcm = (tcm + runtime_corrupt.to(text.device)).clamp(max=1.0)
        elif runtime_corrupt is not None:
            tcm = runtime_corrupt.to(text.device)
        else:
            tcm = torch.zeros(text.size(0), text.size(2), device=text.device, dtype=text_mask.dtype)

        text_m = self.text_model(text_m)
        text = self.text_model(text)
       

        text_h, audio_h, vision_h, text_h_g, audio_h_g, vision_h_g = self.align_subnet(text_m, audio_m, vision_m)
        #[batch_size, seq_len, d]

        need_anchor_eval = bool(return_anchor_analysis and not self.training)
        need_aux = any_kd_enabled or return_fusion_aux or need_anchor_eval

        extra_out = {}

        if not self.args.without_generator:
        
            text_ = self.generator_t(text_h_g)

            audio_ = self.generator_a(audio_h_g)

            vision_ = self.generator_v(vision_h_g)

            text_gen_loss = self.gen_loss(text_, text, text_mask - missing_mask_t)
            audio_gen_loss = self.gen_loss(audio_, audio, audio_mask - missing_mask_a)
            vision_gen_loss = self.gen_loss(vision_, vision, vision_mask - missing_mask_v)

            if need_aux:
                prediction, student_aux = self.fusion_subnet(
                    (text_h, text_mask, missing_mask_t),
                    (audio_h, audio_mask, missing_mask_a),
                    (vision_h, vision_mask, missing_mask_v),
                    return_aux=True,
                )
            else:
                prediction = self.fusion_subnet(
                    (text_h, text_mask, missing_mask_t),
                    (audio_h, audio_mask, missing_mask_a),
                    (vision_h, vision_mask, missing_mask_v),
                )
                student_aux = None
                
            gen_loss = self.args.weight_gen_loss[0] * text_gen_loss + self.args.weight_gen_loss[1] * audio_gen_loss + self.args.weight_gen_loss[2] * vision_gen_loss

            teacher_aux = None
            if any_kd_enabled:
                teacher_pred, teacher_aux = self._distill_teacher_forward(
                    text, audio, vision, text_mask, audio_mask, vision_mask,
                    center_modality=self._teacher_forward_center_modality(),
                )
                cw = float(getattr(self.args, 'distill_clean_weight', 0.0))
                w, hard = _distill_sample_weights(
                    missing_mask_t, text_mask,
                    missing_mask_a, audio_mask,
                    missing_mask_v, vision_mask,
                    tcm, cw,
                )
                kd_logit, kd_feat, kd_rel = self._compute_distill_losses(
                    prediction, teacher_pred, student_aux, teacher_aux, w, hard,
                )
                extra_out['kd_logit'] = kd_logit
                extra_out['kd_feat'] = kd_feat
                extra_out['kd_rel'] = kd_rel
            if return_fusion_aux and student_aux is not None:
                extra_out['student_fusion_aux'] = student_aux
            if any_kd_enabled and return_fusion_aux and teacher_aux is not None:
                extra_out['teacher_fusion_aux'] = teacher_aux

            if need_anchor_eval and student_aux is not None:
                t_hub = self._teacher_forward_center_modality()
                t_pred, t_aux = self._distill_teacher_forward(
                    text, audio, vision, text_mask, audio_mask, vision_mask,
                    center_modality=t_hub,
                )
                extra_out['teacher_pred'] = t_pred
                extra_out['teacher_fusion_aux'] = t_aux
                extra_out['student_fusion_aux'] = student_aux
                extra_out['teacher_effective_center'] = self._effective_teacher_center_name()

            if extra_out:
                return prediction, gen_loss, extra_out
            return prediction, gen_loss
            
        else:
            if need_aux:
                prediction, student_aux = self.fusion_subnet(
                    (text_h, text_mask, missing_mask_t),
                    (audio_h, audio_mask, missing_mask_a),
                    (vision_h, vision_mask, missing_mask_v),
                    return_aux=True,
                )
            else:
                prediction = self.fusion_subnet(
                    (text_h, text_mask, missing_mask_t),
                    (audio_h, audio_mask, missing_mask_a),
                    (vision_h, vision_mask, missing_mask_v),
                )
                student_aux = None

            gen_loss = torch.Tensor([0]).to(self.args.device)

            teacher_aux = None
            if any_kd_enabled:
                teacher_pred, teacher_aux = self._distill_teacher_forward(
                    text, audio, vision, text_mask, audio_mask, vision_mask,
                    center_modality=self._teacher_forward_center_modality(),
                )
                cw = float(getattr(self.args, 'distill_clean_weight', 0.0))
                w, hard = _distill_sample_weights(
                    missing_mask_t, text_mask,
                    missing_mask_a, audio_mask,
                    missing_mask_v, vision_mask,
                    tcm, cw,
                )
                kd_logit, kd_feat, kd_rel = self._compute_distill_losses(
                    prediction, teacher_pred, student_aux, teacher_aux, w, hard,
                )
                extra_out['kd_logit'] = kd_logit
                extra_out['kd_feat'] = kd_feat
                extra_out['kd_rel'] = kd_rel
            if return_fusion_aux and student_aux is not None:
                extra_out['student_fusion_aux'] = student_aux
            if any_kd_enabled and return_fusion_aux and teacher_aux is not None:
                extra_out['teacher_fusion_aux'] = teacher_aux

            if need_anchor_eval and student_aux is not None:
                t_hub = self._teacher_forward_center_modality()
                t_pred, t_aux = self._distill_teacher_forward(
                    text, audio, vision, text_mask, audio_mask, vision_mask,
                    center_modality=t_hub,
                )
                extra_out['teacher_pred'] = t_pred
                extra_out['teacher_fusion_aux'] = t_aux
                extra_out['student_fusion_aux'] = student_aux
                extra_out['teacher_effective_center'] = self._effective_teacher_center_name()

            if extra_out:
                return prediction, gen_loss, extra_out
            return prediction, gen_loss
        
