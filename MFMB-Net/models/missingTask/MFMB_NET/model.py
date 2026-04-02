import contextlib
import torch
from torch import nn

from models.missingTask.MFMB_NET.alignment_1 import Alignment
from models.missingTask.MFMB_NET.generator import Generator
from models.subNets.BertTextEncoder import BertTextEncoder

from models.missingTask.MFMB_NET.fusion_599 import Fusion
from utils.text_corrupt import maybe_corrupt_text_m_only

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

        eps = self.eps.to(mask.device) if isinstance(self.eps, torch.Tensor) else self.eps
        loss = self.loss(pred*mask, target*mask) / (torch.sum(mask) + eps)

        if self.args.recloss_type == 'combine' and self.args.weight_sim_loss!=0:
            loss += (self.args.weight_sim_loss * self.loss_cmd(pred*mask, target*mask) / (torch.sum(mask) + self.eps))
        return loss


class MFMB_NET(nn.Module):
    """
    数据流（蒸馏相关）：
    - 数据集 clean：text / audio / vision；缺失支路：text_m / audio_m / vision_m。
    - BERT 前仅对 **text_m** 做 `maybe_corrupt_text_m_only`；**text 不扰动**，保证教师与 gen 目标均为真正 clean 文本。
    - align / generator / fusion（学生）：text_m, audio_m, vision_m；生成目标为干净 text/audio/vision。
    - 单教师蒸馏：教师前向在共享模块上 **torch.no_grad + 临时 eval**（text_model / align_subnet / fusion_subnet），
      关闭 dropout、BN 用累计统计量，避免与 train 模式随机性叠加，比仅 no_grad 更稳。
    - fusion_aux['fused_rep']：分类头前 utterance 向量（可含 prompt 残差）。
    """
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
        

    def _distill_teacher_forward(
        self,
        text_enc: torch.Tensor,
        audio: torch.Tensor,
        vision: torch.Tensor,
        text_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        vision_mask: torch.Tensor,
    ):
        """
        共享单教师：no_grad（默认）+ 对 text_model / align / fusion **临时 eval**，再恢复原 training 标志。
        教师本步不再次调用 text_model（text_enc 已为学生支路算好）；仍切换 text_model 模式以符合「教师路径涉及模块」一致性与后续扩展。
        """
        mods = [self.text_model, self.align_subnet, self.fusion_subnet]
        backup = [m.training for m in mods]
        for m in mods:
            m.eval()
        try:
            grad_ctx = (
                torch.no_grad()
                if self.args.get('distill_teacher_detach', True)
                else contextlib.nullcontext()
            )
            with grad_ctx:
                text_ht, audio_ht, vision_ht, _, _, _ = self.align_subnet(text_enc, audio, vision)
                full_t = text_mask.float().clamp(0, 1)
                full_a = audio_mask.float().clamp(0, 1)
                full_v = vision_mask.float().clamp(0, 1)
                fusion_in_t = (
                    (text_ht, text_mask, full_t),
                    (audio_ht, audio_mask, full_a),
                    (vision_ht, vision_mask, full_v),
                )
                pred_t, aux_t = self.fusion_subnet(
                    *fusion_in_t, return_aux=True, distill_full_modality_prompt=True
                )
        finally:
            for m, was in zip(mods, backup):
                m.train(was)
        return pred_t, aux_t

    def forward(self, text, audio, vision, return_fusion_aux=False, return_distill=False):
        """
        默认 (return_fusion_aux=False, return_distill=False)：(prediction, gen_loss)，与旧训练一致。

        return_fusion_aux=True：返回 (prediction, gen_loss, fusion_aux)。

        return_distill=True 且 training 且 args.use_distill：
        返回 (student_prediction, gen_loss, distill_pack)。
        distill_pack：teacher_pred, teacher_fused_rep, student_fused_rep（均为 fused_rep 语义）。
        教师分支与共享 align/fusion/BERT；仅输入为 clean 三模态 + distill_full_modality_prompt。
        默认 distill_teacher_detach=True：教师 **no_grad** 且 **临时 eval** 共享子模块，作稳定监督目标。
        """
        text, text_m, missing_mask_t = text
    
        audio, audio_m, audio_mask, missing_mask_a = audio
       

        vision, vision_m, vision_mask, missing_mask_v = vision
       
        text_mask = text[:,1,:]

        text, text_m = maybe_corrupt_text_m_only(
            text, text_m,
            training=self.training,
            text_corrupt_train=self.args.get('text_corrupt_train', 0.0),
            text_corrupt_eval=self.args.get('text_corrupt_eval', 0.0),
            text_corrupt_mode=self.args.get('text_corrupt_mode', 'mix'),
            text_corrupt_span_frac=self.args.get('text_corrupt_span_frac', 0.4),
        )

        text_m = self.text_model(text_m)
        text = self.text_model(text)
       

        text_h, audio_h, vision_h, text_h_g, audio_h_g, vision_h_g = self.align_subnet(text_m, audio_m, vision_m)

        fusion_in_s = (
            (text_h, text_mask, missing_mask_t),
            (audio_h, audio_mask, missing_mask_a),
            (vision_h, vision_mask, missing_mask_v),
        )

        use_distill = (
            self.args.get('use_distill', False)
            and return_distill
            and self.training
        )

        if not self.args.without_generator:
        
            text_ = self.generator_t(text_h_g)

            audio_ = self.generator_a(audio_h_g)

            vision_ = self.generator_v(vision_h_g)

            text_gen_loss = self.gen_loss(text_, text, text_mask - missing_mask_t)
            audio_gen_loss = self.gen_loss(audio_, audio, audio_mask - missing_mask_a)
            vision_gen_loss = self.gen_loss(vision_, vision, vision_mask - missing_mask_v)

            gen = self.args.weight_gen_loss[0] * text_gen_loss + self.args.weight_gen_loss[1] * audio_gen_loss + self.args.weight_gen_loss[2] * vision_gen_loss
        else:
            gen = torch.zeros((), device=text.device, dtype=text.dtype)

        if use_distill:
            pred_s, aux_s = self.fusion_subnet(*fusion_in_s, return_aux=True)
            pred_t, aux_t = self._distill_teacher_forward(
                text, audio, vision, text_mask, audio_mask, vision_mask
            )
            src = self.args.get('distill_feature_source', 'fused_rep')
            z_s = aux_s[src] if isinstance(src, str) and src in aux_s else aux_s['fused_rep']
            z_t = aux_t[src] if isinstance(src, str) and src in aux_t else aux_t['fused_rep']
            r_tc = self.args.get('text_corrupt_train', 0.0) if self.training else self.args.get('text_corrupt_eval', 0.0)
            mode_tc = str(self.args.get('text_corrupt_mode', 'none')).lower()
            distill_pack = {
                'teacher_pred': pred_t,
                'teacher_fused_rep': z_t,
                'student_fused_rep': z_s,
                'student_fusion_aux': aux_s,
                'teacher_used_eval_mode': True,
                'teacher_prompt_full_modality': bool(aux_t.get('distill_full_modality_prompt', False)),
                'teacher_text_corrupted': False,
                'student_text_m_corrupted': bool(r_tc > 0 and mode_tc != 'none'),
            }
            return pred_s, gen, distill_pack

        if return_fusion_aux:
            prediction, fusion_aux = self.fusion_subnet(*fusion_in_s, return_aux=True)
            return prediction, gen, fusion_aux

        prediction = self.fusion_subnet(*fusion_in_s)
        return prediction, gen
        