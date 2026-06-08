import torch
import torch.nn as nn
import torch.nn.functional as F


class LightScalarHead(nn.Module):
    """Linear -> GELU -> Dropout -> Linear -> scalar output."""

    def __init__(self, in_dim, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


def compute_availability(
    text_missing_mask,
    audio_missing_mask,
    vision_missing_mask,
    text_mask=None,
    audio_mask=None,
    vision_mask=None,
    eps=1e-6,
):
    """
    A_m from observed masks. missing_mask fields are observed masks in this codebase.
    Returns A_t, A_a, A_v each [batch, 1].
    """
    if text_mask is None:
        valid_t = text_missing_mask.sum(dim=1, keepdim=True).clamp(min=eps)
        obs_t = text_missing_mask.sum(dim=1, keepdim=True)
    else:
        valid_t = text_mask.sum(dim=1, keepdim=True).clamp(min=eps)
        obs_t = (text_missing_mask * text_mask).sum(dim=1, keepdim=True)

    if audio_mask is not None:
        valid_a = audio_mask.sum(dim=1, keepdim=True).clamp(min=eps)
        obs_a = (audio_missing_mask * audio_mask).sum(dim=1, keepdim=True)
    else:
        valid_a = audio_missing_mask.sum(dim=1, keepdim=True).clamp(min=eps)
        obs_a = audio_missing_mask.sum(dim=1, keepdim=True)

    if vision_mask is not None:
        valid_v = vision_mask.sum(dim=1, keepdim=True).clamp(min=eps)
        obs_v = (vision_missing_mask * vision_mask).sum(dim=1, keepdim=True)
    else:
        valid_v = vision_missing_mask.sum(dim=1, keepdim=True).clamp(min=eps)
        obs_v = vision_missing_mask.sum(dim=1, keepdim=True)

    a_t = obs_t / valid_t
    a_a = obs_a / valid_a
    a_v = obs_v / valid_v
    return a_t, a_a, a_v


class MIDEController(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        t_dim = args.fusion_t_hid
        a_dim = args.fusion_a_hid
        v_dim = args.fusion_v_hid
        hidden = getattr(args, 'mide_head_hidden', 32)
        drop = getattr(args, 'mide_head_dropout', 0.1)

        self.uni_head_t = LightScalarHead(t_dim, hidden, drop)
        self.uni_head_a = LightScalarHead(a_dim, hidden, drop)
        self.uni_head_v = LightScalarHead(v_dim, hidden, drop)

        self.util_head_t = LightScalarHead(t_dim, hidden, drop)
        self.util_head_a = LightScalarHead(a_dim, hidden, drop)
        self.util_head_v = LightScalarHead(v_dim, hidden, drop)

        self.poll_head_t = LightScalarHead(t_dim, hidden, drop)
        self.poll_head_a = LightScalarHead(a_dim, hidden, drop)
        self.poll_head_v = LightScalarHead(v_dim, hidden, drop)
        # Neutral init: U~0.5, P~0.1 -> stable initial density ~0.45*A
        for head in (self.util_head_t, self.util_head_a, self.util_head_v):
            nn.init.zeros_(head.net[-1].weight)
            nn.init.constant_(head.net[-1].bias, 1.5)
        for head in (self.poll_head_t, self.poll_head_a, self.poll_head_v):
            nn.init.zeros_(head.net[-1].weight)
            nn.init.constant_(head.net[-1].bias, -2.2)

    def forward_heads(self, text_rep, audio_rep, vision_rep):
        y_t = self.uni_head_t(text_rep)
        y_a = self.uni_head_a(audio_rep)
        y_v = self.uni_head_v(vision_rep)

        u_logits = torch.cat([
            self.util_head_t(text_rep),
            self.util_head_a(audio_rep),
            self.util_head_v(vision_rep),
        ], dim=1)

        p_logits = torch.cat([
            self.poll_head_t(text_rep),
            self.poll_head_a(audio_rep),
            self.poll_head_v(vision_rep),
        ], dim=1)

        return y_t, y_a, y_v, u_logits, p_logits

    def compute_density(self, a_vec, u_logits, p_logits):
        """a_vec, u_logits, p_logits: [batch, 3] -> D_task [batch, 3]."""
        u_score = torch.sigmoid(u_logits)
        p_score = torch.sigmoid(p_logits)
        d_task = a_vec * u_score * (1.0 - p_score)
        return d_task.clamp(min=1e-3, max=1.0)

    def compute_losses(
        self,
        y_t,
        y_a,
        y_v,
        u_logits,
        p_logits,
        a_vec,
        d_task,
        labels,
        y_all,
        y_without_t,
        y_without_a,
        y_without_v,
        epoch,
        enable_mide_losses=True,
    ):
        args = self.args
        labels = labels.view_as(y_all)
        device = labels.device
        zero = torch.tensor(0.0, device=device)

        l_uni = (
            F.smooth_l1_loss(y_t, labels)
            + F.smooth_l1_loss(y_a, labels)
            + F.smooth_l1_loss(y_v, labels)
        )

        losses = {
            'uni_pred': l_uni,
            'utility_ce': zero,
            'utility_rank': zero,
            'noinfo': zero,
            'pollution': zero,
            'sparse_keep': zero,
        }
        stats = {}

        warmup = getattr(args, 'mide_warmup_epochs', 1)
        if not enable_mide_losses or epoch <= warmup:
            losses['total_mide'] = getattr(args, 'mide_lambda_uni', 0.2) * l_uni if epoch > warmup else zero
            return losses, stats

        e_t = (y_t - labels).abs().detach()
        e_a = (y_a - labels).abs().detach()
        e_v = (y_v - labels).abs().detach()
        tau = getattr(args, 'mide_tau', 0.7)
        neg_e = torch.cat([-e_t, -e_a, -e_v], dim=1) / tau
        u_target = F.softmax(neg_e, dim=1)
        u_log_prob = F.log_softmax(u_logits, dim=1)
        l_uce = F.kl_div(u_log_prob, u_target, reduction='batchmean')

        err_all = (y_all - labels).abs().detach()
        err_wo_t = (y_without_t - labels).abs().detach()
        err_wo_a = (y_without_a - labels).abs().detach()
        err_wo_v = (y_without_v - labels).abs().detach()
        c_t = err_wo_t - err_all
        c_a = err_wo_a - err_all
        c_v = err_wo_v - err_all
        c_vec = torch.cat([c_t, c_a, c_v], dim=1)

        margin = getattr(args, 'mide_margin', 0.05)
        contrib_eps = getattr(args, 'mide_contrib_eps', 0.01)
        l_rank = zero
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                mask = (c_vec[:, i] - c_vec[:, j]) > contrib_eps
                if mask.any():
                    diff = u_logits[mask, i] - u_logits[mask, j]
                    l_rank = l_rank + F.relu(margin - diff).mean()

        avail_high = getattr(args, 'mide_avail_high', 0.8)
        contrib_small = getattr(args, 'mide_contrib_small', 0.01)
        l_noinfo = zero
        for m in range(3):
            noinfo_mask = (a_vec[:, m:m + 1] > avail_high) & (c_vec[:, m:m + 1].abs() < contrib_small)
            if noinfo_mask.any():
                l_noinfo = l_noinfo + (noinfo_mask.float() * d_task[:, m:m + 1]).mean()

        l_poll = zero
        for m in range(3):
            l_poll = l_poll + (F.relu(-c_vec[:, m:m + 1]) * d_task[:, m:m + 1]).mean()

        avail_low = getattr(args, 'mide_avail_low', 0.35)
        contrib_pos = getattr(args, 'mide_contrib_pos', 0.02)
        keep_target = getattr(args, 'mide_keep_density_target', 0.45)
        l_sparse = zero
        for m in range(3):
            sparse_mask = (a_vec[:, m:m + 1] < avail_low) & (c_vec[:, m:m + 1] > contrib_pos)
            if sparse_mask.any():
                l_sparse = l_sparse + (
                    sparse_mask.float() * F.relu(keep_target - d_task[:, m:m + 1])
                ).mean()

        losses.update({
            'utility_ce': l_uce,
            'utility_rank': l_rank,
            'noinfo': l_noinfo,
            'pollution': l_poll,
            'sparse_keep': l_sparse,
        })

        lam_uni = getattr(args, 'mide_lambda_uni', 0.2)
        lam_uce = getattr(args, 'mide_lambda_uce', 0.5)
        lam_rank = getattr(args, 'mide_lambda_rank', 0.2)
        lam_noinfo = getattr(args, 'mide_lambda_noinfo', 0.1)
        lam_poll = getattr(args, 'mide_lambda_poll', 0.2)
        lam_sparse = getattr(args, 'mide_lambda_sparse', 0.1)

        def _safe(x):
            if not torch.is_tensor(x):
                return x
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
            return x

        l_uni, l_uce, l_rank = _safe(l_uni), _safe(l_uce), _safe(l_rank)
        l_noinfo, l_poll, l_sparse = _safe(l_noinfo), _safe(l_poll), _safe(l_sparse)

        losses['uni_pred'] = l_uni
        losses['utility_ce'] = l_uce
        losses['utility_rank'] = l_rank
        losses['noinfo'] = l_noinfo
        losses['pollution'] = l_poll
        losses['sparse_keep'] = l_sparse

        losses['total_mide'] = (
            lam_uni * l_uni
            + lam_uce * l_uce
            + lam_rank * l_rank
            + lam_noinfo * l_noinfo
            + lam_poll * l_poll
            + lam_sparse * l_sparse
        )

        with torch.no_grad():
            stats = {
                'A_mean': a_vec.mean(dim=0).cpu(),
                'U_mean': torch.sigmoid(u_logits).mean(dim=0).cpu(),
                'P_mean': torch.sigmoid(p_logits).mean(dim=0).cpu(),
                'D_mean': d_task.mean(dim=0).cpu(),
                'C_mean': c_vec.mean(dim=0).cpu(),
                'C_std': c_vec.std(dim=0).cpu(),
            }

        return losses, stats

    @staticmethod
    def apply_density_weighting(
        text_rep,
        audio_rep,
        vision_rep,
        text_rep_common,
        audio_rep_common,
        vision_rep_common,
        audio_visual_fusion,
        d_task,
    ):
        d_t = d_task[:, 0:1]
        d_a = d_task[:, 1:2]
        d_v = d_task[:, 2:3]

        text_rep_w = d_t * text_rep
        audio_rep_w = d_a * audio_rep
        vision_rep_w = d_v * vision_rep
        text_common_w = d_t * text_rep_common
        audio_common_w = d_a * audio_rep_common
        vision_common_w = d_v * vision_rep_common

        d_mean = d_task.mean(dim=1, keepdim=True)
        av_fusion_w = d_mean * audio_visual_fusion

        return (
            text_rep_w,
            audio_rep_w,
            vision_rep_w,
            text_common_w,
            audio_common_w,
            vision_common_w,
            av_fusion_w,
            d_t,
            d_a,
            d_v,
        )
