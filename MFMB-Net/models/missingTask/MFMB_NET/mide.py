import torch
import torch.nn as nn
import torch.nn.functional as F


class LightScalarHead(nn.Module):
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

    return obs_t / valid_t, obs_a / valid_a, obs_v / valid_v


def build_targets(
    y_t,
    y_a,
    y_v,
    y_all_ref,
    y_without_t_ref,
    y_without_a_ref,
    y_without_v_ref,
    labels,
    args,
):
    labels = labels.view_as(y_all_ref)
    e_t = (y_t - labels).abs()
    e_a = (y_a - labels).abs()
    e_v = (y_v - labels).abs()
    e_vec = torch.cat([e_t, e_a, e_v], dim=1).detach()

    err_all = (y_all_ref - labels).abs().detach()
    err_wo_t = (y_without_t_ref - labels).abs().detach()
    err_wo_a = (y_without_a_ref - labels).abs().detach()
    err_wo_v = (y_without_v_ref - labels).abs().detach()
    c_vec = torch.cat([
        err_wo_t - err_all,
        err_wo_a - err_all,
        err_wo_v - err_all,
    ], dim=1).detach()

    tau_uni = getattr(args, 'mide_tau_uni', 1.0)
    tau_loo = getattr(args, 'mide_tau_loo', 0.6)
    tau_p = getattr(args, 'mide_tau_p', 0.5)
    beta_uni = getattr(args, 'mide_beta_uni', 0.4)
    beta_loo = getattr(args, 'mide_beta_loo', 0.6)
    neg_threshold = getattr(args, 'mide_neg_threshold', 0.01)

    u_uni = torch.exp(-e_vec / tau_uni)
    u_uni_norm = u_uni / (u_uni.max(dim=1, keepdim=True).values + 1e-6)

    scale = err_all.mean(dim=1, keepdim=True) + 1e-6
    c_norm = torch.tanh(c_vec / scale)
    u_loo = torch.sigmoid(c_norm / tau_loo)

    u_target = torch.clamp(beta_uni * u_uni_norm + beta_loo * u_loo, 0.0, 1.0)
    p_target = torch.sigmoid((-c_norm - neg_threshold) / tau_p)

    return {
        'e_vec': e_vec,
        'c_vec': c_vec,
        'c_norm': c_norm,
        'u_target': u_target.detach(),
        'p_target': p_target.detach(),
        'err_all': err_all,
    }


def compute_d_floor(epoch, args):
    gate_start = getattr(args, 'mide_gate_start_epoch', 2)
    floor_min = getattr(args, 'mide_d_floor_min', 0.65)
    full_start = getattr(args, 'mide_full_start_epoch', 4)
    if epoch <= gate_start:
        return 1.0
    span = max(full_start - gate_start, 1)
    progress = min(1.0, max(0.0, (epoch - gate_start) / span))
    return 1.0 - progress * (1.0 - floor_min)


def gate_enabled(epoch, args):
    return epoch > getattr(args, 'mide_gate_start_epoch', 2)


def compute_d_eff(a_vec, u_logits, p_logits, epoch, args):
    u_pred = torch.sigmoid(u_logits)
    p_pred = torch.sigmoid(p_logits)
    d_raw = a_vec * u_pred * (1.0 - p_pred)
    if not gate_enabled(epoch, args):
        ones = torch.ones_like(d_raw)
        return ones, d_raw
    d_floor = compute_d_floor(epoch, args)
    d_eff = d_floor + (1.0 - d_floor) * d_raw
    return d_eff.clamp(min=1e-3, max=1.0), d_raw


def get_training_phase(epoch, args):
    aux_start = getattr(args, 'mide_aux_start_epoch', 2)
    full_start = getattr(args, 'mide_full_start_epoch', 4)
    if epoch <= aux_start:
        return 1
    if epoch <= full_start:
        return 2
    return 3


class SplitAUPController(nn.Module):
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

        for head in (self.util_head_t, self.util_head_a, self.util_head_v):
            nn.init.zeros_(head.net[-1].weight)
            nn.init.zeros_(head.net[-1].bias)
        for head in (self.poll_head_t, self.poll_head_a, self.poll_head_v):
            nn.init.zeros_(head.net[-1].weight)
            nn.init.constant_(head.net[-1].bias, -2.0)

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

    def compute_losses(
        self,
        y_t,
        y_a,
        y_v,
        u_logits,
        p_logits,
        a_vec,
        d_eff,
        labels,
        y_all_ref,
        y_without_t_ref,
        y_without_a_ref,
        y_without_v_ref,
        epoch,
        enable_mide_losses=True,
    ):
        args = self.args
        device = labels.device
        zero = torch.tensor(0.0, device=device)
        labels = labels.view_as(y_all_ref)

        l_uni = (
            F.smooth_l1_loss(y_t, labels)
            + F.smooth_l1_loss(y_a, labels)
            + F.smooth_l1_loss(y_v, labels)
        )

        losses = {
            'uni_pred': l_uni,
            'util_bce': zero,
            'utility_rank': zero,
            'poll_bce': zero,
            'noinfo': zero,
            'pollution': zero,
            'sparse_keep': zero,
            'sep': zero,
        }
        stats = {}

        if not enable_mide_losses:
            losses['total_mide'] = zero
            return losses, stats

        phase = get_training_phase(epoch, args)
        targets = build_targets(
            y_t, y_a, y_v,
            y_all_ref, y_without_t_ref, y_without_a_ref, y_without_v_ref,
            labels, args,
        )
        u_target = targets['u_target']
        p_target = targets['p_target']
        c_norm = targets['c_norm']
        u_pred = torch.sigmoid(u_logits)
        p_pred = torch.sigmoid(p_logits)

        l_util_bce = F.binary_cross_entropy_with_logits(u_logits, u_target)
        l_poll_bce = F.binary_cross_entropy_with_logits(p_logits, p_target)

        margin = getattr(args, 'mide_margin', 0.05)
        contrib_eps = getattr(args, 'mide_contrib_eps', 0.01)
        c_vec = targets['c_vec']
        l_rank = zero
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                mask = (c_vec[:, i] - c_vec[:, j]) > contrib_eps
                if mask.any():
                    l_rank = l_rank + F.relu(margin - (u_logits[mask, i] - u_logits[mask, j])).mean()

        avail_high = getattr(args, 'mide_avail_high', 0.8)
        u_low = getattr(args, 'mide_u_low', 0.35)
        l_noinfo = zero
        for m in range(3):
            msk = (a_vec[:, m:m + 1] > avail_high) & (u_target[:, m:m + 1] < u_low)
            if msk.any():
                l_noinfo = l_noinfo + (msk.float() * d_eff[:, m:m + 1]).mean()

        neg_threshold = getattr(args, 'mide_neg_threshold', 0.01)
        l_poll = zero
        for m in range(3):
            l_poll = l_poll + (F.relu(-c_norm[:, m:m + 1] - neg_threshold) * d_eff[:, m:m + 1]).mean()

        avail_low = getattr(args, 'mide_avail_low', 0.35)
        pos_threshold = getattr(args, 'mide_pos_threshold', 0.03)
        keep_target = getattr(args, 'mide_keep_density_target', 0.55)
        l_sparse = zero
        for m in range(3):
            msk = (a_vec[:, m:m + 1] < avail_low) & (c_norm[:, m:m + 1] > pos_threshold)
            if msk.any():
                l_sparse = l_sparse + (msk.float() * F.relu(keep_target - d_eff[:, m:m + 1])).mean()

        sep_eps = getattr(args, 'mide_sep_eps', 0.02)
        sep_margin = getattr(args, 'mide_sep_margin', 0.05)
        c_std = c_norm.std(dim=0).mean()
        l_sep = zero
        if c_std > sep_eps:
            l_sep = F.relu(sep_margin - u_pred.std(dim=0).mean())

        losses.update({
            'util_bce': l_util_bce,
            'utility_rank': l_rank,
            'poll_bce': l_poll_bce,
            'noinfo': l_noinfo,
            'pollution': l_poll,
            'sparse_keep': l_sparse,
            'sep': l_sep,
        })

        lam_uni = getattr(args, 'mide_lambda_uni', 0.08)
        lam_util = getattr(args, 'mide_lambda_util_bce', 0.08)
        lam_rank = getattr(args, 'mide_lambda_rank', 0.03)
        lam_poll_bce = getattr(args, 'mide_lambda_poll_bce', 0.03)
        lam_noinfo = getattr(args, 'mide_lambda_noinfo', 0.02)
        lam_poll = getattr(args, 'mide_lambda_poll', 0.03)
        lam_sparse = getattr(args, 'mide_lambda_sparse', 0.02)
        lam_sep = getattr(args, 'mide_lambda_sep', 0.01)

        total = lam_uni * l_uni
        if phase >= 2:
            total = total + lam_util * l_util_bce + lam_rank * l_rank
        if phase >= 3:
            total = total + (
                lam_poll_bce * l_poll_bce
                + lam_noinfo * l_noinfo
                + lam_poll * l_poll
                + lam_sparse * l_sparse
                + lam_sep * l_sep
            )

        def _safe(x):
            if not torch.is_tensor(x):
                return x
            return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)

        for k in list(losses.keys()):
            losses[k] = _safe(losses[k])
        losses['total_mide'] = _safe(total)

        with torch.no_grad():
            stats = {
                'A_mean': a_vec.mean(dim=0).cpu(),
                'U_mean': u_pred.mean(dim=0).cpu(),
                'P_mean': p_pred.mean(dim=0).cpu(),
                'D_mean': d_eff.mean(dim=0).cpu(),
                'C_mean': c_norm.mean(dim=0).cpu(),
                'C_std': c_norm.std(dim=0).cpu(),
                'phase': phase,
            }
        return losses, stats


class OldMIDEController(nn.Module):
    """Reproduce pre-split MIDE r3 logic for ablation."""

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
        u_logits = torch.cat([self.util_head_t(text_rep), self.util_head_a(audio_rep), self.util_head_v(vision_rep)], dim=1)
        p_logits = torch.cat([self.poll_head_t(text_rep), self.poll_head_a(audio_rep), self.poll_head_v(vision_rep)], dim=1)
        return y_t, y_a, y_v, u_logits, p_logits

    def compute_density(self, a_vec, u_logits, p_logits):
        d = a_vec * torch.sigmoid(u_logits) * (1.0 - torch.sigmoid(p_logits))
        return d.clamp(min=1e-3, max=1.0)

    def compute_losses(self, y_t, y_a, y_v, u_logits, p_logits, a_vec, d_task, labels,
                       y_all, y_without_t, y_without_a, y_without_v, epoch, enable_mide_losses=True):
        args = self.args
        labels = labels.view_as(y_all)
        device = labels.device
        zero = torch.tensor(0.0, device=device)
        l_uni = F.smooth_l1_loss(y_t, labels) + F.smooth_l1_loss(y_a, labels) + F.smooth_l1_loss(y_v, labels)
        losses = {'uni_pred': l_uni, 'utility_ce': zero, 'utility_rank': zero, 'noinfo': zero, 'pollution': zero, 'sparse_keep': zero}
        warmup = getattr(args, 'mide_warmup_epochs', 1)
        if not enable_mide_losses or epoch <= warmup:
            losses['total_mide'] = zero
            return losses, {}
        e = torch.cat([(y_t - labels).abs(), (y_a - labels).abs(), (y_v - labels).abs()], dim=1).detach()
        tau = getattr(args, 'mide_tau', 1.0)
        u_target = F.softmax(-e / tau, dim=1)
        l_uce = F.kl_div(F.log_softmax(u_logits, dim=1), u_target, reduction='batchmean')
        err_all = (y_all - labels).abs().detach()
        c_vec = torch.cat([(y_without_t - labels).abs() - err_all,
                           (y_without_a - labels).abs() - err_all,
                           (y_without_v - labels).abs() - err_all], dim=1)
        margin, eps = getattr(args, 'mide_margin', 0.05), getattr(args, 'mide_contrib_eps', 0.01)
        l_rank = zero
        for i in range(3):
            for j in range(3):
                if i != j:
                    m = (c_vec[:, i] - c_vec[:, j]) > eps
                    if m.any():
                        l_rank += F.relu(margin - (u_logits[m, i] - u_logits[m, j])).mean()
        l_noinfo = zero
        for m in range(3):
            msk = (a_vec[:, m:m+1] > 0.8) & (c_vec[:, m:m+1].abs() < 0.01)
            if msk.any():
                l_noinfo += (msk.float() * d_task[:, m:m+1]).mean()
        l_poll = sum((F.relu(-c_vec[:, m:m+1]) * d_task[:, m:m+1]).mean() for m in range(3))
        l_sparse = zero
        for m in range(3):
            msk = (a_vec[:, m:m+1] < 0.35) & (c_vec[:, m:m+1] > 0.02)
            if msk.any():
                l_sparse += (msk.float() * F.relu(0.45 - d_task[:, m:m+1])).mean()
        losses.update({'utility_ce': l_uce, 'utility_rank': l_rank, 'noinfo': l_noinfo, 'pollution': l_poll, 'sparse_keep': l_sparse})
        losses['total_mide'] = (
            getattr(args, 'mide_lambda_uni', 0.1) * l_uni
            + getattr(args, 'mide_lambda_uce', 0.15) * l_uce
            + getattr(args, 'mide_lambda_rank', 0.05) * l_rank
            + getattr(args, 'mide_lambda_noinfo', 0.03) * l_noinfo
            + getattr(args, 'mide_lambda_poll', 0.05) * l_poll
            + getattr(args, 'mide_lambda_sparse', 0.03) * l_sparse
        )
        stats = {'A_mean': a_vec.mean(0).cpu(), 'U_mean': torch.sigmoid(u_logits).mean(0).cpu(),
                 'P_mean': torch.sigmoid(p_logits).mean(0).cpu(), 'D_mean': d_task.mean(0).cpu(),
                 'C_mean': c_vec.mean(0).cpu(), 'C_std': c_vec.std(0).cpu()}
        return losses, stats

    @staticmethod
    def apply_density_weighting(text_rep, audio_rep, vision_rep, text_common, audio_common, vision_common, av_fusion, d_task):
        d_t, d_a, d_v = d_task[:, 0:1], d_task[:, 1:2], d_task[:, 2:3]
        return (
            d_t * text_rep, d_a * audio_rep, d_v * vision_rep,
            d_t * text_common, d_a * audio_common, d_v * vision_common,
            d_task.mean(1, keepdim=True) * av_fusion, d_t, d_a, d_v,
        )


def create_mide_controller(args):
    variant = getattr(args, 'mide_variant', 'split_aup')
    if variant == 'old':
        return OldMIDEController(args)
    return SplitAUPController(args)


# Backward compat alias
MIDEController = SplitAUPController
