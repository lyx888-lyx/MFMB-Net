import torch
import torch.nn as nn
import torch.nn.functional as F


class DecAlignAdapter(nn.Module):
    """
    MFMB + DecAlign v2 (stable version for missing-modality setting).

    Main changes vs the first hybrid attempt:
      1) keep decoupling (private/shared) but DROP private/hete OT alignment
      2) only align shared/common features
      3) use TEXT as the anchor; audio/vision are pulled toward text
      4) text anchor can be detached so noisy A/V gradients do not drag text
      5) refinement only updates audio/vision; text stays unchanged
      6) alignment is utterance-level, not sequence-level, for higher stability
    """

    def __init__(self, args, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.residual_ratio = getattr(args, 'align_residual_ratio', 0.10)
        self.mmd_bandwidth = getattr(args, 'align_mmd_bandwidth', 10.0)
        self.detach_text_anchor = getattr(args, 'align_detach_text_anchor', True)
        self.anchor_cosine_weight = getattr(args, 'align_anchor_cosine_weight', 0.5)

        # modality-private branch
        self.uni_t = nn.Conv1d(in_dim, in_dim, kernel_size=1, bias=False)
        self.uni_a = nn.Conv1d(in_dim, in_dim, kernel_size=1, bias=False)
        self.uni_v = nn.Conv1d(in_dim, in_dim, kernel_size=1, bias=False)
        # shared/common branch (shared weights across modalities)
        self.com = nn.Conv1d(in_dim, in_dim, kernel_size=1, bias=False)

        # refine only A/V using their own shared features
        self.refine_a = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim),
            nn.Sigmoid(),
        )
        self.refine_v = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim),
            nn.Sigmoid(),
        )

    def masked_mean(self, x, mask):
        mask = mask.float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (x * mask).sum(dim=1) / denom

    def batch_mean_var(self, pooled):
        mean = pooled.mean(dim=0)
        var = pooled.var(dim=0, unbiased=False)
        return mean, var

    def dec_loss_one(self, private_feat, shared_feat, mask):
        p = self.masked_mean(private_feat, mask)
        s = self.masked_mean(shared_feat, mask)
        return F.cosine_similarity(p, s, dim=-1).abs().mean()

    def compute_mmd(self, x, y):
        # x, y: [B, D]
        bw = max(self.mmd_bandwidth, 1e-4)
        xx = torch.cdist(x, x, p=2).pow(2)
        yy = torch.cdist(y, y, p=2).pow(2)
        xy = torch.cdist(x, y, p=2).pow(2)
        k_xx = torch.exp(-xx / (2.0 * bw))
        k_yy = torch.exp(-yy / (2.0 * bw))
        k_xy = torch.exp(-xy / (2.0 * bw))
        return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()

    def shared_anchor_loss(self, pooled_text, pooled_other):
        mean_t, var_t = self.batch_mean_var(pooled_text)
        mean_o, var_o = self.batch_mean_var(pooled_other)
        stat = (mean_t - mean_o).pow(2).mean() + (var_t - var_o).pow(2).mean()
        cos = (1.0 - F.cosine_similarity(pooled_other, pooled_text, dim=-1)).mean()
        mmd = self.compute_mmd(pooled_other, pooled_text)
        return stat + self.anchor_cosine_weight * cos + mmd

    def refine_other(self, raw, private_feat, shared_feat, gate_net):
        gate = gate_net(torch.cat([private_feat, shared_feat], dim=-1))
        resid = gate * shared_feat
        return raw + self.residual_ratio * resid

    def forward(self, text_h, audio_h, vision_h, text_mask, audio_mask, vision_mask):
        t = text_h.transpose(1, 2)
        a = audio_h.transpose(1, 2)
        v = vision_h.transpose(1, 2)

        p_t = self.uni_t(t).transpose(1, 2)
        p_a = self.uni_a(a).transpose(1, 2)
        p_v = self.uni_v(v).transpose(1, 2)
        s_t = self.com(t).transpose(1, 2)
        s_a = self.com(a).transpose(1, 2)
        s_v = self.com(v).transpose(1, 2)

        dec_loss = (
            self.dec_loss_one(p_t, s_t, text_mask) +
            self.dec_loss_one(p_a, s_a, audio_mask) +
            self.dec_loss_one(p_v, s_v, vision_mask)
        ) / 3.0

        pooled_t = self.masked_mean(s_t, text_mask)
        pooled_a = self.masked_mean(s_a, audio_mask)
        pooled_v = self.masked_mean(s_v, vision_mask)

        if self.detach_text_anchor:
            pooled_t = pooled_t.detach()

        # shared-only, text-anchor homo alignment
        homo_loss = (
            self.shared_anchor_loss(pooled_t, pooled_a) +
            self.shared_anchor_loss(pooled_t, pooled_v)
        ) / 2.0

        # disable heterogeneity/private OT for missing-modality setting
        hete_loss = torch.zeros((), device=text_h.device, dtype=text_h.dtype)

        # keep text untouched; refine only audio/vision to avoid contaminating the anchor
        text_refined = text_h
        audio_refined = self.refine_other(audio_h, p_a, s_a, self.refine_a)
        vision_refined = self.refine_other(vision_h, p_v, s_v, self.refine_v)

        return {
            'text_refined': text_refined,
            'audio_refined': audio_refined,
            'vision_refined': vision_refined,
            'dec_loss': dec_loss,
            'hete_loss': hete_loss,
            'homo_loss': homo_loss,
        }
