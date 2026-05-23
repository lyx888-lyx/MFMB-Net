import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.missingTask.MFMB_NET.modules.transformer import TransformerEncoder
from einops.layers.torch import Rearrange


class MLP_block(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MLP_Communicator(nn.Module):
    def __init__(self, token, channel, hidden_size, depth=2):
        super(MLP_Communicator, self).__init__()
        self.depth = depth
        self.token_mixer = nn.Sequential(
            Rearrange('b n d -> b d n'),
            MLP_block(input_size=channel, hidden_size=hidden_size),
            Rearrange('b n d -> b d n'),
        )
        self.channel_mixer = nn.Sequential(
            MLP_block(input_size=token, hidden_size=hidden_size)
        )

    def forward(self, x):
        for _ in range(self.depth):
            x = x + self.token_mixer(x)
            x = x + self.channel_mixer(x)
        return x


class BottleAttentionNet(nn.Module):
    def __init__(self):
        super(BottleAttentionNet, self).__init__()
        self.embed_dim = 90  # mosi/mosei after alignment
        self.embed_dim_common = 32
        self.seq = 4

        self.layer_unimodal = 2
        self.layer_multimodal = 2
        self.Linear_common = nn.Linear(self.embed_dim, self.embed_dim_common)
        self.transformer = TransformerEncoder(embed_dim=self.embed_dim, num_heads=10, layers=4, attn_mask=False)

    def forward(self, audio, visual, text):
        for _ in range(self.layer_unimodal):
            audio = self.transformer(audio)

        for _ in range(self.layer_unimodal):
            visual = self.transformer(visual)

        fsn = torch.zeros(self.seq, audio.size(1), self.embed_dim, device=audio.device)

        x = torch.cat([audio, fsn], dim=0)

        for i in range(self.layer_multimodal):
            if i == 0:
                # [audio,fsn]-[fsn',visual]-[fsn'',text]-----[fsn''']
                x = self.transformer(x)
                x = torch.cat([x[audio.size(0):, :, :], visual], dim=0)
                x = self.transformer(x)
                x = torch.cat([x[:self.seq, :, :], text], dim=0)
                x = self.transformer(x)
            else:
                x = self.transformer(x)

        x = x[:self.seq, :, :]
        return x


class GRUencoder(nn.Module):
    """Pad for utterances with variable lengths and maintain the order after GRU."""

    def __init__(self, embedding_dim, utterance_dim, num_layers):
        super(GRUencoder, self).__init__()
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=utterance_dim,
            bidirectional=True,
            num_layers=num_layers,
        )

    def forward(self, utterance, utterance_lens):
        """Args:
            utterance: [batch, max_word_len, embedding_dim]
            utterance_lens: [batch]
        Returns:
            [batch, max_word_len, 2 * utterance_dim]
        """
        utterance_embs = utterance.transpose(0, 1)

        sorted_utter_length, indices = torch.sort(utterance_lens, descending=True)
        _, indices_unsort = torch.sort(indices)

        s_embs = utterance_embs.index_select(1, indices)

        utterance_packed = pack_padded_sequence(s_embs, sorted_utter_length.cpu())
        utterance_output = self.gru(utterance_packed)[0]
        utterance_output = pad_packed_sequence(utterance_output, total_length=utterance.size(1))[0]

        utterance_output = utterance_output.index_select(1, indices_unsort)

        return utterance_output.transpose(0, 1)


class C_GATE(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, drop):
        super(C_GATE, self).__init__()

        self.gru = GRUencoder(embedding_dim, hidden_dim, num_layers)
        self.cnn = nn.Conv1d(in_channels=2 * hidden_dim, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(hidden_dim * 2 + embedding_dim, hidden_dim)
        self.dropout_in = nn.Dropout(drop)

    def forward(self, utterance, utterance_mask):
        add_zero = torch.zeros(size=[utterance.shape[0], 1], requires_grad=False).type_as(utterance_mask).to(utterance_mask.device)
        utterance_mask = torch.cat((utterance_mask, add_zero), dim=1)
        utterance_lens = torch.argmin(utterance_mask, dim=1)

        transformed_ = self.gru(utterance, utterance_lens)

        gate = torch.sigmoid(self.cnn(transformed_.transpose(1, 2)).transpose(1, 2))
        gate_x = torch.tanh(transformed_) * gate

        utterance_rep = torch.tanh(self.fc(torch.cat([utterance, gate_x], dim=-1)))
        utterance_rep = torch.max(utterance_rep, dim=1)[0]
        utterance_rep = self.dropout_in(utterance_rep)

        return utterance_rep


class ReliabilityAnchorRouter(nn.Module):
    """Sample-level dynamic anchor router.

    Notes:
    - In this codebase, `*_missing_mask` from data loader is actually an observed mask
      (1 means preserved/observed position, not missing position).
    - The router uses modality reliability cues (representation + availability) to
      choose a center anchor ahead of micro-fusion for each sample.
    """

    def __init__(self, common_size, hidden_dim=64, dropout=0.1):
        super().__init__()
        in_dim = common_size * 6 + 3
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    @staticmethod
    def _availability(valid_mask, observed_mask):
        valid_denom = valid_mask.sum(dim=1).clamp_min(1.0)
        observed_num = observed_mask.sum(dim=1)
        return (observed_num / valid_denom).unsqueeze(-1)

    def forward(
        self,
        text_rep_common,
        audio_rep_common,
        vision_rep_common,
        text_mask,
        audio_mask,
        vision_mask,
        text_observed_mask,
        audio_observed_mask,
        vision_observed_mask,
        temperature=1.0,
        missing_bias=2.0,
        use_missing=True,
    ):
        avail_t = self._availability(text_mask, text_observed_mask)
        avail_a = self._availability(audio_mask, audio_observed_mask)
        avail_v = self._availability(vision_mask, vision_observed_mask)

        abs_ta = torch.abs(text_rep_common - audio_rep_common)
        abs_tv = torch.abs(text_rep_common - vision_rep_common)
        abs_av = torch.abs(audio_rep_common - vision_rep_common)

        router_in = torch.cat(
            [
                text_rep_common,
                audio_rep_common,
                vision_rep_common,
                avail_t,
                avail_a,
                avail_v,
                abs_ta,
                abs_tv,
                abs_av,
            ],
            dim=-1,
        )

        logits = self.mlp(router_in)
        availability = torch.cat([avail_t, avail_a, avail_v], dim=-1)
        if use_missing:
            logits = logits + missing_bias * availability

        weights = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
        return logits, weights, availability


class GATE_F(nn.Module):
    def __init__(self, args):
        super(GATE_F, self).__init__()

        self.args = args
        self.fusion_center_mode = getattr(args, 'fusion_center_mode', 'text')
        self.use_anchor_moe = bool(getattr(args, 'use_anchor_moe', 0))
        self.router_temperature = float(getattr(args, 'router_temperature', 1.0))
        self.router_missing_bias = float(getattr(args, 'router_missing_bias', 2.0))
        self.router_use_missing = bool(getattr(args, 'router_use_missing', 1))
        self.router_balance_lambda = float(getattr(args, 'router_balance_lambda', 0.0))

        self.text_encoder = C_GATE(args.fusion_t_in, args.fusion_t_hid, args.fusion_gru_layers, args.fusion_drop)
        self.audio_encoder = C_GATE(args.fusion_a_in, args.fusion_a_hid, args.fusion_gru_layers, args.fusion_drop)
        self.vision_encoder = C_GATE(args.fusion_v_in, args.fusion_v_hid, args.fusion_gru_layers, args.fusion_drop)

        self.audio_visual_model = BottleAttentionNet()

        # temporal projections for AV fusion
        self.seq_v_mosi = nn.Linear(500, 50)
        self.seq_a_mosi = nn.Linear(375, 50)
        self.seq_v_mosei = nn.Linear(500, 50)
        self.seq_a_mosei = nn.Linear(500, 50)

        self.common_size = 64

        self.t_stack_linear = nn.Linear(36, self.common_size)
        self.v_stack_linear = nn.Linear(48, self.common_size)
        self.a_stack_linear = nn.Linear(20, self.common_size)

        self.pair_specific_bn = nn.ModuleDict({
            'tv': nn.BatchNorm1d(2, affine=False),
            'ta': nn.BatchNorm1d(2, affine=False),
            'at': nn.BatchNorm1d(2, affine=False),
            'av': nn.BatchNorm1d(2, affine=False),
            'vt': nn.BatchNorm1d(2, affine=False),
            'va': nn.BatchNorm1d(2, affine=False),
        })

        self.anchor_router = ReliabilityAnchorRouter(
            common_size=self.common_size,
            hidden_dim=int(getattr(args, 'router_hidden_dim', 64)),
            dropout=float(getattr(args, 'router_dropout', 0.1)),
        )

        self.classifier1 = nn.Sequential()
        self.classifier1.add_module('linear_trans_norm', nn.BatchNorm1d(self.common_size * 4 + 90))
        self.classifier1.add_module('linear_trans_hidden', nn.Linear(self.common_size * 4 + 90, args.cls_hidden_dim))
        self.classifier1.add_module('linear_trans_activation', nn.LeakyReLU())
        self.classifier1.add_module('linear_trans_drop', nn.Dropout(args.cls_dropout))
        self.classifier1.add_module('linear_trans_final', nn.Linear(args.cls_hidden_dim, 1))

        full_in_dim = args.fusion_t_hid + args.fusion_a_hid + args.fusion_v_hid + self.common_size * 4 + 90
        self.classifier2 = nn.Sequential()
        self.classifier2.add_module('linear_trans_norm', nn.BatchNorm1d(full_in_dim))
        self.classifier2.add_module('linear_trans_hidden', nn.Linear(full_in_dim, args.cls_hidden_dim))
        self.classifier2.add_module('linear_trans_activation', nn.ReLU())
        self.classifier2.add_module('linear_trans_drop', nn.Dropout(args.cls_dropout))
        self.classifier2.add_module('linear_trans_final', nn.Linear(args.cls_hidden_dim, 1))

        # Lightweight Anchor-MoE heads: one expert per anchor center.
        if self.use_anchor_moe:
            expert_hidden = max(args.cls_hidden_dim // 2, 16)
            self.expert_t = self._build_expert_head(full_in_dim, expert_hidden, args.cls_dropout)
            self.expert_a = self._build_expert_head(full_in_dim, expert_hidden, args.cls_dropout)
            self.expert_v = self._build_expert_head(full_in_dim, expert_hidden, args.cls_dropout)

        self.last_router_info = {}

    @staticmethod
    def _build_expert_head(input_dim, hidden_dim, dropout):
        return nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _pair_stack(self, center_common, other_common, pair_key):
        # center_common/other_common: [B, common_size] -> [B, 2*common_size]
        stack_rep = torch.stack((center_common, other_common), dim=1)
        stack_rep = self.pair_specific_bn[pair_key](stack_rep)
        bsz = center_common.size(0)
        return stack_rep.reshape(bsz, -1)

    def _build_center_micro(self, center_name, t_common, a_common, v_common):
        if center_name == 'text':
            pair1 = self._pair_stack(t_common, v_common, 'tv')
            pair2 = self._pair_stack(t_common, a_common, 'ta')
        elif center_name == 'audio':
            pair1 = self._pair_stack(a_common, t_common, 'at')
            pair2 = self._pair_stack(a_common, v_common, 'av')
        elif center_name == 'vision':
            pair1 = self._pair_stack(v_common, t_common, 'vt')
            pair2 = self._pair_stack(v_common, a_common, 'va')
        else:
            raise ValueError(f'Unsupported center name: {center_name}')
        return torch.cat([pair1, pair2], dim=1)

    def _resolve_router_weights(
        self,
        t_common,
        a_common,
        v_common,
        text_mask,
        audio_mask,
        vision_mask,
        text_observed_mask,
        audio_observed_mask,
        vision_observed_mask,
        device,
    ):
        bsz = t_common.size(0)
        center_mode = self.fusion_center_mode

        if center_mode in ['text', 'audio', 'vision']:
            weights = torch.zeros(bsz, 3, device=device)
            idx = {'text': 0, 'audio': 1, 'vision': 2}[center_mode]
            weights[:, idx] = 1.0
            availability = torch.stack([
                text_observed_mask.sum(dim=1) / text_mask.sum(dim=1).clamp_min(1.0),
                audio_observed_mask.sum(dim=1) / audio_mask.sum(dim=1).clamp_min(1.0),
                vision_observed_mask.sum(dim=1) / vision_mask.sum(dim=1).clamp_min(1.0),
            ], dim=-1)
            return None, weights, availability

        logits, soft_weights, availability = self.anchor_router(
            text_rep_common=t_common,
            audio_rep_common=a_common,
            vision_rep_common=v_common,
            text_mask=text_mask,
            audio_mask=audio_mask,
            vision_mask=vision_mask,
            text_observed_mask=text_observed_mask,
            audio_observed_mask=audio_observed_mask,
            vision_observed_mask=vision_observed_mask,
            temperature=self.router_temperature,
            missing_bias=self.router_missing_bias,
            use_missing=self.router_use_missing,
        )

        # dynamic_hard: keep train-time differentiable soft routing; in eval use argmax hard routing.
        if center_mode == 'dynamic_hard' and (not self.training):
            hard_idx = torch.argmax(soft_weights, dim=-1)
            hard_weights = F.one_hot(hard_idx, num_classes=3).float()
            return logits, hard_weights, availability
        return logits, soft_weights, availability

    @staticmethod
    def _router_balance_loss(router_weights):
        eps = 1e-8
        avg_w = router_weights.mean(dim=0)
        return torch.sum(avg_w * torch.log(avg_w * 3.0 + eps))

    def _project_for_av_fusion(self, audio_x, vision_x):
        # keep original behavior for MOSI and be compatible with MOSEI
        if str(getattr(self.args, 'datasetName', 'mosi')).lower() == 'mosei':
            audio_proj = self.seq_a_mosei(audio_x.permute(0, 2, 1))
            vision_proj = self.seq_v_mosei(vision_x.permute(0, 2, 1))
        else:
            audio_proj = self.seq_a_mosi(audio_x.permute(0, 2, 1))
            vision_proj = self.seq_v_mosi(vision_x.permute(0, 2, 1))

        return audio_proj.permute(2, 0, 1), vision_proj.permute(2, 0, 1)

    def forward(self, text_x, audio_x, vision_x):
        # tuple includes observed mask in missing-data mode.
        if len(text_x) == 3:
            text_x, text_mask, text_observed_mask = text_x
        else:
            text_x, text_mask = text_x
            text_observed_mask = text_mask

        if len(audio_x) == 3:
            audio_x, audio_mask, audio_observed_mask = audio_x
        else:
            audio_x, audio_mask = audio_x
            audio_observed_mask = audio_mask

        if len(vision_x) == 3:
            vision_x, vision_mask, vision_observed_mask = vision_x
        else:
            vision_x, vision_mask = vision_x
            vision_observed_mask = vision_mask

        text_x_fusion = text_x.permute(1, 0, 2)
        audio_x_fusion, vision_x_fusion = self._project_for_av_fusion(audio_x, vision_x)

        audio_visual_fusion = self.audio_visual_model(audio_x_fusion, vision_x_fusion, text_x_fusion)
        audio_visual_fusion = audio_visual_fusion[-1]

        text_rep = self.text_encoder(text_x, text_mask)
        audio_rep = self.audio_encoder(audio_x, audio_mask)
        vision_rep = self.vision_encoder(vision_x, vision_mask)

        text_rep_common = self.t_stack_linear(text_rep)
        audio_rep_common = self.a_stack_linear(audio_rep)
        vision_rep_common = self.v_stack_linear(vision_rep)

        h_text_center = self._build_center_micro('text', text_rep_common, audio_rep_common, vision_rep_common)
        h_audio_center = self._build_center_micro('audio', text_rep_common, audio_rep_common, vision_rep_common)
        h_vision_center = self._build_center_micro('vision', text_rep_common, audio_rep_common, vision_rep_common)

        _, router_weights, availability = self._resolve_router_weights(
            text_rep_common,
            audio_rep_common,
            vision_rep_common,
            text_mask,
            audio_mask,
            vision_mask,
            text_observed_mask,
            audio_observed_mask,
            vision_observed_mask,
            device=text_rep.device,
        )

        # weighted micro representation [B, 4*common_size]
        micro_stack = torch.stack([h_text_center, h_audio_center, h_vision_center], dim=1)
        weighted_micro = torch.sum(router_weights.unsqueeze(-1) * micro_stack, dim=1)

        if self.use_anchor_moe:
            feat_base = [text_rep, audio_rep, vision_rep]
            feat_t = torch.cat(feat_base + [h_text_center, audio_visual_fusion], dim=1)
            feat_a = torch.cat(feat_base + [h_audio_center, audio_visual_fusion], dim=1)
            feat_v = torch.cat(feat_base + [h_vision_center, audio_visual_fusion], dim=1)

            pred_t = self.expert_t(feat_t)
            pred_a = self.expert_a(feat_a)
            pred_v = self.expert_v(feat_v)
            pred_stack = torch.cat([pred_t, pred_a, pred_v], dim=1)
            pred = torch.sum(router_weights * pred_stack, dim=1, keepdim=True)
        else:
            utterance_rep = torch.cat(
                (text_rep, audio_rep, vision_rep, weighted_micro, audio_visual_fusion),
                dim=1,
            )
            pred = self.classifier2(utterance_rep)

        selected_idx = torch.argmax(router_weights, dim=-1)

        self.last_router_info = {
            'router_weights': router_weights.detach(),
            'availability': availability.detach(),
            'selected_anchor_idx': selected_idx.detach(),
        }

        if self.router_balance_lambda > 0 and self.fusion_center_mode in ['dynamic_soft', 'dynamic_hard']:
            balance_loss = self._router_balance_loss(router_weights)
            return pred, self.router_balance_lambda * balance_loss
        return pred


MODULE_MAP = {
    'c_gate': GATE_F,
}


class Fusion(nn.Module):
    def __init__(self, args):
        super(Fusion, self).__init__()

        select_model = MODULE_MAP[args.fusionModule]
        self.Model = select_model(args)

    def forward(self, text_x, audio_x, vision_x):
        return self.Model(text_x, audio_x, vision_x)
