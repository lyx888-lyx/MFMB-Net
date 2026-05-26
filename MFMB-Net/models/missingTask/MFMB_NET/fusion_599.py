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


class ReliabilityTaskGate(nn.Module):
    """Sample-level gate to fuse reliability-aware and task-aware routes."""

    def __init__(self, input_dim, hidden_dim=32, dropout=0.1, init_bias=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # Bias controls initial preference:
        # <0 prefers reliability route, >0 prefers task-aware route.
        with torch.no_grad():
            self.net[-1].bias.fill_(float(init_bias))

    def forward(self, gate_input):
        gate_logit = self.net(gate_input)
        gate_task = torch.sigmoid(gate_logit)
        return gate_logit, gate_task


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
        self.use_task_aware_router = bool(getattr(args, 'use_task_aware_router', 0))
        self.task_router_lambda = float(getattr(args, 'task_router_lambda', 0.1))
        self.center_aux_lambda = float(getattr(args, 'center_aux_lambda', 0.05))
        self.router_oracle_temperature = float(getattr(args, 'router_oracle_temperature', 0.5))
        self.router_oracle_type = str(getattr(args, 'router_oracle_type', 'soft')).lower()
        self.router_task_detach_oracle = bool(getattr(args, 'router_task_detach_oracle', 1))
        self.use_reliability_task_gate = bool(getattr(args, 'use_reliability_task_gate', 0))
        self.rta_gate_mode = str(getattr(args, 'rta_gate_mode', 'learned')).lower()
        self.gate_balance_lambda = float(getattr(args, 'gate_balance_lambda', 0.0))
        self.gate_target = float(getattr(args, 'gate_target', 0.5))
        self.use_prediction_gate_supervision = bool(getattr(args, 'use_prediction_gate_supervision', 1))
        self.gate_oracle_temperature = float(getattr(args, 'gate_oracle_temperature', 0.8))
        self.gate_task_lambda = float(getattr(args, 'gate_task_lambda', 0.02))
        self.rta_pred_residual = bool(getattr(args, 'rta_pred_residual', 0))
        self.gate_supervision_mode = str(getattr(args, 'gate_supervision_mode', 'full')).lower()
        self.gate_margin = float(getattr(args, 'gate_margin', 0.05))

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
        # Separate task-aware router for dynamic_rta so task supervision does not
        # overwrite the pure reliability-aware routing behavior.
        self.task_anchor_router = ReliabilityAnchorRouter(
            common_size=self.common_size,
            hidden_dim=int(getattr(args, 'router_hidden_dim', 64)),
            dropout=float(getattr(args, 'router_dropout', 0.1)),
        )
        # gate_input = availability(3) + entropies(2) + max_conf(2)
        #            + disagreement(1) + route_gap_or_dummy(1) + pred_std/pred_range(2)
        # Keep a unified dimensionality for dynamic_rta and dynamic_rta_pred.
        self.reliability_task_gate = ReliabilityTaskGate(
            input_dim=11,
            hidden_dim=int(getattr(args, 'gate_hidden_dim', 32)),
            dropout=float(getattr(args, 'gate_dropout', 0.1)),
            init_bias=float(getattr(args, 'gate_init_bias', 0.0)),
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

        # Task-aware router auxiliary center heads (used by dynamic_task_soft only).
        aux_hidden = max(args.cls_hidden_dim // 2, 16)
        self.center_aux_text = self._build_expert_head(full_in_dim, aux_hidden, args.cls_dropout)
        self.center_aux_audio = self._build_expert_head(full_in_dim, aux_hidden, args.cls_dropout)
        self.center_aux_vision = self._build_expert_head(full_in_dim, aux_hidden, args.cls_dropout)

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

    def _resolve_task_router_weights(
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
    ):
        return self.task_anchor_router(
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

    @staticmethod
    def _router_balance_loss(router_weights):
        eps = 1e-8
        avg_w = router_weights.mean(dim=0)
        return torch.sum(avg_w * torch.log(avg_w * 3.0 + eps))

    @staticmethod
    def _entropy(weights):
        return -torch.sum(weights * torch.log(weights + 1e-8), dim=-1, keepdim=True)

    def _build_task_router_supervision(
        self,
        y,
        pred_t,
        pred_a,
        pred_v,
        router_logits,
        router_weights,
    ):
        center_aux_loss = (
            F.smooth_l1_loss(pred_t, y)
            + F.smooth_l1_loss(pred_a, y)
            + F.smooth_l1_loss(pred_v, y)
        ) / 3.0

        center_err = torch.cat([
            torch.abs(pred_t - y),
            torch.abs(pred_a - y),
            torch.abs(pred_v - y),
        ], dim=1)
        err_for_oracle = center_err.detach() if self.router_task_detach_oracle else center_err

        if self.router_oracle_type == 'hard':
            oracle_label = torch.argmin(err_for_oracle, dim=-1)
            router_task_loss = F.cross_entropy(router_logits, oracle_label)
            oracle_weight = F.one_hot(oracle_label, num_classes=3).float()
        else:
            oracle_weight = torch.softmax(
                -err_for_oracle / max(self.router_oracle_temperature, 1e-6),
                dim=-1,
            )
            router_task_loss = F.kl_div(
                torch.log(router_weights + 1e-8),
                oracle_weight,
                reduction='batchmean',
            )
            oracle_label = torch.argmax(oracle_weight, dim=-1)

        return center_aux_loss, router_task_loss, center_err, oracle_weight, oracle_label

    def _build_prediction_gate_supervision(self, y, pred_rel, pred_task, gate_task):
        # Two-route oracle: rel vs task prediction errors.
        err_rel = torch.abs(pred_rel - y)
        err_task = torch.abs(pred_task - y)
        err_pair = torch.cat([err_rel, err_task], dim=1)
        err_for_oracle = err_pair.detach() if self.router_task_detach_oracle else err_pair
        oracle_gate = torch.softmax(
            -err_for_oracle / max(self.gate_oracle_temperature, 1e-6),
            dim=-1,
        )
        gate_pred = torch.cat([1.0 - gate_task, gate_task], dim=1)
        abs_err_diff = torch.abs(err_rel - err_task)
        gate_margin_mask = (abs_err_diff > self.gate_margin).squeeze(-1)
        if self.gate_supervision_mode == 'margin':
            if gate_margin_mask.any():
                gate_task_loss = F.kl_div(
                    torch.log(gate_pred[gate_margin_mask] + 1e-8),
                    oracle_gate[gate_margin_mask],
                    reduction='batchmean',
                )
            else:
                gate_task_loss = torch.tensor(0.0, device=gate_task.device)
        else:
            gate_task_loss = F.kl_div(
                torch.log(gate_pred + 1e-8),
                oracle_gate,
                reduction='batchmean',
            )
        oracle_gate_label = torch.argmax(oracle_gate, dim=-1)
        return gate_task_loss, err_rel, err_task, oracle_gate, oracle_gate_label, abs_err_diff, gate_margin_mask

    def _project_for_av_fusion(self, audio_x, vision_x):
        # keep original behavior for MOSI and be compatible with MOSEI
        if str(getattr(self.args, 'datasetName', 'mosi')).lower() == 'mosei':
            audio_proj = self.seq_a_mosei(audio_x.permute(0, 2, 1))
            vision_proj = self.seq_v_mosei(vision_x.permute(0, 2, 1))
        else:
            audio_proj = self.seq_a_mosi(audio_x.permute(0, 2, 1))
            vision_proj = self.seq_v_mosi(vision_x.permute(0, 2, 1))

        return audio_proj.permute(2, 0, 1), vision_proj.permute(2, 0, 1)

    def forward(self, text_x, audio_x, vision_x, labels=None):
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

        router_logits, router_weights, availability = self._resolve_router_weights(
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

        # reliability-aware weighted micro representation [B, 4*common_size]
        micro_stack = torch.stack([h_text_center, h_audio_center, h_vision_center], dim=1)
        weighted_micro_rel = torch.sum(router_weights.unsqueeze(-1) * micro_stack, dim=1)

        center_mode = self.fusion_center_mode
        feat_base = [text_rep, audio_rep, vision_rep]
        feat_t = torch.cat(feat_base + [h_text_center, audio_visual_fusion], dim=1)
        feat_a = torch.cat(feat_base + [h_audio_center, audio_visual_fusion], dim=1)
        feat_v = torch.cat(feat_base + [h_vision_center, audio_visual_fusion], dim=1)

        pred_t = pred_a = pred_v = None
        pred_rel = None
        pred_task_route = None
        task_router_logits = None
        task_router_weights = None
        weighted_micro_task = None
        gate_logit = None
        gate_task = None
        gate_rel = None
        router_disagreement = None
        entropy_rel = self._entropy(router_weights)
        entropy_task = None
        pred_std = None
        pred_range = None
        abs_pred_rel_task = None
        weighted_micro_final = weighted_micro_rel

        if center_mode in ['dynamic_task_soft', 'dynamic_rta', 'dynamic_rta_pred', 'dynamic_rta_pred_residual']:
            # Center-specific auxiliary predictions are used to build task-aware router supervision.
            pred_t = self.center_aux_text(feat_t)
            pred_a = self.center_aux_audio(feat_a)
            pred_v = self.center_aux_vision(feat_v)
            if center_mode in ['dynamic_rta', 'dynamic_rta_pred', 'dynamic_rta_pred_residual']:
                task_router_logits, task_router_weights, _ = self._resolve_task_router_weights(
                    text_rep_common,
                    audio_rep_common,
                    vision_rep_common,
                    text_mask,
                    audio_mask,
                    vision_mask,
                    text_observed_mask,
                    audio_observed_mask,
                    vision_observed_mask,
                )
                weighted_micro_task = torch.sum(task_router_weights.unsqueeze(-1) * micro_stack, dim=1)
                entropy_task = self._entropy(task_router_weights)

                max_w_rel = torch.max(router_weights, dim=-1, keepdim=True)[0]
                max_w_task = torch.max(task_router_weights, dim=-1, keepdim=True)[0]
                router_disagreement = torch.mean(torch.abs(router_weights - task_router_weights), dim=-1, keepdim=True)

                center_preds = torch.cat([pred_t, pred_a, pred_v], dim=1)
                pred_std = torch.std(center_preds, dim=1, keepdim=True, correction=0)
                pred_range = (
                    torch.max(center_preds, dim=1, keepdim=True)[0]
                    - torch.min(center_preds, dim=1, keepdim=True)[0]
                )

                if center_mode in ['dynamic_rta_pred', 'dynamic_rta_pred_residual']:
                    feat_rel = torch.cat((text_rep, audio_rep, vision_rep, weighted_micro_rel, audio_visual_fusion), dim=1)
                    feat_task = torch.cat((text_rep, audio_rep, vision_rep, weighted_micro_task, audio_visual_fusion), dim=1)
                    pred_rel = self.classifier2(feat_rel)
                    pred_task_route = self.classifier2(feat_task)
                    abs_pred_rel_task = torch.abs(pred_rel - pred_task_route)
                    gate_input = torch.cat(
                        [
                            availability,
                            entropy_rel,
                            entropy_task,
                            max_w_rel,
                            max_w_task,
                            router_disagreement,
                            abs_pred_rel_task,
                            pred_std,
                            pred_range,
                        ],
                        dim=1,
                    )
                else:
                    # For feature-level RTA we keep gate input dimensionality aligned
                    # with prediction-level RTA by using a zero dummy route-gap feature.
                    dummy_route_gap = torch.zeros_like(pred_std)
                    gate_input = torch.cat(
                        [
                            availability,
                            entropy_rel,
                            entropy_task,
                            max_w_rel,
                            max_w_task,
                            router_disagreement,
                            dummy_route_gap,
                            pred_std,
                            pred_range,
                        ],
                        dim=1,
                    )

                # Gate ablations for feature/prediction-level RTA.
                gm = self.rta_gate_mode
                if gm == 'force_rel':
                    gate_task = torch.zeros_like(max_w_rel)
                    gate_logit = torch.full_like(max_w_rel, -20.0)
                elif gm == 'force_task':
                    gate_task = torch.ones_like(max_w_rel)
                    gate_logit = torch.full_like(max_w_rel, 20.0)
                elif gm == 'fixed_half':
                    gate_task = torch.full_like(max_w_rel, 0.5)
                    gate_logit = torch.zeros_like(max_w_rel)
                else:
                    if self.use_reliability_task_gate:
                        gate_logit, gate_task = self.reliability_task_gate(gate_input)
                    else:
                        gate_task = torch.full_like(max_w_rel, 0.5)
                        gate_logit = torch.zeros_like(gate_task)
                gate_rel = 1.0 - gate_task

                if center_mode in ['dynamic_rta_pred', 'dynamic_rta_pred_residual']:
                    if center_mode == 'dynamic_rta_pred_residual' or self.rta_pred_residual:
                        pred = pred_rel + gate_task * (pred_task_route - pred_rel)
                    else:
                        pred = gate_rel * pred_rel + gate_task * pred_task_route
                    weighted_micro_final = weighted_micro_rel
                else:
                    weighted_micro_final = gate_rel * weighted_micro_rel + gate_task * weighted_micro_task
                    utterance_rep = torch.cat((text_rep, audio_rep, vision_rep, weighted_micro_final, audio_visual_fusion), dim=1)
                    pred = self.classifier2(utterance_rep)
            else:
                weighted_micro_final = weighted_micro_rel
                utterance_rep = torch.cat((text_rep, audio_rep, vision_rep, weighted_micro_final, audio_visual_fusion), dim=1)
                pred = self.classifier2(utterance_rep)
        elif self.use_anchor_moe:
            feat_base = [text_rep, audio_rep, vision_rep]

            pred_t = self.expert_t(feat_t)
            pred_a = self.expert_a(feat_a)
            pred_v = self.expert_v(feat_v)
            pred_stack = torch.cat([pred_t, pred_a, pred_v], dim=1)
            pred = torch.sum(router_weights * pred_stack, dim=1, keepdim=True)
        else:
            utterance_rep = torch.cat(
                (text_rep, audio_rep, vision_rep, weighted_micro_rel, audio_visual_fusion),
                dim=1,
            )
            pred = self.classifier2(utterance_rep)

        selected_idx = torch.argmax(router_weights, dim=-1)
        selected_task_idx = torch.argmax(task_router_weights, dim=-1) if task_router_weights is not None else None

        task_aux_loss = torch.tensor(0.0, device=pred.device)
        center_aux_loss = torch.tensor(0.0, device=pred.device)
        router_task_loss = torch.tensor(0.0, device=pred.device)
        gate_task_loss = torch.tensor(0.0, device=pred.device)
        gate_balance_loss = torch.tensor(0.0, device=pred.device)
        oracle_weight = None
        oracle_label = None
        center_err = None
        gate_oracle = None
        gate_oracle_label = None
        err_rel = None
        err_task = None

        # Task-aware router supervision is train-only and must not use labels at inference time.
        if (
            center_mode in ['dynamic_task_soft', 'dynamic_rta', 'dynamic_rta_pred', 'dynamic_rta_pred_residual']
            and self.use_task_aware_router
            and self.training
            and labels is not None
            and pred_t is not None
            and (
                (center_mode == 'dynamic_task_soft' and router_logits is not None)
                or (center_mode in ['dynamic_rta', 'dynamic_rta_pred', 'dynamic_rta_pred_residual'] and task_router_logits is not None)
            )
        ):
            y = labels
            if y.dim() == 1:
                y = y.unsqueeze(-1)
            y = y.to(pred.device)

            if center_mode in ['dynamic_rta', 'dynamic_rta_pred', 'dynamic_rta_pred_residual']:
                ref_logits = task_router_logits
                ref_weights = task_router_weights
            else:
                ref_logits = router_logits
                ref_weights = router_weights

            center_aux_loss, router_task_loss, center_err, oracle_weight, oracle_label = self._build_task_router_supervision(
                y=y,
                pred_t=pred_t,
                pred_a=pred_a,
                pred_v=pred_v,
                router_logits=ref_logits,
                router_weights=ref_weights,
            )

            task_aux_loss = self.center_aux_lambda * center_aux_loss + self.task_router_lambda * router_task_loss

            if center_mode in ['dynamic_rta', 'dynamic_rta_pred', 'dynamic_rta_pred_residual'] and gate_task is not None and self.gate_balance_lambda > 0:
                mean_g = torch.mean(gate_task)
                gate_balance_loss = (mean_g - self.gate_target) ** 2
                task_aux_loss = task_aux_loss + self.gate_balance_lambda * gate_balance_loss

            if (
                center_mode in ['dynamic_rta_pred', 'dynamic_rta_pred_residual']
                and self.use_prediction_gate_supervision
                and gate_task is not None
                and pred_rel is not None
                and pred_task_route is not None
            ):
                gate_task_loss, err_rel, err_task, gate_oracle, gate_oracle_label, abs_err_diff, gate_margin_mask = self._build_prediction_gate_supervision(
                    y=y,
                    pred_rel=pred_rel,
                    pred_task=pred_task_route,
                    gate_task=gate_task,
                )
                task_aux_loss = task_aux_loss + self.gate_task_lambda * gate_task_loss
            else:
                abs_err_diff = None
                gate_margin_mask = None
        else:
            abs_err_diff = None
            gate_margin_mask = None

        self.last_router_info = {
            'router_weights': router_weights.detach(),
            'router_logits': router_logits.detach() if router_logits is not None else None,
            'availability': availability.detach(),
            'selected_anchor_idx': selected_idx.detach(),
            'task_router_weights': task_router_weights.detach() if task_router_weights is not None else None,
            'task_router_logits': task_router_logits.detach() if task_router_logits is not None else None,
            'task_selected_anchor_idx': selected_task_idx.detach() if selected_task_idx is not None else None,
            'pred_text_center': pred_t.detach() if pred_t is not None else None,
            'pred_audio_center': pred_a.detach() if pred_a is not None else None,
            'pred_vision_center': pred_v.detach() if pred_v is not None else None,
            'center_err': center_err.detach() if center_err is not None else None,
            'oracle_weight': oracle_weight.detach() if oracle_weight is not None else None,
            'oracle_label': oracle_label.detach() if oracle_label is not None else None,
            'gate_task': gate_task.detach() if gate_task is not None else None,
            'gate_rel': gate_rel.detach() if gate_rel is not None else None,
            'gate_logit': gate_logit.detach() if gate_logit is not None else None,
            'entropy_rel': entropy_rel.detach() if entropy_rel is not None else None,
            'entropy_task': entropy_task.detach() if entropy_task is not None else None,
            'router_disagreement': router_disagreement.detach() if router_disagreement is not None else None,
            'pred_std': pred_std.detach() if pred_std is not None else None,
            'pred_range': pred_range.detach() if pred_range is not None else None,
            'abs_pred_rel_task': abs_pred_rel_task.detach() if abs_pred_rel_task is not None else None,
            'h_rel': weighted_micro_rel.detach(),
            'h_task': weighted_micro_task.detach() if weighted_micro_task is not None else None,
            'h_ours': weighted_micro_final.detach(),
            'pred_rel': pred_rel.detach() if pred_rel is not None else None,
            'pred_task_route': pred_task_route.detach() if pred_task_route is not None else None,
            'gate_oracle': gate_oracle.detach() if gate_oracle is not None else None,
            'gate_oracle_label': gate_oracle_label.detach() if gate_oracle_label is not None else None,
            'err_rel': err_rel.detach() if err_rel is not None else None,
            'err_task': err_task.detach() if err_task is not None else None,
            'abs_err_diff': abs_err_diff.detach() if abs_err_diff is not None else None,
            'gate_margin_mask': gate_margin_mask.detach() if gate_margin_mask is not None else None,
            'gate_task_loss': gate_task_loss.detach() if gate_task_loss is not None else None,
            'gate_balance_loss': gate_balance_loss.detach() if gate_balance_loss is not None else None,
        }

        aux_total = task_aux_loss
        if self.router_balance_lambda > 0 and self.fusion_center_mode in ['dynamic_soft', 'dynamic_hard', 'dynamic_task_soft', 'dynamic_rta', 'dynamic_rta_pred', 'dynamic_rta_pred_residual']:
            balance_loss = self._router_balance_loss(router_weights)
            aux_total = aux_total + self.router_balance_lambda * balance_loss

        if self.fusion_center_mode in ['dynamic_task_soft', 'dynamic_rta', 'dynamic_rta_pred', 'dynamic_rta_pred_residual'] or aux_total.abs().item() > 0:
            return pred, aux_total
        return pred


MODULE_MAP = {
    'c_gate': GATE_F,
}


class Fusion(nn.Module):
    def __init__(self, args):
        super(Fusion, self).__init__()

        select_model = MODULE_MAP[args.fusionModule]
        self.Model = select_model(args)

    def forward(self, text_x, audio_x, vision_x, labels=None):
        return self.Model(text_x, audio_x, vision_x, labels=labels)
