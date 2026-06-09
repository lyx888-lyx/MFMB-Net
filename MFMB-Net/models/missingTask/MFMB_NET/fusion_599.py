import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.missingTask.MFMB_NET.modules.transformer import TransformerEncoder
from models.missingTask.MFMB_NET.mide import (
    compute_availability,
    create_mide_controller,
    compute_d_eff,
    gate_enabled,
    OldMIDEController,
)
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
            MLP_block(input_size=token, hidden_size=hidden_size),
        )

    def forward(self, x):
        for _ in range(self.depth):
            x = x + self.token_mixer(x)
            x = x + self.channel_mixer(x)
        return x


class BottleAttentionNet(nn.Module):
    def __init__(self):
        super(BottleAttentionNet, self).__init__()
        self.embed_dim = 90
        self.seq = 4
        self.layer_unimodal = 2
        self.layer_multimodal = 2
        self.transformer = TransformerEncoder(embed_dim=self.embed_dim, num_heads=10, layers=4, attn_mask=False)

    def forward(self, audio, visual, text):
        for _ in range(self.layer_unimodal):
            audio = self.transformer(audio)
        for _ in range(self.layer_unimodal):
            visual = self.transformer(visual)
        device = audio.device
        fsn = torch.zeros(self.seq, audio.size(1), self.embed_dim, device=device)
        x = torch.cat([audio, fsn], dim=0)
        for i in range(self.layer_multimodal):
            if i == 0:
                x = self.transformer(x)
                x = torch.cat([x[audio.size(0):, :, :], visual], dim=0)
                x = self.transformer(x)
                x = torch.cat([x[:self.seq, :, :], text], dim=0)
                x = self.transformer(x)
            else:
                x = self.transformer(x)
        return x[:self.seq, :, :]


class GRUencoder(nn.Module):
    def __init__(self, embedding_dim, utterance_dim, num_layers):
        super(GRUencoder, self).__init__()
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=utterance_dim,
                          bidirectional=True, num_layers=num_layers)

    def forward(self, utterance, utterance_lens):
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
        gate = F.sigmoid(self.cnn(transformed_.transpose(1, 2)).transpose(1, 2))
        gate_x = torch.tanh(transformed_) * gate
        utterance_rep = torch.tanh(self.fc(torch.cat([utterance, gate_x], dim=-1)))
        utterance_rep = torch.max(utterance_rep, dim=1)[0]
        return self.dropout_in(utterance_rep)


class GATE_F(nn.Module):
    def __init__(self, args):
        super(GATE_F, self).__init__()
        self.args = args
        self.text_encoder = C_GATE(args.fusion_t_in, args.fusion_t_hid, args.fusion_gru_layers, args.fusion_drop)
        self.audio_encoder = C_GATE(args.fusion_a_in, args.fusion_a_hid, args.fusion_gru_layers, args.fusion_drop)
        self.vision_encoder = C_GATE(args.fusion_v_in, args.fusion_v_hid, args.fusion_gru_layers, args.fusion_drop)
        self.audio_visual_model = BottleAttentionNet()
        self.seq_v_mosi = nn.Linear(500, 50)
        self.seq_a_mosi = nn.Linear(375, 50)
        self.common_size = 64
        self.batchnorm = nn.BatchNorm1d(2, affine=False)
        self.t_stack_linear = nn.Linear(36, self.common_size)
        self.v_stack_linear = nn.Linear(48, self.common_size)
        self.a_stack_linear = nn.Linear(20, self.common_size)
        self.classifier2 = nn.Sequential()
        hid_in = args.fusion_t_hid + args.fusion_a_hid + args.fusion_v_hid + self.common_size * 4 + 90
        self.classifier2.add_module('linear_trans_norm', nn.BatchNorm1d(hid_in))
        self.classifier2.add_module('linear_trans_hidden', nn.Linear(hid_in, args.cls_hidden_dim))
        self.classifier2.add_module('linear_trans_activation', nn.ReLU())
        self.classifier2.add_module('linear_trans_drop', nn.Dropout(args.cls_dropout))
        self.classifier2.add_module('linear_trans_final', nn.Linear(args.cls_hidden_dim, 1))
        self.mide_enable = getattr(args, 'mide_enable', False)
        self.mide_variant = getattr(args, 'mide_variant', 'split_aup')
        if self.mide_enable:
            self.mide = create_mide_controller(args)

    def _mask_raw_inputs(self, text_x, audio_x, vision_x, disabled=None):
        if disabled is None:
            return text_x, audio_x, vision_x
        zt, za, zv = torch.zeros_like(text_x), torch.zeros_like(audio_x), torch.zeros_like(vision_x)
        if disabled == 't':
            return zt, audio_x, vision_x
        if disabled == 'a':
            return text_x, za, vision_x
        if disabled == 'v':
            return text_x, audio_x, zv
        return text_x, audio_x, vision_x

    def _encode_modalities(self, text_x, text_mask, audio_x, audio_mask, vision_x, vision_mask, disabled=None):
        text_x, audio_x, vision_x = self._mask_raw_inputs(text_x, audio_x, vision_x, disabled)
        text_x_fusion = text_x.permute(1, 0, 2)
        audio_x_fusion = self.seq_a_mosi(audio_x.permute(0, 2, 1)).permute(2, 0, 1)
        vision_x_fusion = self.seq_v_mosi(vision_x.permute(0, 2, 1)).permute(2, 0, 1)
        audio_visual_fusion = self.audio_visual_model(audio_x_fusion, vision_x_fusion, text_x_fusion)[-1]
        text_rep = self.text_encoder(text_x, text_mask)
        audio_rep = self.audio_encoder(audio_x, audio_mask)
        vision_rep = self.vision_encoder(vision_x, vision_mask)
        text_common = self.t_stack_linear(text_rep)
        audio_common = self.a_stack_linear(audio_rep)
        vision_common = self.v_stack_linear(vision_rep)
        return {
            'text_rep': text_rep,
            'audio_rep': audio_rep,
            'vision_rep': vision_rep,
            'text_common': text_common,
            'audio_common': audio_common,
            'vision_common': vision_common,
            'audio_visual_fusion': audio_visual_fusion,
        }

    def _build_pairwise_stacks(self, text_common, audio_common, vision_common, d_eff=None):
        bs = text_common.size(0)
        stack_ta = torch.stack((text_common, audio_common), dim=0)
        stack_tv = torch.stack((text_common, vision_common), dim=0)
        stack_ta = self.batchnorm(stack_ta.permute(1, 0, 2)).reshape(bs, -1)
        stack_tv = self.batchnorm(stack_tv.permute(1, 0, 2)).reshape(bs, -1)
        if d_eff is not None:
            d_t, d_a, d_v = d_eff[:, 0:1], d_eff[:, 1:2], d_eff[:, 2:3]
            stack_ta = stack_ta * torch.sqrt((d_t * d_a).clamp(min=1e-6))
            stack_tv = stack_tv * torch.sqrt((d_t * d_v).clamp(min=1e-6))
        return stack_ta, stack_tv

    def _apply_split_gate(self, reps, d_eff):
        d_t, d_a, d_v = d_eff[:, 0:1], d_eff[:, 1:2], d_eff[:, 2:3]
        text_rep = reps['text_rep'] * d_t
        audio_rep = reps['audio_rep'] * d_a
        vision_rep = reps['vision_rep'] * d_v
        av = reps['audio_visual_fusion'] * torch.sqrt((d_a * d_v).clamp(min=1e-6))
        stack_ta, stack_tv = self._build_pairwise_stacks(
            reps['text_common'], reps['audio_common'], reps['vision_common'], d_eff=d_eff,
        )
        return text_rep, audio_rep, vision_rep, stack_ta, stack_tv, av

    def _classify(self, text_rep, audio_rep, vision_rep, stack_ta, stack_tv, av_fusion):
        utterance_rep = torch.cat((text_rep, audio_rep, vision_rep, stack_tv, stack_ta, av_fusion), dim=1)
        return self.classifier2(utterance_rep)

    def _forward_prediction_path(self, text_x, text_mask, audio_x, audio_mask, vision_x, vision_mask,
                               disabled=None, d_eff=None):
        reps = self._encode_modalities(text_x, text_mask, audio_x, audio_mask, vision_x, vision_mask, disabled)
        if d_eff is not None:
            text_rep, audio_rep, vision_rep, stack_ta, stack_tv, av = self._apply_split_gate(reps, d_eff)
        else:
            stack_ta, stack_tv = self._build_pairwise_stacks(
                reps['text_common'], reps['audio_common'], reps['vision_common'], d_eff=None,
            )
            text_rep, audio_rep, vision_rep, av = (
                reps['text_rep'], reps['audio_rep'], reps['vision_rep'], reps['audio_visual_fusion'],
            )
        return self._classify(text_rep, audio_rep, vision_rep, stack_ta, stack_tv, av), reps

    def _forward_old_path(self, text_x, text_mask, audio_x, audio_mask, vision_x, vision_mask,
                          disabled=None, d_task=None, apply_density=False):
        reps = self._encode_modalities(text_x, text_mask, audio_x, audio_mask, vision_x, vision_mask, disabled)
        tr, ar, vr = reps['text_rep'], reps['audio_rep'], reps['vision_rep']
        tc, ac, vc = reps['text_common'], reps['audio_common'], reps['vision_common']
        av = reps['audio_visual_fusion']
        if apply_density and d_task is not None:
            tr, ar, vr, tc, ac, vc, av, d_t, d_a, d_v = OldMIDEController.apply_density_weighting(
                tr, ar, vr, tc, ac, vc, av, d_task,
            )
            stack_ta, stack_tv = self._build_pairwise_stacks(tc, ac, vc, d_eff=None)
            stack_ta = stack_ta * torch.sqrt((d_t * d_a).clamp(min=1e-6))
            stack_tv = stack_tv * torch.sqrt((d_t * d_v).clamp(min=1e-6))
        else:
            stack_ta, stack_tv = self._build_pairwise_stacks(tc, ac, vc, d_eff=None)
        return self._classify(tr, ar, vr, stack_ta, stack_tv, av), reps

    def forward(self, text_x, audio_x, vision_x, mide_context=None):
        text_x, text_mask = text_x
        audio_x, audio_mask = audio_x
        vision_x, vision_mask = vision_x

        if not self.mide_enable or mide_context is None:
            pred, _ = self._forward_prediction_path(
                text_x, text_mask, audio_x, audio_mask, vision_x, vision_mask,
            )
            return pred, None

        labels = mide_context.get('labels')
        epoch = mide_context.get('epoch', 1)
        training = mide_context.get('training', False)
        a_t, a_a, a_v = compute_availability(
            mide_context['text_missing_mask'], mide_context['audio_missing_mask'], mide_context['vision_missing_mask'],
            text_mask=mide_context.get('text_mask'), audio_mask=mide_context.get('audio_mask'),
            vision_mask=mide_context.get('vision_mask'),
        )
        a_vec = torch.cat([a_t, a_a, a_v], dim=1)

        if self.mide_variant == 'old':
            _, base_reps = self._forward_prediction_path(
                text_x, text_mask, audio_x, audio_mask, vision_x, vision_mask,
            )
            y_t, y_a, y_v, u_logits, p_logits = self.mide.forward_heads(
                base_reps['text_rep'], base_reps['audio_rep'], base_reps['vision_rep'],
            )
            d_task = self.mide.compute_density(a_vec, u_logits, p_logits)
            warmup = getattr(self.args, 'mide_warmup_epochs', 1)
            apply_density = epoch > warmup
            y_all_ref, _ = self._forward_old_path(
                text_x, text_mask, audio_x, audio_mask, vision_x, vision_mask,
                disabled=None, d_task=d_task, apply_density=apply_density,
            )
            y_wo = {}
            for k in ('t', 'a', 'v'):
                y_wo[k], _ = self._forward_old_path(
                    text_x, text_mask, audio_x, audio_mask, vision_x, vision_mask,
                    disabled=k, d_task=d_task, apply_density=False,
                )
            prediction = y_all_ref
            mide_losses, mide_stats = self.mide.compute_losses(
                y_t, y_a, y_v, u_logits, p_logits, a_vec, d_task,
                labels if labels is not None else y_all_ref.detach() * 0,
                y_all_ref, y_wo['t'], y_wo['a'], y_wo['v'],
                epoch, enable_mide_losses=training and labels is not None,
            )
            return prediction, {'losses': mide_losses, 'stats': mide_stats}

        # split_aup / split_aup_plus
        _, base_reps = self._forward_prediction_path(
            text_x, text_mask, audio_x, audio_mask, vision_x, vision_mask,
        )
        y_t, y_a, y_v, u_logits, p_logits = self.mide.forward_heads(
            base_reps['text_rep'], base_reps['audio_rep'], base_reps['vision_rep'],
        )
        d_eff, d_raw = compute_d_eff(a_vec, u_logits, p_logits, epoch, self.args)
        use_gate = gate_enabled(epoch, self.args)
        d_for_path = d_eff if use_gate else None

        y_all_ref, _ = self._forward_prediction_path(
            text_x, text_mask, audio_x, audio_mask, vision_x, vision_mask,
            disabled=None, d_eff=d_for_path,
        )
        y_wo = {}
        for k in ('t', 'a', 'v'):
            y_wo[k], _ = self._forward_prediction_path(
                text_x, text_mask, audio_x, audio_mask, vision_x, vision_mask,
                disabled=k, d_eff=d_for_path,
            )
        prediction = y_all_ref
        mide_losses, mide_stats = self.mide.compute_losses(
            y_t, y_a, y_v, u_logits, p_logits, a_vec, d_eff,
            labels if labels is not None else y_all_ref.detach() * 0,
            y_all_ref, y_wo['t'], y_wo['a'], y_wo['v'],
            epoch, enable_mide_losses=training and labels is not None,
        )
        return prediction, {'losses': mide_losses, 'stats': mide_stats}


MODULE_MAP = {'c_gate': GATE_F}


class Fusion(nn.Module):
    def __init__(self, args):
        super(Fusion, self).__init__()
        self.Model = MODULE_MAP[args.fusionModule](args)

    def forward(self, text_x, audio_x, vision_x, mide_context=None):
        return self.Model(text_x, audio_x, vision_x, mide_context=mide_context)
