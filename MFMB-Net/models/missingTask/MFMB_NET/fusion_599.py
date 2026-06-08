import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.missingTask.MFMB_NET.modules.transformer import TransformerEncoder
from models.missingTask.MFMB_NET.mide import MIDEController, compute_availability
from einops.layers.torch import Rearrange


class MLP_block(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class MLP_Communicator(nn.Module):
    def __init__(self, token, channel, hidden_size, depth=2):
        super(MLP_Communicator, self).__init__()
        self.depth = depth
        self.token_mixer = nn.Sequential(
            Rearrange('b n d -> b d n'),
            MLP_block(input_size=channel, hidden_size=hidden_size),
            Rearrange('b n d -> b d n')
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
        self.embed_dim = 90
        self.embed_dim_common = 32
        self.seq = 4

        self.layer_unimodal = 2
        self.layer_multimodal = 2
        self.Linear_common = nn.Linear(self.embed_dim, self.embed_dim_common)
        self.transformer = TransformerEncoder(embed_dim=self.embed_dim, num_heads=10, layers=4, attn_mask=False)

    def forward(self, audio, visual, text):
        for i in range(self.layer_unimodal):
            audio = self.transformer(audio)

        for i in range(self.layer_unimodal):
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

        x = x[:self.seq, :, :]
        return x


class GRUencoder(nn.Module):
    """Pad for utterances with variable lengths and maintain the order of them after GRU"""

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

        utterance_rep = self.dropout_in(utterance_rep)

        return utterance_rep


class GATE_F(nn.Module):
    def __init__(self, args):
        super(GATE_F, self).__init__()

        self.text_encoder = C_GATE(args.fusion_t_in, args.fusion_t_hid, args.fusion_gru_layers, args.fusion_drop)
        self.audio_encoder = C_GATE(args.fusion_a_in, args.fusion_a_hid, args.fusion_gru_layers, args.fusion_drop)
        self.vision_encoder = C_GATE(args.fusion_v_in, args.fusion_v_hid, args.fusion_gru_layers, args.fusion_drop)

        self.audio_visual_model = BottleAttentionNet()

        self.seq_v_mosi = nn.Linear(500, 50)
        self.seq_a_mosi = nn.Linear(375, 50)

        self.seq_v_mosei = nn.Linear(500, 50)
        self.seq_a_mosei = nn.Linear(500, 50)

        self.common_size = 64
        self.dim = self.common_size
        self.batchnorm = nn.BatchNorm1d(2, affine=False)
        self.MLP_Communicator1 = MLP_Communicator(self.dim, 2, hidden_size=64, depth=1)
        self.MLP_Communicator2 = MLP_Communicator(self.dim, 2, hidden_size=64, depth=1)
        self.args = args

        self.t_stack_linear = nn.Linear(36, self.common_size)
        self.v_stack_linear = nn.Linear(48, self.common_size)
        self.a_stack_linear = nn.Linear(20, self.common_size)

        self.classifier1 = nn.Sequential()
        self.classifier1.add_module('linear_trans_norm', nn.BatchNorm1d(self.common_size * 4 + 90))
        self.classifier1.add_module('linear_trans_hidden', nn.Linear(self.common_size * 4 + 90, args.cls_hidden_dim))
        self.classifier1.add_module('linear_trans_activation', nn.LeakyReLU())
        self.classifier1.add_module('linear_trans_drop', nn.Dropout(args.cls_dropout))
        self.classifier1.add_module('linear_trans_final', nn.Linear(args.cls_hidden_dim, 1))

        self.classifier2 = nn.Sequential()
        self.classifier2.add_module('linear_trans_norm', nn.BatchNorm1d(args.fusion_t_hid + args.fusion_a_hid + args.fusion_v_hid + self.common_size * 4 + 90))
        self.classifier2.add_module('linear_trans_hidden', nn.Linear(args.fusion_t_hid + args.fusion_a_hid + args.fusion_v_hid + self.common_size * 4 + 90, args.cls_hidden_dim))
        self.classifier2.add_module('linear_trans_activation', nn.ReLU())
        self.classifier2.add_module('linear_trans_drop', nn.Dropout(args.cls_dropout))
        self.classifier2.add_module('linear_trans_final', nn.Linear(args.cls_hidden_dim, 1))

        self.mide_enable = getattr(args, 'mide_enable', False)
        if self.mide_enable:
            self.mide = MIDEController(args)

    def _build_stack_features(self, text_common, audio_common, vision_common):
        bs = text_common.size(0)
        stack_rep_ta = torch.stack((text_common, audio_common), dim=0)
        stack_rep_tv = torch.stack((text_common, vision_common), dim=0)

        stack_rep_ta = self.batchnorm(stack_rep_ta.permute(1, 0, 2))
        stack_rep_tv = self.batchnorm(stack_rep_tv.permute(1, 0, 2))

        stack_rep_ta = stack_rep_ta.reshape(bs, -1)
        stack_rep_tv = stack_rep_tv.reshape(bs, -1)

        sqrt_d_ta = None
        sqrt_d_tv = None
        return stack_rep_ta, stack_rep_tv, sqrt_d_ta, sqrt_d_tv

    def _apply_pairwise_density(self, stack_rep_ta, stack_rep_tv, d_t, d_a, d_v):
        sqrt_d_ta = torch.sqrt((d_t * d_a).clamp(min=1e-6))
        sqrt_d_tv = torch.sqrt((d_t * d_v).clamp(min=1e-6))
        stack_rep_ta = stack_rep_ta * sqrt_d_ta
        stack_rep_tv = stack_rep_tv * sqrt_d_tv
        return stack_rep_ta, stack_rep_tv

    def _classify(
        self,
        text_rep,
        audio_rep,
        vision_rep,
        stack_rep_ta,
        stack_rep_tv,
        audio_visual_fusion,
    ):
        utterance_rep = torch.cat(
            (text_rep, audio_rep, vision_rep, stack_rep_tv, stack_rep_ta, audio_visual_fusion),
            dim=1,
        )
        return self.classifier2(utterance_rep)

    def _mask_modality(self, text_rep, audio_rep, vision_rep, text_common, audio_common, vision_common, disabled):
        if disabled is None:
            return text_rep, audio_rep, vision_rep, text_common, audio_common, vision_common
        z_t = torch.zeros_like(text_rep)
        z_a = torch.zeros_like(audio_rep)
        z_v = torch.zeros_like(vision_rep)
        z_tc = torch.zeros_like(text_common)
        z_ac = torch.zeros_like(audio_common)
        z_vc = torch.zeros_like(vision_common)
        if disabled == 't':
            return z_t, audio_rep, vision_rep, z_tc, audio_common, vision_common
        if disabled == 'a':
            return text_rep, z_a, vision_rep, text_common, z_ac, vision_common
        if disabled == 'v':
            return text_rep, audio_rep, z_v, text_common, audio_common, z_vc
        return text_rep, audio_rep, vision_rep, text_common, audio_common, vision_common

    def forward(
        self,
        text_x,
        audio_x,
        vision_x,
        mide_context=None,
    ):
        text_x, text_mask = text_x

        text_x_fusion = text_x.permute(1, 0, 2)

        audio_x, audio_mask = audio_x
        audio_x_fusion = self.seq_a_mosi(audio_x.permute(0, 2, 1))
        audio_x_fusion = audio_x_fusion.permute(2, 0, 1)

        vision_x, vision_mask = vision_x
        vision_x_fusion = self.seq_v_mosi(vision_x.permute(0, 2, 1))
        vision_x_fusion = vision_x_fusion.permute(2, 0, 1)

        audio_visual_fusion = self.audio_visual_model(audio_x_fusion, vision_x_fusion, text_x_fusion)
        audio_visual_fusion = audio_visual_fusion[-1]

        text_rep = self.text_encoder(text_x, text_mask)
        text_rep_common = self.t_stack_linear(text_rep)

        audio_rep = self.audio_encoder(audio_x, audio_mask)
        audio_rep_common = self.a_stack_linear(audio_rep)

        vision_rep = self.vision_encoder(vision_x, vision_mask)
        vision_rep_common = self.v_stack_linear(vision_rep)

        mide_aux = None
        if self.mide_enable and mide_context is not None:
            labels = mide_context.get('labels')
            epoch = mide_context.get('epoch', 1)
            training = mide_context.get('training', False)

            a_t, a_a, a_v = compute_availability(
                mide_context['text_missing_mask'],
                mide_context['audio_missing_mask'],
                mide_context['vision_missing_mask'],
                text_mask=mide_context.get('text_mask'),
                audio_mask=mide_context.get('audio_mask'),
                vision_mask=mide_context.get('vision_mask'),
            )
            a_vec = torch.cat([a_t, a_a, a_v], dim=1)

            y_t, y_a, y_v, u_logits, p_logits = self.mide.forward_heads(text_rep, audio_rep, vision_rep)
            d_task = self.mide.compute_density(a_vec, u_logits, p_logits)

            stack_all_ta, stack_all_tv = self._build_stack_features(
                text_rep_common, audio_rep_common, vision_rep_common
            )[:2]

            y_all_base = self._classify(
                text_rep, audio_rep, vision_rep, stack_all_ta, stack_all_tv, audio_visual_fusion
            )

            y_wo = {}
            for key in ('t', 'a', 'v'):
                tr, ar, vr, tc, ac, vc = self._mask_modality(
                    text_rep, audio_rep, vision_rep,
                    text_rep_common, audio_rep_common, vision_rep_common,
                    disabled=key,
                )
                sta, stv = self._build_stack_features(tc, ac, vc)[:2]
                y_wo[key] = self._classify(tr, ar, vr, sta, stv, audio_visual_fusion)

            warmup = getattr(self.args, 'mide_warmup_epochs', 1)
            apply_density = (not training) or (epoch > warmup)

            if apply_density:
                (
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
                ) = MIDEController.apply_density_weighting(
                    text_rep,
                    audio_rep,
                    vision_rep,
                    text_rep_common,
                    audio_rep_common,
                    vision_rep_common,
                    audio_visual_fusion,
                    d_task,
                )
                stack_rep_ta, stack_rep_tv = self._build_stack_features(
                    text_common_w, audio_common_w, vision_common_w
                )[:2]
                stack_rep_ta, stack_rep_tv = self._apply_pairwise_density(stack_rep_ta, stack_rep_tv, d_t, d_a, d_v)
                prediction = self._classify(
                    text_rep_w, audio_rep_w, vision_rep_w,
                    stack_rep_ta, stack_rep_tv, av_fusion_w,
                )
            else:
                stack_rep_ta, stack_rep_tv = self._build_stack_features(
                    text_rep_common, audio_rep_common, vision_rep_common
                )[:2]
                prediction = self._classify(
                    text_rep, audio_rep, vision_rep,
                    stack_rep_ta, stack_rep_tv, audio_visual_fusion,
                )

            mide_losses, mide_stats = self.mide.compute_losses(
                y_t,
                y_a,
                y_v,
                u_logits,
                p_logits,
                a_vec,
                d_task,
                labels if labels is not None else y_all_base.detach() * 0,
                y_all_base,
                y_wo['t'],
                y_wo['a'],
                y_wo['v'],
                epoch,
                enable_mide_losses=training and labels is not None,
            )

            mide_aux = {
                'losses': mide_losses,
                'stats': mide_stats,
                'y_all_loo': y_all_base,
            }
            return prediction, mide_aux

        stack_rep_ta, stack_rep_tv = self._build_stack_features(
            text_rep_common, audio_rep_common, vision_rep_common
        )[:2]
        prediction = self._classify(
            text_rep, audio_rep, vision_rep, stack_rep_ta, stack_rep_tv, audio_visual_fusion
        )
        return prediction, mide_aux


MODULE_MAP = {
    'c_gate': GATE_F,
}


class Fusion(nn.Module):
    def __init__(self, args):
        super(Fusion, self).__init__()

        select_model = MODULE_MAP[args.fusionModule]

        self.Model = select_model(args)

    def forward(self, text_x, audio_x, vision_x, mide_context=None):
        return self.Model(text_x, audio_x, vision_x, mide_context=mide_context)
