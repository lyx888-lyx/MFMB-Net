"""
GATE_F 融合：固定锚点 + dynamic 多 hub 加权。

--------------------------------------------------------------------
A) fixed text / audio / vision
--------------------------------------------------------------------
  与历史单锚点一致：仅一条 bottle 序（HUB_TO_BOTTLE），无 router、无 alpha。

--------------------------------------------------------------------
B) dynamic + fusion_dynamic_mode=hard + use_reliability_router=True
--------------------------------------------------------------------
  ReliabilityRouterMLP 输出 logits → **argmax** → **one-hot alpha**（硬动态锚点）。
  alpha 索引顺序：0=text-hub, 1=audio-hub, 2=vision-hub。

--------------------------------------------------------------------
C) dynamic + fusion_dynamic_mode=soft + use_reliability_router=True
--------------------------------------------------------------------
  logits → softmax(logits/temperature) → **连续 alpha**（可靠性感知软路由）。

--------------------------------------------------------------------
D) dynamic + fusion_dynamic_mode=soft + use_reliability_router=False
--------------------------------------------------------------------
  **不调用 router**，固定 **alpha=(1/3,1/3,1/3)** 对三路 hub 做加权。
  这是 **uniform soft fusion baseline**（均匀软融合基线），
  **不是**「退化为旧版单路径融合」；旧版单路径请用 fixed *。

--------------------------------------------------------------------
Checkpoint：dynamic 且启用 router 时参数为 ReliabilityRouterMLP，与旧版 Linear(6,3) `anchor_scorer`
**不兼容**，需重新训练；fixed 路径结构未改，旧权重仍可按 strict=False 策略加载（见 run.py）。

三路 hub 顺序 [text, audio, vision] 与 HUB_TO_BOTTLE 中 avt / tav / tva 一致。
Prompt 定义见 corruption_prompt.py 顶部注释。
"""
import logging
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Optional, Tuple, Dict, Any

from models.missingTask.MFMB_NET.modules.transformer import TransformerEncoder
from models.missingTask.MFMB_NET.corruption_prompt import CorruptionPromptEncoder
from models.missingTask.MFMB_NET.reliability_router import ReliabilityRouterMLP
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
    def __init__(self, bottle_order: str = 'avt'):
        super(BottleAttentionNet, self).__init__()
        self.embed_dim = 90
        self.seq = 4
        assert len(bottle_order) == 3 and set(bottle_order) == {'a', 'v', 't'}
        self.bottle_order = bottle_order
        self.layer_unimodal = 2
        self.layer_multimodal = 2
        self.transformer = TransformerEncoder(embed_dim=self.embed_dim, num_heads=10, layers=4, attn_mask=False)

    def forward(self, audio, visual, text, bottle_order: Optional[str] = None):
        order = bottle_order if bottle_order is not None else self.bottle_order
        for _ in range(self.layer_unimodal):
            audio = self.transformer(audio)
        for _ in range(self.layer_unimodal):
            visual = self.transformer(visual)
        modalities = {'a': audio, 'v': visual, 't': text}
        m1 = modalities[order[0]]
        m2 = modalities[order[1]]
        m3 = modalities[order[2]]
        device, dtype = m1.device, m1.dtype
        fsn = torch.zeros(self.seq, m1.size(1), self.embed_dim, device=device, dtype=dtype)
        x = torch.cat([m1, fsn], dim=0)
        m1_len = m1.size(0)
        for i in range(self.layer_multimodal):
            if i == 0:
                x = self.transformer(x)
                x = torch.cat([x[m1_len:, :, :], m2], dim=0)
                x = self.transformer(x)
                x = torch.cat([x[:self.seq, :, :], m3], dim=0)
                x = self.transformer(x)
            else:
                x = self.transformer(x)
        return x[:self.seq, :, :]


def _masked_mean_energy(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.float().unsqueeze(-1)
    num = (x * x * m).sum(dim=(1, 2))
    den = m.sum(dim=(1, 2)).clamp(min=1e-8)
    return num / den


def compute_modality_quality_scores(
    text_x: torch.Tensor,
    audio_x: torch.Tensor,
    vision_x: torch.Tensor,
    text_mask: torch.Tensor,
    audio_mask: torch.Tensor,
    vision_mask: torch.Tensor,
    missing_mask_t: torch.Tensor,
    missing_mask_a: torch.Tensor,
    missing_mask_v: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    m_t = text_mask.float() * missing_mask_t.float()
    int_t = m_t.sum(dim=1) / text_mask.float().sum(dim=1).clamp(min=eps)
    e_t = _masked_mean_energy(text_x, m_t)
    m_a = audio_mask.float() * missing_mask_a.float()
    int_a = m_a.sum(dim=1) / audio_mask.float().sum(dim=1).clamp(min=eps)
    e_a = _masked_mean_energy(audio_x, m_a)
    m_v = vision_mask.float() * missing_mask_v.float()
    int_v = m_v.sum(dim=1) / vision_mask.float().sum(dim=1).clamp(min=eps)
    e_v = _masked_mean_energy(vision_x, m_v)
    integrity = torch.stack([int_t, int_a, int_v], dim=1)
    log_e = torch.log(torch.stack([e_t, e_a, e_v], dim=1).clamp(min=eps))
    return integrity, log_e


HUB_TO_BOTTLE = {'text': 'avt', 'audio': 'tav', 'vision': 'tva'}
# alpha / logits / selected_anchor 维序：与下标一致
HUB_ANCHOR_NAMES = ('text', 'audio', 'vision')

_FUSION_LOG = logging.getLogger('MSA')


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
        gate = torch.sigmoid(self.cnn(transformed_.transpose(1, 2)).transpose(1, 2))
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

        # Storage(dict) 对缺失键的 __getattr__ 会返回 False，不能用 getattr(..., default)；用 .get 取默认
        center = args.get('fusion_center_modality', 'text')
        self.fusion_center_modality = center
        self.fusion_dynamic_mode = args.get('fusion_dynamic_mode', 'soft')
        self.use_reliability_router = args.get('use_reliability_router', True)
        self.fusion_router_temperature = args.get('fusion_router_temperature', 1.0)
        self.use_corruption_prompt = args.get('use_corruption_prompt', False)
        self.corruption_prompt_dim = args.get('corruption_prompt_dim', 32)
        self.use_prompt_in_router = args.get('use_prompt_in_router', False)
        self.use_prompt_in_fusion = args.get('use_prompt_in_fusion', False)

        router_in = 6
        if self.use_prompt_in_router and self.use_corruption_prompt:
            router_in = 6 + self.corruption_prompt_dim

        self.router = None
        if center == 'dynamic' and self.use_reliability_router:
            self.router = ReliabilityRouterMLP(
                router_in,
                hidden=args.get('fusion_router_hidden', 64),
                dropout=args.get('fusion_router_drop', 0.1),
            )

        self.prompt_encoder = None
        if self.use_corruption_prompt:
            self.prompt_encoder = CorruptionPromptEncoder(
                self.corruption_prompt_dim,
                dropout=args.get('corruption_prompt_drop', 0.0),
            )

        if center == 'dynamic':
            init_order = 'avt'
        else:
            init_order = HUB_TO_BOTTLE[center]
        self.audio_visual_model = BottleAttentionNet(bottle_order=init_order)

        self.seq_v_mosi = nn.Linear(500, 50)
        self.seq_a_mosi = nn.Linear(375, 50)
        self.seq_v_mosei = nn.Linear(500, 50)
        self.seq_a_mosei = nn.Linear(500, 50)

        self.common_size = 64
        self.batchnorm = nn.BatchNorm1d(2, affine=False)
        self.args = args
        self.batch_size = self.args.batch_size
        self.t_stack_linear = nn.Linear(36, self.common_size)
        self.v_stack_linear = nn.Linear(48, self.common_size)
        self.a_stack_linear = nn.Linear(20, self.common_size)

        clf_in = args.fusion_t_hid + args.fusion_a_hid + args.fusion_v_hid + self.common_size * 4 + 90
        self.utter_dim = clf_in
        self.prompt_to_utter = None
        if self.use_prompt_in_fusion and self.use_corruption_prompt:
            self.prompt_to_utter = nn.Linear(self.corruption_prompt_dim, clf_in)
            nn.init.zeros_(self.prompt_to_utter.weight)
            nn.init.zeros_(self.prompt_to_utter.bias)

        self.classifier2 = nn.Sequential()
        self.classifier2.add_module('linear_trans_norm', nn.BatchNorm1d(clf_in))
        self.classifier2.add_module('linear_trans_hidden', nn.Linear(clf_in, args.cls_hidden_dim))
        self.classifier2.add_module('linear_trans_activation', nn.ReLU())
        self.classifier2.add_module('linear_trans_drop', nn.Dropout(args.cls_dropout))
        self.classifier2.add_module('linear_trans_final', nn.Linear(args.cls_hidden_dim, 1))

        self._log_fusion_init()

    def _log_fusion_init(self) -> None:
        c = self.fusion_center_modality
        if c == 'dynamic':
            if self.use_reliability_router:
                _FUSION_LOG.info(
                    "MFMB fusion: center=dynamic, router=ReliabilityRouterMLP, dynamic_mode=%s, "
                    "temp=%.4f, prompt=%s, prompt_in_router=%s, prompt_in_fusion=%s",
                    self.fusion_dynamic_mode,
                    float(self.fusion_router_temperature),
                    self.use_corruption_prompt,
                    self.use_prompt_in_router,
                    self.use_prompt_in_fusion,
                )
            else:
                _FUSION_LOG.info(
                    "MFMB fusion: center=dynamic, router=OFF → uniform soft fusion baseline "
                    "(alpha=1/3 per hub; NOT equivalent to legacy single fixed path). "
                    "dynamic_mode=%s (soft path only; hard ignored without router). "
                    "prompt=%s",
                    self.fusion_dynamic_mode,
                    self.use_corruption_prompt,
                )
        else:
            _FUSION_LOG.info("MFMB fusion: center=fixed (%s), no dynamic router.", c)

    def _stack_for_hub(
        self,
        hub: str,
        text_rep_common: torch.Tensor,
        audio_rep_common: torch.Tensor,
        vision_rep_common: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hub == 'text':
            stack_rep_ta = torch.stack((text_rep_common, audio_rep_common), dim=0)
            stack_rep_tv = torch.stack((text_rep_common, vision_rep_common), dim=0)
        elif hub == 'audio':
            stack_rep_ta = torch.stack((audio_rep_common, text_rep_common), dim=0)
            stack_rep_tv = torch.stack((audio_rep_common, vision_rep_common), dim=0)
        elif hub == 'vision':
            stack_rep_ta = torch.stack((vision_rep_common, text_rep_common), dim=0)
            stack_rep_tv = torch.stack((vision_rep_common, audio_rep_common), dim=0)
        else:
            raise ValueError(hub)
        stack_rep_ta = self.batchnorm(stack_rep_ta.permute(1, 0, 2))
        stack_rep_tv = self.batchnorm(stack_rep_tv.permute(1, 0, 2))
        return stack_rep_ta.reshape(batch_size, -1), stack_rep_tv.reshape(batch_size, -1)

    def forward(
        self,
        text_x,
        audio_x,
        vision_x,
        return_aux: bool = False,
        distill_full_modality_prompt: bool = False,
    ):
        """
        distill_full_modality_prompt=True（蒸馏教师分支专用）：
        corruption prompt 使用「全模态完好」语义——missing 掩码在有效位置上全 1、noise_type=none，
        不沿用学生侧的缺失/噪声推断；不改变 router / fusion 主数学，仅改 prompt 编码输入。
        """
        if len(text_x) == 3:
            text_x, text_mask, missing_mask_t = text_x
        else:
            text_x, text_mask = text_x
            missing_mask_t = torch.ones_like(text_mask)
        if len(audio_x) == 3:
            audio_x, audio_mask, missing_mask_a = audio_x
        else:
            audio_x, audio_mask = audio_x
            missing_mask_a = torch.ones_like(audio_mask)
        if len(vision_x) == 3:
            vision_x, vision_mask, missing_mask_v = vision_x
        else:
            vision_x, vision_mask = vision_x
            missing_mask_v = torch.ones_like(vision_mask)

        B = text_x.size(0)
        device = text_x.device
        dtype = text_x.dtype

        text_x_fusion = text_x.permute(1, 0, 2)
        audio_x_fusion = self.seq_a_mosi(audio_x.permute(0, 2, 1)).permute(2, 0, 1)
        vision_x_fusion = self.seq_v_mosi(vision_x.permute(0, 2, 1)).permute(2, 0, 1)

        integrity, log_e = compute_modality_quality_scores(
            text_x, audio_x, vision_x,
            text_mask, audio_mask, vision_mask,
            missing_mask_t, missing_mask_a, missing_mask_v,
        )

        prompt_emb = None
        prompt_debug = None
        if self.prompt_encoder is not None:
            if distill_full_modality_prompt:
                noise_ids = torch.zeros((B,), device=device, dtype=torch.long)
                mm_t = text_mask.float().clamp(0, 1)
                mm_a = audio_mask.float().clamp(0, 1)
                mm_v = vision_mask.float().clamp(0, 1)
                prompt_emb, prompt_debug = self.prompt_encoder.forward_from_tensors(
                    mm_t, mm_a, mm_v, text_mask, audio_mask, vision_mask, noise_ids,
                )
            else:
                nid = CorruptionPromptEncoder.noise_id_from_args(self.args)
                noise_ids = torch.full((B,), int(nid), device=device, dtype=torch.long)
                prompt_emb, prompt_debug = self.prompt_encoder.forward_from_tensors(
                    missing_mask_t, missing_mask_a, missing_mask_v,
                    text_mask, audio_mask, vision_mask, noise_ids,
                )

        hub = self.fusion_center_modality
        aux: Dict[str, Any] = {}
        logits = None
        router_mode = f'fixed:{hub}'
        selected_anchor: Optional[torch.Tensor] = None

        if hub == 'dynamic':
            base_feats = torch.cat([integrity, log_e], dim=1)
            router_in = base_feats
            if self.use_prompt_in_router and prompt_emb is not None:
                router_in = torch.cat([base_feats, prompt_emb], dim=1)
            tau = max(float(self.fusion_router_temperature), 1e-3)
            if self.router is not None:
                logits = self.router(router_in)
                if str(self.fusion_dynamic_mode).lower() == 'hard':
                    router_mode = 'dynamic_hard_mlp'
                    idx = logits.argmax(dim=1)
                    selected_anchor = idx.detach()
                    alpha = torch.zeros(B, 3, device=device, dtype=dtype)
                    alpha.scatter_(1, idx.unsqueeze(1), 1.0)
                else:
                    router_mode = 'dynamic_soft_mlp'
                    alpha = F.softmax(logits / tau, dim=1)
            else:
                router_mode = 'dynamic_uniform_soft_baseline'
                alpha = torch.full((B, 3), 1.0 / 3.0, device=device, dtype=dtype)

            av_orders = [HUB_TO_BOTTLE['text'], HUB_TO_BOTTLE['audio'], HUB_TO_BOTTLE['vision']]
            av_list = []
            for o in av_orders:
                out = self.audio_visual_model(audio_x_fusion, vision_x_fusion, text_x_fusion, bottle_order=o)
                av_list.append(out[-1])
            av_stack = torch.stack(av_list, dim=1)
            audio_visual_fusion = (alpha.unsqueeze(-1) * av_stack).sum(dim=1)
        else:
            bo = HUB_TO_BOTTLE[hub]
            audio_visual_fusion = self.audio_visual_model(
                audio_x_fusion, vision_x_fusion, text_x_fusion, bottle_order=bo
            )[-1]

        text_rep = self.text_encoder(text_x, text_mask)
        text_rep_common = self.t_stack_linear(text_rep)
        audio_rep = self.audio_encoder(audio_x, audio_mask)
        audio_rep_common = self.a_stack_linear(audio_rep)
        vision_rep = self.vision_encoder(vision_x, vision_mask)
        vision_rep_common = self.v_stack_linear(vision_rep)

        if hub == 'dynamic':
            sta_t, stv_t = self._stack_for_hub('text', text_rep_common, audio_rep_common, vision_rep_common, B)
            sta_a, stv_a = self._stack_for_hub('audio', text_rep_common, audio_rep_common, vision_rep_common, B)
            sta_v, stv_v = self._stack_for_hub('vision', text_rep_common, audio_rep_common, vision_rep_common, B)
            sta_all = torch.stack([sta_t, sta_a, sta_v], dim=1)
            stv_all = torch.stack([stv_t, stv_a, stv_v], dim=1)
            alpha_stack = alpha.unsqueeze(-1)
            stack_rep_ta = (alpha_stack * sta_all).sum(dim=1)
            stack_rep_tv = (alpha_stack * stv_all).sum(dim=1)
            aux['alpha'] = alpha
            aux['router_logits'] = logits
            aux['router_mode'] = router_mode
            aux['selected_anchor'] = selected_anchor
            if selected_anchor is not None:
                aux['selected_anchor_name'] = [HUB_ANCHOR_NAMES[int(i)] for i in selected_anchor.cpu().tolist()]
            else:
                aux['selected_anchor_name'] = None
            aux['hub_order_names'] = list(HUB_ANCHOR_NAMES)
        else:
            stack_rep_ta, stack_rep_tv = self._stack_for_hub(
                hub, text_rep_common, audio_rep_common, vision_rep_common, B
            )

        utterance_rep = torch.cat((text_rep, audio_rep, vision_rep, stack_rep_tv, stack_rep_ta, audio_visual_fusion), dim=1)
        if self.prompt_to_utter is not None and prompt_emb is not None:
            utterance_rep = utterance_rep + self.prompt_to_utter(prompt_emb)

        out = self.classifier2(utterance_rep)
        if return_aux:
            aux['fused_rep'] = utterance_rep
            aux['distill_full_modality_prompt'] = bool(distill_full_modality_prompt)
            aux['integrity'] = integrity
            aux['log_energy'] = log_e
            aux['use_prompt_in_router'] = bool(self.use_prompt_in_router and self.prompt_encoder is not None)
            aux['use_prompt_in_fusion'] = bool(self.prompt_to_utter is not None)
            aux['prompt_debug'] = prompt_debug
            aux['router_mode'] = aux.get('router_mode', router_mode)
            if hub != 'dynamic':
                aux['router_logits'] = None
                aux['alpha'] = None
                aux['selected_anchor'] = None
                aux['selected_anchor_name'] = None
                aux['hub_order_names'] = list(HUB_ANCHOR_NAMES)
            return out, aux
        return out


MODULE_MAP = {
    'c_gate': GATE_F,
}


class Fusion(nn.Module):
    def __init__(self, args):
        super(Fusion, self).__init__()
        select_model = MODULE_MAP[args.fusionModule]
        self.Model = select_model(args)

    def forward(
        self,
        text_x,
        audio_x,
        vision_x,
        return_aux: bool = False,
        distill_full_modality_prompt: bool = False,
    ):
        """
        return_aux=False：默认训练路径，仅返回 pred。
        return_aux=True：返回 (pred, aux)，aux 含 fused_rep（分类头前向量）、router 调试字段、prompt_debug 等，
        供实验日志与后续蒸馏（阶段 C）复用；不改变默认调用行为。
        distill_full_modality_prompt：见 GATE_F.forward。
        """
        return self.Model(
            text_x,
            audio_x,
            vision_x,
            return_aux=return_aux,
            distill_full_modality_prompt=distill_full_modality_prompt,
        )
