import logging
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.missingTask.MFMB_NET.modules.transformer import TransformerEncoder
from einops.layers.torch import Rearrange

# Modality index order for micro-fusion dynamic center (tie-break: text > audio > vision == argmax on this axis).
_MOD_TEXT, _MOD_AUDIO, _MOD_VISION = 0, 1, 2


def _compute_modality_valid_ratios_from_missing(
    text_missing_mask,
    audio_missing_mask,
    vision_missing_mask,
    text_content_mask,
    audio_content_mask,
    vision_content_mask,
):
    """Per-sample valid ratios (text, audio, vision) from missing masks only.

    Definitions (see data/load_data.py ``generate_m``):
    - *content_mask*: 1 on non-padding timesteps, 0 on padding. Text uses the BERT
      attention mask row (same as fusion ``text_mask``). Audio/vision masks are
      1 for frames ``0 .. length-1`` and 0 for padded frames.
    - *missing_mask*: ``(rand > missing_rate) * content_mask``, then text forces
      CLS and last real token to 1. So on padding both masks are 0; on content
      positions, 1 means observed after missing simulation, 0 means intentionally
      dropped.

    For each modality and sample:
        denom = sum(content_mask)  (effective sequence length, excludes padding)
        valid_ratio = sum(missing_mask) / max(denom, 1)
        missing_ratio = 1 - valid_ratio  (fraction of content timesteps dropped)

    Returns:
        valid_ratios: FloatTensor [B, 3] columns (text, audio, vision).
    """
    def _ratio(mm, cm):
        denom = cm.sum(dim=1).clamp(min=1).to(dtype=torch.float32)
        num = mm.sum(dim=1).to(dtype=torch.float32)
        return num / denom

    r_t = _ratio(text_missing_mask, text_content_mask)
    r_a = _ratio(audio_missing_mask, audio_content_mask)
    r_v = _ratio(vision_missing_mask, vision_content_mask)
    return torch.stack((r_t, r_a, r_v), dim=1)


def _select_dynamic_center_from_missing(valid_ratios, min_valid_ratio=1e-3):
    """Rule router: among modalities with valid_ratio >= min_valid_ratio, pick
    the highest valid_ratio; ties break as text > audio > vision by argmax
    order (text column index 0).

    Args:
        valid_ratios: [B, 3] (text, audio, vision).
        min_valid_ratio: treat modality as unavailable below this threshold.

    Returns:
        center_idx: LongTensor [B] with values 0=text, 1=audio, 2=vision.
    """
    eligible = valid_ratios >= min_valid_ratio
    scores = valid_ratios.clone()
    scores = scores.masked_fill(~eligible, float('-inf'))
    center_idx = scores.argmax(dim=1)
    return center_idx


def _micro_fusion_pairs_from_centers(Ut, Ua, Uv, center_idx):
    """Build [2, B, D] stacks: row0 = center modality, matching fixed-center semantics.

    Ut/Ua/Uv are [B, D] with fixed naming: text, audio, vision common projections.
    center_idx: [B] in {0,1,2} for text/audio/vision center.
    """
    # U_stacked[m, b, :] = modality m for sample b
    U_stacked = torch.stack((Ut, Ua, Uv), dim=0)
    b = torch.arange(Ut.size(0), device=Ut.device, dtype=torch.long)
    cent = U_stacked[center_idx, b]
    # Second modality in pair1 / pair2 for each fixed center mode (see GATE_F):
    p1_other = torch.tensor((2, 0, 0), device=Ut.device, dtype=torch.long)[center_idx]
    p2_other = torch.tensor((1, 2, 1), device=Ut.device, dtype=torch.long)[center_idx]
    other1 = U_stacked[p1_other, b]
    other2 = U_stacked[p2_other, b]
    pair1 = torch.stack((cent, other1), dim=0)
    pair2 = torch.stack((cent, other2), dim=0)
    return pair1, pair2
class MLP_block(nn.Module):
    def __init__(self,input_size,hidden_size,dropout=0.5):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size,input_size),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        x=self.net(x)
        return x

class MLP_Communicator(nn.Module):
    def __init__(self,token,channel,hidden_size,depth=2):
        super(MLP_Communicator,self).__init__()
        self.depth=depth
        self.token_mixer=nn.Sequential(
            Rearrange('b n d -> b d n'),
            MLP_block(input_size=channel,hidden_size=hidden_size),
            Rearrange('b n d -> b d n')
        )
        self.channel_mixer=nn.Sequential(
            MLP_block(input_size=token,hidden_size=hidden_size)
        )

    def forward(self,x):
        for _ in range(self.depth):
            x=x+self.token_mixer(x)
            x=x+self.channel_mixer(x)
        return x
    
class BottleAttentionNet(nn.Module):
    def __init__(self):
        super(BottleAttentionNet,self).__init__()
        self.embed_dim=90#mosi
        self.embed_dim_common=32
        self.seq=4

        self.layer_unimodal=2
        self.layer_multimodal=2
        self.Linear_common=nn.Linear(self.embed_dim,self.embed_dim_common)
        self.transformer=TransformerEncoder(embed_dim=self.embed_dim,num_heads=10,layers=4,attn_mask=False)

    def forward(self,audio,visual,text):
        #audio=self.Linear_common(audio)
        #visual=self.Linear_common(visual)

        for i in range(self.layer_unimodal):
            audio=self.transformer(audio)#seq_len,batchsize,dim

        for i in range(self.layer_unimodal):
            visual=self.transformer(visual)
            #text=self.transformer(text)

        fsn=torch.zeros(self.seq,audio.size(1),self.embed_dim).cuda()

        x=torch.cat([audio,fsn],dim=0)

        for i in range(self.layer_multimodal):
            if i==0:
                #[audio,fsn]-[fsn',visual]-[fsn'',text]-----[fsn''']
                x=self.transformer(x) #54，24，32
                x=torch.cat([x[audio.size(0):,:,:],visual],dim=0)
                x=self.transformer(x)
                x=torch.cat([x[:self.seq,:,:],text],dim=0)
                x = self.transformer(x)
            else:
                x=self.transformer(x)

        x=x[:self.seq,:,:]
        return x

class GRUencoder(nn.Module):
    """Pad for utterances with variable lengths and maintain the order of them after GRU"""
    def __init__(self, embedding_dim, utterance_dim, num_layers):
        super(GRUencoder, self).__init__()
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=utterance_dim,
                          bidirectional=True, num_layers=num_layers)

    def forward(self, utterance, utterance_lens):
        """Server as simple GRU Layer.
        Args:
            utterance (tensor): [utter_num, max_word_len, embedding_dim]
            utterance_lens (tensor): [utter_num]
        Returns:
            transformed utterance representation (tensor): [utter_num, max_word_len, 2 * utterance_dim]
        """
        utterance_embs = utterance.transpose(0,1)

        # SORT BY LENGTH.
        sorted_utter_length, indices = torch.sort(utterance_lens, descending=True)
        _, indices_unsort = torch.sort(indices)
        
        s_embs = utterance_embs.index_select(1, indices)

        # PADDING & GRU MODULE & UNPACK.
        utterance_packed = pack_padded_sequence(s_embs, sorted_utter_length.cpu())
        utterance_output = self.gru(utterance_packed)[0]
        utterance_output = pad_packed_sequence(utterance_output, total_length=utterance.size(1))[0]

        # UNSORT BY LENGTH.
        utterance_output = utterance_output.index_select(1, indices_unsort)

        return utterance_output.transpose(0,1)

class C_GATE(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, drop):
        super(C_GATE, self).__init__()

        # BI-GRU to get the historical context.
        self.gru = GRUencoder(embedding_dim, hidden_dim, num_layers)
        # Calculate the gate.
        self.cnn = nn.Conv1d(in_channels= 2 * hidden_dim, out_channels=1, kernel_size=3, stride=1, padding=1)
        # Linear Layer to get the representation.
        self.fc = nn.Linear(hidden_dim * 2 + embedding_dim, hidden_dim)
        # Utterance Dropout.
        self.dropout_in = nn.Dropout(drop)
        
    def forward(self, utterance, utterance_mask):
        """Returns:
            utterance_rep: [utter_num, utterance_dim]
        """
        add_zero = torch.zeros(size=[utterance.shape[0], 1], requires_grad=False).type_as(utterance_mask).to(utterance_mask.device)
        utterance_mask = torch.cat((utterance_mask, add_zero), dim=1)
        utterance_lens = torch.argmin(utterance_mask, dim=1)

        transformed_ = self.gru(utterance, utterance_lens) # [batch_size, seq_len, 2 * hidden_dim]

        gate = F.sigmoid(self.cnn(transformed_.transpose(1, 2)).transpose(1, 2))  # [batch_size, seq_len, 1]
        gate_x = torch.tanh(transformed_) * gate # [batch_size, seq_len, 2 * hidden_dim]

        utterance_rep = torch.tanh(self.fc(torch.cat([utterance, gate_x], dim=-1))) # [batch_size, seq_len, hidden_dim]

        utterance_rep = torch.max(utterance_rep, dim=1)[0] # [batch_size, hidden_dim]

        # UTTERANCE DROPOUT
        utterance_rep = self.dropout_in(utterance_rep) # [utter_num, utterance_dim]

        return utterance_rep

class GATE_F(nn.Module):
    def __init__(self, args):
        super(GATE_F, self).__init__()
        
        self.text_encoder = C_GATE(args.fusion_t_in, args.fusion_t_hid, args.fusion_gru_layers, args.fusion_drop)
        self.audio_encoder = C_GATE(args.fusion_a_in, args.fusion_a_hid, args.fusion_gru_layers, args.fusion_drop)
        self.vision_encoder = C_GATE(args.fusion_v_in, args.fusion_v_hid, args.fusion_gru_layers, args.fusion_drop)

        self.audio_visual_model = BottleAttentionNet()

       
        #1.For mosi
        self.seq_v_mosi=nn.Linear(500,50)
        self.seq_a_mosi=nn.Linear(375,50)
        
        
        #2.mosei
        self.seq_v_mosei=nn.Linear(500,50)
        self.seq_a_mosei=nn.Linear(500,50)
        
        
        #stack
        self.common_size=64
        self.dim=self.common_size
        self.batchnorm=nn.BatchNorm1d(2,affine=False)
        self.MLP_Communicator1=MLP_Communicator(self.dim,2,hidden_size=64,depth=1)
        self.MLP_Communicator2 = MLP_Communicator(self.dim, 2, hidden_size=64, depth=1)
        self.args=args
        self.fusion_center_modality = getattr(args, 'fusion_center_modality', 'text')
        if self.fusion_center_modality not in ('text', 'audio', 'vision', 'dynamic_missing'):
            raise ValueError(
                "fusion_center_modality must be 'text', 'audio', 'vision', or 'dynamic_missing'"
            )
        # Ut/Uv/Ua in common space; used for logging (micro-fusion pair semantics).
        if self.fusion_center_modality == 'dynamic_missing':
            self._micro_fusion_pair_labels = (
                '(per-sample: center from missing masks)',
                '(pair1/pair2 follow text/audio/vision center rules)',
            )
        else:
            self._micro_fusion_pair_labels = {
                'text': ('(text,vision)', '(text,audio)'),
                'audio': ('(audio,text)', '(audio,vision)'),
                'vision': ('(vision,text)', '(vision,audio)'),
            }[self.fusion_center_modality]
        self._micro_fusion_debug_printed = False
        self._dynamic_missing_first_batch_logged = False
        #stack
        self.t_stack_linear=nn.Linear(36,self.common_size)
        self.v_stack_linear = nn.Linear(48,self.common_size)
        self.a_stack_linear = nn.Linear(20,self.common_size)
        
        
        # 1.classification
        self.classifier1 = nn.Sequential()
        self.classifier1.add_module('linear_trans_norm', nn.BatchNorm1d(self.common_size * 4+90))
        self.classifier1.add_module('linear_trans_hidden', nn.Linear(self.common_size * 4+90, args.cls_hidden_dim))
        self.classifier1.add_module('linear_trans_activation', nn.LeakyReLU())
        self.classifier1.add_module('linear_trans_drop', nn.Dropout(args.cls_dropout))
        self.classifier1.add_module('linear_trans_final', nn.Linear(args.cls_hidden_dim, 1))

        # 2.concat all for classification
        self.classifier2 = nn.Sequential()
        self.classifier2.add_module('linear_trans_norm', nn.BatchNorm1d(args.fusion_t_hid + args.fusion_a_hid + args.fusion_v_hid+ self.common_size*4+90))
        self.classifier2.add_module('linear_trans_hidden', nn.Linear(args.fusion_t_hid + args.fusion_a_hid + args.fusion_v_hid  +self.common_size*4+90, args.cls_hidden_dim))
        self.classifier2.add_module('linear_trans_activation', nn.ReLU())
        self.classifier2.add_module('linear_trans_drop', nn.Dropout(args.cls_dropout))
        self.classifier2.add_module('linear_trans_final', nn.Linear(args.cls_hidden_dim, 1))

        log = logging.getLogger(__name__)
        log.info(
            'GATE_F fusion_center_modality=%s micro_fusion_pairs=%s, %s (dim0 of stack = center modality)',
            self.fusion_center_modality,
            self._micro_fusion_pair_labels[0],
            self._micro_fusion_pair_labels[1],
        )

    def forward(self, text_x, audio_x, vision_x, missing_masks_for_dynamic_center=None):
        text_x, text_mask = text_x

        #mosi mosei
        text_x_fusion=text_x.permute(1,0,2)#50,24,90 #seq,batchsize,dim
        #text_x_fusion=text_x_fusion.permute(2,0,1)#seq_len,batchsize,dim 50,24,90
  
        audio_x, audio_mask = audio_x
        audio_x_fusion=self.seq_a_mosi(audio_x.permute(0,2,1))#24,90,375--24,90,50-
        audio_x_fusion=audio_x_fusion.permute(2,0,1)#seq_len,batchsize,dim 50,24,90

        vision_x, vision_mask = vision_x
        vision_x_fusion=self.seq_v_mosi(vision_x.permute(0,2,1))
        vision_x_fusion=vision_x_fusion.permute(2,0,1)#50,24,90

        audio_visual_fusion=self.audio_visual_model(audio_x_fusion,vision_x_fusion,text_x_fusion)#4，24，90
        audio_visual_fusion=audio_visual_fusion[-1]#24,90

        #C_GATE
        text_rep = self.text_encoder(text_x, text_mask)
        text_rep_common=self.t_stack_linear(text_rep)
     
        audio_rep = self.audio_encoder(audio_x, audio_mask)
        audio_rep_common = self.a_stack_linear(audio_rep)
     
        vision_rep = self.vision_encoder(vision_x, vision_mask)
        vision_rep_common = self.v_stack_linear(vision_rep)

        # Ut, Uv, Ua in common_size (paper notation); micro-fusion pairs share one BN then flatten.
        # Original MFMB-Net: stack(Ut,Uv), stack(Ut,Ua). Coarse-to-fine here is BatchNorm1d(2) (MLP_Communicator* unused in forward).
        Ut, Uv, Ua = text_rep_common, vision_rep_common, audio_rep_common
        if self.fusion_center_modality == 'dynamic_missing':
            if missing_masks_for_dynamic_center is None:
                raise ValueError(
                    'fusion_center_modality=dynamic_missing requires missing_masks_for_dynamic_center '
                    '(tuple of text/audio/vision missing_mask tensors).'
                )
            mm_t, mm_a, mm_v = missing_masks_for_dynamic_center
            if mm_t.shape != text_mask.shape or mm_a.shape != audio_mask.shape or mm_v.shape != vision_mask.shape:
                raise ValueError(
                    'missing_masks_for_dynamic_center shapes must match fusion masks '
                    f'(text {text_mask.shape} vs mm_t {mm_t.shape}, '
                    f'audio {audio_mask.shape} vs mm_a {mm_a.shape}, '
                    f'vision {vision_mask.shape} vs mm_v {mm_v.shape}).'
                )
            valid_ratios = _compute_modality_valid_ratios_from_missing(
                mm_t, mm_a, mm_v, text_mask, audio_mask, vision_mask
            )
            center_idx = _select_dynamic_center_from_missing(valid_ratios)
            pair1, pair2 = _micro_fusion_pairs_from_centers(Ut, Ua, Uv, center_idx)
        elif self.fusion_center_modality == 'text':
            pair1 = torch.stack((Ut, Uv), dim=0)
            pair2 = torch.stack((Ut, Ua), dim=0)
        elif self.fusion_center_modality == 'audio':
            pair1 = torch.stack((Ua, Ut), dim=0)
            pair2 = torch.stack((Ua, Uv), dim=0)
        else:
            pair1 = torch.stack((Uv, Ut), dim=0)
            pair2 = torch.stack((Uv, Ua), dim=0)

        def _micro_fusion_stack_to_vec(stack_2mod):
            x = self.batchnorm(stack_2mod.permute(1, 0, 2))
            # Use actual batch dim (valid/test may use last batch < args.batch_size when drop_last=False).
            return x.reshape(x.size(0), -1)

        stack_local_a = _micro_fusion_stack_to_vec(pair1)
        stack_local_b = _micro_fusion_stack_to_vec(pair2)

        if not self._micro_fusion_debug_printed:
            log = logging.getLogger(__name__)
            log.info(
                '[GATE_F] first forward: fusion_center_modality=%s pair1=%s pair2=%s',
                self.fusion_center_modality,
                self._micro_fusion_pair_labels[0],
                self._micro_fusion_pair_labels[1],
            )
            self._micro_fusion_debug_printed = True

        if self.fusion_center_modality == 'dynamic_missing' and not self._dynamic_missing_first_batch_logged:
            log = logging.getLogger(__name__)
            n_txt = int((center_idx == _MOD_TEXT).sum().item())
            n_aud = int((center_idx == _MOD_AUDIO).sum().item())
            n_vis = int((center_idx == _MOD_VISION).sum().item())
            log.info('[GATE_F] dynamic_missing: enabled; per-sample micro-fusion center from valid ratios only.')
            log.info(
                '[GATE_F] dynamic_missing (first batch): batch center counts: text=%d, audio=%d, vision=%d (batch_size=%d)',
                n_txt, n_aud, n_vis, center_idx.numel(),
            )
            log.info(
                '[GATE_F] dynamic_missing (first batch): mean valid_ratio text=%.4f audio=%.4f vision=%.4f',
                float(valid_ratios[:, 0].mean().item()),
                float(valid_ratios[:, 1].mean().item()),
                float(valid_ratios[:, 2].mean().item()),
            )
            self._dynamic_missing_first_batch_logged = True

        #utterance_rep = torch.cat((stack_rep_tv, stack_rep_ta,audio_visual_fusion), dim=1)
        #return self.classifier1(utterance_rep)

        utterance_rep = torch.cat(
            (text_rep, audio_rep, vision_rep, stack_local_a, stack_local_b, audio_visual_fusion), dim=1
        )


        
        return self.classifier2(utterance_rep)

       
      
        

MODULE_MAP = {
    'c_gate': GATE_F,
}

class Fusion(nn.Module):
    def __init__(self, args):
        super(Fusion, self).__init__()

        select_model = MODULE_MAP[args.fusionModule]

        self.Model = select_model(args)

    def forward(self, text_x, audio_x, vision_x, missing_masks_for_dynamic_center=None):

        return self.Model(text_x, audio_x, vision_x, missing_masks_for_dynamic_center)
