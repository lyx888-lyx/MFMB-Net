"""
AIO -- All Model in One
"""
import torch.nn as nn

from models.missingTask import *

__all__ = ['AMIO']

MODEL_MAP = {
    # missing-task
    'mfmb_net': MFMB_NET,
}

def _resolve_model_key(model_name: str) -> str:
    m = str.lower(model_name)
    if m.startswith('mfmb_net'):
        return 'mfmb_net'
    return m


class AMIO(nn.Module):
    def __init__(self, args):
        super(AMIO, self).__init__()
        # simulating word-align network (for seq_len_T == seq_len_A == seq_len_V)
        key = _resolve_model_key(args.modelName)
        lastModel = MODEL_MAP[key]
        self.Model = lastModel(args)

    def forward(
        self,
        text_x,
        audio_x,
        video_x,
        text_corrupt_mask=None,
        return_fusion_aux=False,
        return_anchor_analysis=False,
    ):
        return self.Model(
            text_x, audio_x, video_x,
            text_corrupt_mask=text_corrupt_mask,
            return_fusion_aux=return_fusion_aux,
            return_anchor_analysis=return_anchor_analysis,
        )
