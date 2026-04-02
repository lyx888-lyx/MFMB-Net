"""
AIO -- All Trains in One
"""

from trains.missingTask import *

__all__ = ['ATIO']

class ATIO():
    def __init__(self):
        self.TRAIN_MAP = {
            # missing-task
            'mfmb_net': MFMB_NET,
        }
    
    def getTrain(self, args):
        key = args.modelName.lower()
        if key.startswith('mfmb_net'):
            key = 'mfmb_net'
        return self.TRAIN_MAP[key](args)
