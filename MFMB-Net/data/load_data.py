import os
import logging
import pickle
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

UNK_TOKEN_ID = 100

__all__ = ['MMDataLoader']

logger = logging.getLogger('MSA')

class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATA_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            'sims': self.__init_sims,
        }
        DATA_MAP[args.datasetName]()

    def __init_mosi(self):
        with open(self.args.dataPath, 'rb') as f:
            data = pickle.load(f)
        if self.args.use_bert:
            self.text = data[self.mode]['text_bert'].astype(np.float32)
        else:
            self.text = data[self.mode]['text'].astype(np.float32)
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.rawText = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']

        self.labels = {
            'M': data[self.mode][self.args.train_mode+'_labels'].astype(np.float32)
        }
        if self.args.datasetName == 'sims':
            for m in "TAV":
                self.labels[m] = data[self.mode][self.args.train_mode+'_labels_'+m]

        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

        if not self.args.need_data_aligned:
            self.audio_lengths = data[self.mode]['audio_lengths']
            self.vision_lengths = data[self.mode]['vision_lengths']
        self.audio[self.audio == -np.inf] = 0

        self.text_corrupt_rate = float(getattr(self.args, 'text_corrupt_train', 0.0) if self.mode == 'train' else getattr(self.args, 'text_corrupt_eval', 0.0))
        self.text_corrupt_mode = getattr(self.args, 'text_corrupt_mode', 'none')
        self.text_corrupt_span_frac = float(getattr(self.args, 'text_corrupt_span_frac', 0.4))
        self.text_corrupt_seed = int(getattr(self.args, 'text_corrupt_seed', 2026))

        if self.args.data_missing:
            # Current Support Unaligned Data Missing.
            self.text_m, self.text_length, self.text_mask, self.text_missing_mask = self.generate_m(self.text[:,0,:], self.text[:,1,:], None,
                                                                                        self.args.missing_rate[0], self.args.missing_seed[0], mode='text')
            # use_distill=1 且 online_text_corrupt=1：扰动改在 model 内只做一次，此处跳过，避免与 maybe_corrupt_text_m_only 双重污染
            _online_corrupt = int(getattr(self.args, 'online_text_corrupt', 0)) != 0
            _use_distill = int(getattr(self.args, 'use_distill', 0)) != 0
            if _use_distill and _online_corrupt:
                self.text_corrupt_mask = np.zeros_like(self.text_missing_mask, dtype=np.float32)
            else:
                self.text_m, self.text_corrupt_mask = self.generate_text_corruption(
                    self.text_m, self.text_mask, self.text_missing_mask,
                    self.text_corrupt_rate, self.text_corrupt_seed, self.text_corrupt_mode, self.text_corrupt_span_frac
                )
            Input_ids_m = np.expand_dims(self.text_m, 1)
            Input_mask = np.expand_dims(self.text_mask, 1)
            Segment_ids = np.expand_dims(self.text[:,2,:], 1)
            self.text_m = np.concatenate((Input_ids_m, Input_mask, Segment_ids), axis=1) 

            self.audio_m, self.audio_length, self.audio_mask, self.audio_missing_mask = self.generate_m(self.audio, None, self.audio_lengths,
                                                                                        self.args.missing_rate[1], self.args.missing_seed[1], mode='audio')
            self.vision_m, self.vision_length, self.vision_mask, self.vision_missing_mask = self.generate_m(self.vision, None, self.vision_lengths,
                                                                                        self.args.missing_rate[2], self.args.missing_seed[2], mode='vision')
        if self.args.need_truncated:
            self.__truncated()

        if  self.args.need_normalized:
            self.__normalize()
    
    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()

    def generate_m(self, modality, input_mask, input_len, missing_rate, missing_seed, mode='text'):
        
        if mode == 'text':
            input_len = np.argmin(input_mask, axis=1)
        elif mode == 'audio' or mode == 'vision':
            input_mask = np.array([np.array([1] * length + [0] * (modality.shape[1] - length)) for length in input_len])
        np.random.seed(missing_seed)
        missing_mask = (np.random.uniform(size=input_mask.shape) > missing_rate) * input_mask
        
        assert missing_mask.shape == input_mask.shape
        
        if mode == 'text':
            # CLS and SEP tokens unchanged.
            for i, instance in enumerate(missing_mask):
                sep_idx = max(int(input_len[i]) - 1, 0)
                instance[0] = 1
                instance[sep_idx] = 1

            modality_m = missing_mask * modality + (UNK_TOKEN_ID * np.ones_like(modality)) * (input_mask - missing_mask) # UNK token.
        elif mode == 'audio' or mode == 'vision':
            modality_m = missing_mask.reshape(modality.shape[0], modality.shape[1], 1) * modality
        
        return modality_m, input_len, input_mask, missing_mask

    def generate_text_corruption(self, input_ids, input_mask, missing_mask, corrupt_rate, corrupt_seed, mode='mix', span_frac=0.4):
        input_ids = input_ids.copy()
        corrupt_mask = np.zeros_like(input_ids, dtype=np.float32)
        if corrupt_rate <= 0 or mode == 'none':
            return input_ids, corrupt_mask

        rng = np.random.RandomState(corrupt_seed + {'train': 0, 'valid': 1000, 'test': 2000}[self.mode])
        for i in range(input_ids.shape[0]):
            valid_positions = np.where(input_mask[i] > 0)[0]
            if len(valid_positions) <= 2:
                continue
            sep_idx = valid_positions[-1]
            candidate = valid_positions[(valid_positions != 0) & (valid_positions != sep_idx)]
            if candidate.size == 0:
                continue
            candidate = candidate[missing_mask[i, candidate] > 0]
            if candidate.size == 0:
                continue

            budget = int(round(candidate.size * corrupt_rate))
            budget = min(max(budget, 0), int(candidate.size))
            if budget <= 0:
                continue

            selected = []
            if mode in ('span', 'mix'):
                span_budget = budget if mode == 'span' else int(round(budget * span_frac))
                if span_budget > 0:
                    ordered = np.sort(candidate)
                    max_span = max(1, min(4, span_budget))
                    tries = 0
                    used = set()
                    while len(selected) < span_budget and tries < 20:
                        tries += 1
                        start = int(rng.choice(ordered))
                        span_len = int(rng.randint(1, max_span + 1))
                        current = [idx for idx in range(start, start + span_len) if idx in ordered and idx not in used]
                        if not current:
                            continue
                        for idx in current:
                            used.add(idx)
                            selected.append(idx)
                            if len(selected) >= span_budget:
                                break

            if mode in ('token', 'mix'):
                remaining = budget - len(selected)
                if remaining > 0:
                    available = np.array([idx for idx in candidate if idx not in set(selected)])
                    if available.size > 0:
                        chosen = rng.choice(available, size=min(remaining, available.size), replace=False)
                        selected.extend(chosen.tolist())

            if len(selected) < budget:
                available = np.array([idx for idx in candidate if idx not in set(selected)])
                if available.size > 0:
                    filler = rng.choice(available, size=min(budget - len(selected), available.size), replace=False)
                    selected.extend(filler.tolist())

            if not selected:
                continue
            selected = np.array(sorted(set(selected)), dtype=np.int64)
            input_ids[i, selected] = UNK_TOKEN_ID
            corrupt_mask[i, selected] = 1.0
        return input_ids, corrupt_mask

    def __truncated(self):
        # NOTE: Here for dataset we manually cut the input into specific length.
        def Truncated(modal_features, length):
            if length == modal_features.shape[1]:
                return modal_features
            truncated_feature = []
            padding = np.array([0 for i in range(modal_features.shape[2])])
            for instance in modal_features:
                for index in range(modal_features.shape[1]):
                    if((instance[index] == padding).all()):
                        if(index + length >= modal_features.shape[1]):
                            truncated_feature.append(instance[index:index+20])
                            break
                    else:                        
                        truncated_feature.append(instance[index:index+20])
                        break
            truncated_feature = np.array(truncated_feature)
            return truncated_feature
                       
        text_length, audio_length, video_length = self.args.seq_lens
        self.vision = Truncated(self.vision, video_length)
        self.text = Truncated(self.text, text_length)
        self.audio = Truncated(self.audio, audio_length)

    def __normalize(self):
        # (num_examples,max_len,feature_dim) -> (max_len, num_examples, feature_dim)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        # for visual and audio modality, we average across time
        # here the original data has shape (max_len, num_examples, feature_dim)
        # after averaging they become (1, num_examples, feature_dim)
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)

        # remove possible NaN values
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

        if self.args.data_missing:
            self.vision_m = np.transpose(self.vision_m, (1, 0, 2))
            self.audio_m = np.transpose(self.audio_m, (1, 0, 2))

            self.vision_m = np.mean(self.vision_m, axis=0, keepdims=True)
            self.audio_m = np.mean(self.audio_m, axis=0, keepdims=True)
    
            # remove possible NaN values
            self.vision_m[self.vision_m != self.vision_m] = 0
            self.audio_m[self.audio_m != self.audio_m] = 0

            self.vision_m = np.transpose(self.vision_m, (1, 0, 2))
            self.audio_m = np.transpose(self.audio_m, (1, 0, 2))

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        if self.args.use_bert:
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):
        if self.args.data_missing:
            sample = {
                'text': torch.Tensor(self.text[index]), # [batch_size, 3, 50]
                'text_m': torch.Tensor(self.text_m[index]), # [batch_size, 3, 50]
                'text_missing_mask': torch.Tensor(self.text_missing_mask[index]),
                'text_corrupt_mask': torch.Tensor(self.text_corrupt_mask[index]) if hasattr(self, 'text_corrupt_mask') else torch.zeros_like(torch.Tensor(self.text_missing_mask[index])),
                'audio': torch.Tensor(self.audio[index]),
                'audio_m': torch.Tensor(self.audio_m[index]),
                'audio_lengths': self.audio_lengths[index],
                'audio_mask': self.audio_mask[index],
                'audio_missing_mask': self.audio_missing_mask[index],
                'vision': torch.Tensor(self.vision[index]),
                'vision_m': torch.Tensor(self.vision_m[index]),
                'vision_lengths': self.vision_lengths[index],
                'vision_mask': self.vision_mask[index],
                'vision_missing_mask': self.vision_missing_mask[index],
                'index': index,
                'id': self.ids[index],
                'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
            }
        else:
            sample = {
                'raw_text': self.rawText[index],
                'text': torch.Tensor(self.text[index]), 
                'audio': torch.Tensor(self.audio[index]),
                'vision': torch.Tensor(self.vision[index]),
                'index': index,
                'id': self.ids[index],
                'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
            } 
            if not self.args.need_data_aligned:
                sample['audio_lengths'] = self.audio_lengths[index]
                sample['vision_lengths'] = self.vision_lengths[index]
        return sample

def MMDataLoader(args):

    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    if 'seq_lens' in args:
        args.seq_lens = datasets['train'].get_seq_len() 

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args.batch_size,
                       num_workers=args.num_workers,
                       shuffle=True,
                       drop_last=True
                       )
        for ds in datasets.keys()
    }
    
    return dataLoader