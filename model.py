import numpy as np
import os
import pandas as pd
import pdb
import random
import soundfile as sf
import torch
import torchaudio

import torch.nn.functional as F

from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Model
from typing import Union
from utils import AngProtoLoss4
from pytorch_metric_learning.losses import SphereFaceLoss


class Voxceleb12_sample_cssl2(Dataset):

    """
    this dataset class samples a fixed size window from an utterance
    and a positive example from another utterance of the same speaker
    works with both Vox 1 + 2
    """

    def __init__(self, base_path1, base_path2, data_df1, data_df2, seed=0, window_len=32000):
        self.base_path1 = base_path1
        self.base_path2 = base_path2
        self.rel_path = pd.concat([data_df1['rel_path'], data_df2['rel_path']]).reset_index(drop=True)
        self.speaker = pd.concat([data_df1['speaker_id'], data_df2['speaker_id']]).reset_index(drop=True)
        self.len1 = len(data_df1)
        self.window_len = window_len
        # self.speakers_to_id = speakers_to_id
        self.speakers_dict = self.get_speaker_dict()
        
        random.seed(seed)

    def get_speaker_dict(self):
        unique_speakers = set(self.speaker)
        speakers_d = defaultdict(list)
        for i, sp in enumerate(self.speaker):
            speakers_d[sp].append(i)
        
        return speakers_d

        
    def __len__(self):
        return len(self.rel_path)
    
    def __getitem__(self, index):
        
        # load audio data
        rpath = self.rel_path[index]
        if index < self.len1:
            abs_path = os.path.join(self.base_path1, rpath)
        else:
            abs_path = os.path.join(self.base_path2, rpath)

        # read in small chunk
        speech_arr = sf.SoundFile(abs_path)
        length_arr = speech_arr.frames

        max_start = length_arr - self.window_len
        start_ind = random.randint(0, max_start)
        seg_1, _ = sf.read(abs_path, start=start_ind, stop=start_ind+self.window_len)

        # for hard pos example
        curr_speaker = self.speaker[index]
        # pos_pair_ind = random.choice([ix for ix, sp in enumerate(self.speaker) if sp == curr_speaker])
        pos_pair_ind = random.choice(self.speakers_dict[curr_speaker])

        rpath_pos = self.rel_path[pos_pair_ind]
        if pos_pair_ind < self.len1:
            abs_path_pos = os.path.join(self.base_path1, rpath_pos)
        else:
            abs_path_pos = os.path.join(self.base_path2, rpath_pos)
        # read in small chunk
        speech_arr_pos = sf.SoundFile(abs_path_pos)
        length_arr_pos = speech_arr_pos.frames

        max_start_pos = length_arr_pos - self.window_len
        start_ind_pos = random.randint(0, max_start_pos)
        seg_2, _ = sf.read(abs_path_pos, start=start_ind_pos, stop=start_ind_pos+self.window_len)
        
        seg_1 = torch.tensor(seg_1).to(dtype=torch.float32)
        seg_2 = torch.tensor(seg_2).to(dtype=torch.float32)

        return seg_1, seg_2, curr_speaker


class AMI_sample_cssl(Dataset):

    """
    this dataset class samples a fixed size window from an utterance
    and a positive example from another utterance of the same speaker
    works with AMI 
    """

    def __init__(self, base_path, data_df, seed=0, window_len=32000, sampling_rate=16000):
        self.base_path = base_path
        self.meeting_id = data_df['meeting_id']
        self.start_ind = data_df['start_ind']
        self.end_ind = data_df['end_ind']
        self.speaker = data_df['label']
        self.window_len = window_len
        self.sampling_rate = sampling_rate
        self.min_train_len = 0.205 * self.sampling_rate # min length accepted by w2v2
        # self.speakers_to_id = speakers_to_id
        self.speakers_dict = self.get_speaker_dict()
        
        random.seed(seed)

    def get_speaker_dict(self):
        unique_speakers = set(self.speaker)
        speakers_d = defaultdict(list)
        for i, sp in enumerate(self.speaker):
            speakers_d[sp].append(i)
        
        print(len(speakers_d.keys()))
        return speakers_d

        
    def __len__(self):
        return len(self.meeting_id)
    
    def __getitem__(self, index):
        
        # load audio data
        file_name = self.meeting_id[index]
        file_name = file_name + '_MDM8.wav'
        abs_path = os.path.join(self.base_path, file_name)

        start_frame = self.start_ind[index] * self.sampling_rate // 100
        end_frame = self.end_ind[index] * self.sampling_rate // 100
        duration = end_frame - start_frame

        if duration < (self.window_len):
            seg_1, _ = sf.read(abs_path, start=start_frame, stop=end_frame)
            if duration < self.min_train_len:
                num_repeat = int(np.ceil(self.min_train_len / duration))
                seg_1 = np.tile(seg_1, num_repeat)
            # num_repeat = int(np.ceil(self.window_len / duration))
            # seg_1 = np.tile(seg_1, num_repeat)[:self.window_len]
        # for utterances that are long
        else:
            max_end = max(duration - self.window_len, 0)    
            rand_start = random.randint(0, max_end)
            start_frame = start_frame + rand_start
            end_frame = start_frame + self.window_len
            seg_1, _ = sf.read(abs_path, start=start_frame, stop=end_frame)

        # for hard pos example
        curr_speaker = self.speaker[index]
        # pos_pair_ind = random.choice([ix for ix, sp in enumerate(self.speaker) if sp == curr_speaker])
        pos_pair_ind = random.choice(self.speakers_dict[curr_speaker])
        pos_file_name = self.meeting_id[pos_pair_ind]
        pos_file_name = pos_file_name + '_MDM8.wav'
        pos_abs_path = os.path.join(self.base_path, pos_file_name)

        pos_start_frame = self.start_ind[pos_pair_ind] * self.sampling_rate // 100
        pos_end_frame = self.end_ind[pos_pair_ind] * self.sampling_rate // 100
        pos_duration = pos_end_frame - pos_start_frame

        if pos_duration < (self.window_len):
            seg_2, _ = sf.read(pos_abs_path, start=pos_start_frame, stop=pos_end_frame)
            if pos_duration < self.min_train_len:
                num_repeat = int(np.ceil(self.min_train_len / pos_duration))
                seg_1 = np.tile(seg_2, num_repeat)
            # num_repeat = int(np.ceil(self.window_len / pos_duration))
            # seg_2 = np.tile(seg_2, num_repeat)[:self.window_len]
        # for utterances that are long
        else:
            pos_max_end = max(pos_duration - self.window_len, 0)    
            pos_rand_start = random.randint(0, pos_max_end)
            pos_start_frame = pos_start_frame + pos_rand_start
            pos_end_frame = pos_start_frame + self.window_len
            seg_2, _ = sf.read(pos_abs_path, start=pos_start_frame, stop=pos_end_frame)
        
        seg_1 = torch.tensor(seg_1).to(dtype=torch.float32)
        seg_2 = torch.tensor(seg_2).to(dtype=torch.float32)

        return seg_1, seg_2, curr_speaker


class Wav2VecSpeakerCSSL(torch.nn.Module):

    def __init__(self, w2v2_model, config):
        super().__init__()
        self.config = config
        self.config_w2v2 = self.config['w2v2_config']

        self.model = w2v2_model   
        self.dropout = torch.nn.Dropout(self.config['dropout_val'] if 'dropout_val' in self.config else 0.2)
        self.with_relu = config['with_relu']
        self.relu = torch.nn.ReLU()
        self.layer_to_extract = config['layer_to_extract'] if 'layer_to_extract' in self.config else -1


        self.loss_fn = AngProtoLoss4(
            config,
            device=self.config["device"], 
            refine_matrix=config['refine_matrix'], 
            g_blur=config['g_blur'],
            p_pct=config['p_pct'],
            mse_fac=config['mse_fac'])

        if self.config['custom_embed_size']:
            self.fc1 = torch.nn.Linear(768, self.config["custom_embed_size"]) 
            self.init_weights()

    def freeze_base(self):
        for param in self.model.parameters():
            param.requires_grad = False
            
    def unfreeze_base(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, seg_1, seg_2, labels=None):
        f1 = self.model(seg_1).last_hidden_state
        f2 = self.model(seg_2).last_hidden_state

        f1 = torch.mean(f1, dim=1)
        f2 = torch.mean(f2, dim=1)

        f_all = torch.stack((f1, f2), dim=1)
        if self.with_relu:
            f_all = self.relu(f_all)

        if self.config['custom_embed_size']:
            f_all = self.fc1(f_all)
            if self.with_relu:
                f_all = self.relu(f_all)

        loss = self.loss_fn(f_all, labels)

        return loss, f_all

    def extract_embeddings(self, input_values):
        
        features = self.model(input_values, output_hidden_states=True)
        features = features.hidden_states[self.layer_to_extract]
        pooled_output = torch.mean(features, dim=1)
        if self.with_relu:
            pooled_output = self.relu(pooled_output)

        if self.config['custom_embed_size']:
            pooled_output = self.fc1(pooled_output)
            if self.with_relu:
                pooled_output = self.relu(pooled_output)

        return pooled_output

