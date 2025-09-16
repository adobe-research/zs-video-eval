# Copyright 2025 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.

import os
import json
import torch
from overrides import overrides
from cached_path import cached_path
from torch.utils.data import DataLoader
from typing import TypeVar

from ._base_dataset_class import GroundingVideoTextDataset
from aligner.data.video_data_module import VideoTextDataModule
from aligner.data.video_dataset import VideoDataset
from aligner.utils.typing_utils import TYPE_PATH

T = TypeVar("T")

class CharadesSTA(GroundingVideoTextDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _get_video_ids(self, annotations_file):
        annos = json.load(open(annotations_file,'r'))
        video_ids = list(annos['videos'].keys())
        return video_ids
    
    def _compute_annotaions(self, annotations_file: str) -> list:
        '''
            Parse annotation file and produce annotations list.
        '''
        annos = json.load(open(annotations_file,'r'))
        formatted_annotations = []
        cnt = 0
        for m in annos['moments']:
            vid = m['video']
            video_duration = self.video_durations[vid]
            moment = torch.tensor(m['time']).clamp(0, video_duration)
            formatted_annotations.append(
                    {
                        'video_id': vid,
                        'sentence_id': cnt,
                        'moment'  : moment,
                        'query'   : m['description'],
                        'duration': video_duration,
                    }
                )
            cnt += 1       
        return formatted_annotations 

class CharadesSTADataModule(VideoTextDataModule):  # noqa
    def __init__(self, base_path: TYPE_PATH, clip_length_in_frames: int, frames_between_clips:int, 
                frame_rate: int, dataset_name: str, use_motion_vectors: bool = False,
                 use_residuals: bool = False,**kwargs) -> None:
        super().__init__(**kwargs)
        base_path = cached_path(base_path)
        self.videos_folder = os.path.join(base_path, "videos/")
        self.train_annotations_path = os.path.join(base_path, "annotations/train.json")
        self.test_annotations_path  = os.path.join(base_path, "annotations/test.json")
        self.clip_length_in_frames  = clip_length_in_frames
        self.frames_between_clips   = frames_between_clips
        self.use_motion_vectors     = use_motion_vectors
        self.use_residuals          = use_residuals
        self.frame_rate = frame_rate
        
    def _dataset(self, annotations_file: TYPE_PATH, train: bool) -> VideoDataset:
        return CharadesSTA(video_folder=self.videos_folder, 
                     annotations_file=annotations_file,
                     clip_length_in_frames=self.clip_length_in_frames,
                     frames_between_clips=self.frames_between_clips,
                     frame_rate=self.frame_rate, train=train, 
                     use_motion_vectors=self.use_motion_vectors, use_residuals=self.use_residuals,
                     **self._create_dataset_encoder_kwargs(train=train))

    @overrides
    def train_dataloader(self) -> DataLoader:
        self.dataset = self._dataset(annotations_file=self.train_annotations_path, train=True)
        return self._create_dataloader(self.dataset, train=True)

    @overrides
    def val_dataloader(self) -> DataLoader:
        self.dataset = self._dataset(annotations_file=self.test_annotations_path, train=False)
        return self._create_dataloader(self.dataset, train=False)
    

    @overrides
    def test_dataloader(self) -> DataLoader:
        self.dataset = self._dataset(annotations_file=self.test_annotations_path, train=False)
        return self._create_dataloader(self.dataset, train=False)
