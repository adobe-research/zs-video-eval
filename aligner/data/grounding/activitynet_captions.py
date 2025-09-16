# Copyright 2025 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.

import os
import json
import math
import torch
import functools
import torchvision
from pathlib import Path
from overrides import overrides
from cached_path import cached_path
from torch.utils.data import DataLoader
from typing import TypeVar, Mapping, Union

from torchvision.transforms.functional import to_tensor
from aligner.data.video_data_module import VideoTextDataModule
from aligner.data.video_dataset import VideoDataset
from aligner.data.video_text_dataset import VideoTextDataset
from aligner.utils.typing_utils import TYPE_PATH
from aligner.utils.video_utils import NormalizeVideo

import PIL

T = TypeVar("T")

class ClipIteratorAnet():
    def __init__(self, video_paths: list, clip_length_in_frames: int, frames_between_clips: int, frame_rate: float, 
                    video_info_file: str, use_motion_vectors: bool = False, use_residuals: bool = False, 
                    precomputed_compressed_information: str = '')-> None:

        # self.video_paths = video_paths
        self.clip_length_in_frames = clip_length_in_frames    
        self.frames_between_clips = frames_between_clips
        self.frame_rate = frame_rate
        self.use_motion_vectors = use_motion_vectors
        self.use_residuals = use_residuals
        self.precomputed_compressed_information = precomputed_compressed_information
        self.compressed_info_file = None

        self.metadata = {'video_paths': video_paths}
        self.video_durations, self.video_fps, self.video_num_frames = \
                self._compute_video_durations(video_paths, video_info_file)
        self._compute_frame_indexes(video_paths)

    def __len__(self):
        return len(self.clips)
    
    def get_video_durations(self):
        return self.video_durations

    def get_video_fps(self):
        return self.video_fps

    def _compute_video_durations(self,video_paths, video_info_file):
        video_durations = {}
        video_fps = {}
        video_num_frames = {}
        failed = False
        if os.path.exists(video_info_file):
            with open(video_info_file, 'r') as f:
                video_info = json.load(f)
                video_durations  = video_info['video_durations']
                video_num_frames = video_info['video_num_frames']
                video_fps        = video_info['video_fps']

            # Change the FPS to 1 as we extracted the frames at 1 FPS
            video_fps = {k:1.0 for k in video_fps.keys()}
            # Get the number of frames from the video paths so we don't mess around
            video_num_frames = {Path(path).stem: len(os.listdir(path)) for path in video_paths}

        else:
            raise ValueError('Not implemented yet. We never extracted this info, and for Anet we work over frames. ')
            for path in video_paths:
                try:
                    video_reader = VideoReader.from_path(path)
                    num_frames = len(video_reader)
                    fps = video_reader.get_avg_fps()
                    video_duration = num_frames/fps

                    video_id = Path(path).stem
                    video_num_frames[video_id] = num_frames
                    video_durations[video_id] = video_duration
                    video_fps[video_id] = fps
                except:
                    print(path)
                    failed=True
                
            if failed:
                sys.exit(0)
                
            with open(video_info_file, 'w') as f:
                json.dump({'video_durations': video_durations, 'video_fps': video_fps, 'video_num_frames': video_num_frames}, f)

        return video_durations, video_fps, video_num_frames
                   
    def _compute_frame_indexes(self, video_paths):
        clip_counter = 0
        self.clips = []

        video_paths = {v.split('/')[-1]: v for v in video_paths}

        for i, (video_id, total_frames) in enumerate(self.video_num_frames.items()):
            frame_skip = self.video_fps[video_id] / self.frame_rate

            # Calculate the number of possible clips in the video
            frame_step_size = int(round(frame_skip * self.clip_length_in_frames))  # total frames covered in one full clip
            num_windows = math.ceil(total_frames / frame_step_size)

            for window_number in range(num_windows):
                start_frame = window_number * frame_step_size
                end_frame = start_frame + frame_step_size
                # Calculate the frame indices for this clip
                frame_indices = torch.arange(start_frame, end_frame, frame_skip, dtype=torch.int)
                valid_frames  = frame_indices < total_frames
                frame_indices = frame_indices.clamp(1, total_frames).tolist()  # Ensure indices are within bounds
                frame_paths = [os.path.join(video_paths[video_id], f'{video_id}_{i:06d}.jpg') for i in frame_indices]

                self.clips.append({
                    'video_id': video_id,
                    'clip_idx': clip_counter,
                    'video_idx': i,
                    'frame_paths': frame_paths,
                    'valid_frames': valid_frames.numpy()  # Tensor of validity indicators
                })
                clip_counter += 1

    def get_clip_location(self, idx: int):
        return self.clips[idx]['video_id'], self.clips[idx]['video_idx'], self.clips[idx]['clip_idx']

    def get_compressed_info(self, video_id, video_path, frame_indexes, window_size=10, scale=4):
        if self.compressed_info_file is None and os.path.isfile(self.precomputed_compressed_information):
            self.compressed_info_file = h5py.File(self.precomputed_compressed_information, 'r')

        if self.compressed_info_file is None:
            max_num_frames =  self.video_num_frames[video_id]
            all_indexes = sorted(set(
                            idx
                            for i in frame_indexes
                            for idx in range(max(0, i - window_size // 2), min(i + window_size // 2+1, max_num_frames))
                        ))

            video_frames = cv_reader.read_video_frames(video_path=video_path, frame_indexes=all_indexes, with_residual=self.use_residuals)
            video_frames = {i: vf for i, vf in zip(all_indexes, video_frames)}
  
            motion_vectors, residuals = [], []
            for i in frame_indexes:
                start_idx = max(0,i-window_size//2)
                stop_idx  = min(i+window_size//2+1, max_num_frames)

                if self.use_residuals:
                    residual_block = [torch.from_numpy(video_frames[block_idx]['residual']) for block_idx in range(start_idx, stop_idx)]
                    # preprocess each image by removing the mean, normalizing, taking the norm across channels and then the average across frames
                    residual_block = torch.stack(residual_block).float() / 255 - 0.5
                    # residual_block = torch.stack(residual_block).mean(dim=0, dtype=torch.float)/255
                    residual_block_norm = torch.norm(residual_block, dim=-1)
                    if len(residual_block_norm.shape) > 2:
                       residual_block_norm = residual_block_norm.mean(0)
                    residuals.append(residual_block_norm)
                else:
                    residuals.append(torch.empty(1))

                if self.use_motion_vectors:
                    motion_vectos_block = [video_frames[block_idx]['motion_vector'] for block_idx in range(start_idx, stop_idx)]
                    compounded_motion_vector = np.array(motion_vectos_block).sum(axis=(0,-1))
                    upscaled_motion_vector = np.repeat(np.repeat(compounded_motion_vector, scale, axis=0), scale, axis=1)
                    motion_vectors.append(torch.from_numpy(upscaled_motion_vector))
                else:
                    motion_vectors.append(torch.empty(1))

            motion_vectors = torch.stack(motion_vectors).float()
            residuals = torch.stack(residuals)

            del video_frames
            return {'motion_vectors': motion_vectors, 'residuals': residuals}

        else:
            if self.use_residuals:
                raise ValueError('Not implemented yet.')

            video_frames_motion_vectors = self.compressed_info_file[f'{video_id}']['motion_vectors']
            # pict_types = self.compressed_info_file[f'{video_id}/pict_types'][frame_indexes]

            motion_vectors = []
            for i in frame_indexes:
                start_idx = max(0, i - window_size // 2)
                stop_idx  = min(i + window_size // 2 + 1, len(video_frames_motion_vectors))

                motion_vectos_block = [video_frames_motion_vectors[block_idx] for block_idx in range(start_idx, stop_idx)]
                compounded_motion_vector = np.array(motion_vectos_block).sum(axis=0)

                try: 
                    upscaled_motion_vector = np.repeat(np.repeat(compounded_motion_vector, scale, axis=0), scale, axis=1)
                except Exception as e:
                    print(video_id, len(video_frames_motion_vectors), start_idx, stop_idx)
                    first_frame_shape = video_frames_motion_vectors[0].shape
                    motion_vector = np.zeros(first_frame_shape)
                    upscaled_motion_vector = np.repeat(np.repeat(motion_vector, scale, axis=0), scale, axis=1)
                
                motion_vectors.append(torch.from_numpy(upscaled_motion_vector))
            motion_vectors = torch.stack(motion_vectors).float()
            return {'motion_vectors': motion_vectors}

    def get_clip(self, idx: int):
        video_idx = self.clips[idx]['video_idx']
        frame_paths = self.clips[idx]['frame_paths']
        valid_frames = self.clips[idx]['valid_frames']
        frames = [to_tensor(PIL.Image.open(f)) for f in frame_paths]
        frames = torch.stack(frames).permute(0, 2, 3,1)

        compressed_info = {'motion_vectors': torch.empty(0), 'residuals': torch.empty(0)}
        if self.use_motion_vectors or self.use_residuals:
            video_id = self.clips[idx]['video_id']
            compressed_info = self.get_compressed_info(video_id, video_path, frame_indexes)

        return frames, compressed_info, valid_frames

class ActivityNetCaptions(VideoTextDataset):
    def __init__(self, video_folder: list, annotations_file: TYPE_PATH, clip_length_in_frames: int, 
                 frames_between_clips: int, frame_rate: int, train: str, use_motion_vectors:bool = False,
                 use_residuals: bool = False, **kwargs) -> None:
        
        self.cnt = 0
        self.train = train
        self.video_folder = video_folder
        self.clip_length_in_frames = clip_length_in_frames
        self.frames_between_clips  = frames_between_clips
        self.frame_rate = frame_rate 
        self.use_motion_vectors = use_motion_vectors
        self.use_residuals= use_residuals

        self.video_ids   = self._get_video_ids(annotations_file)
        all_videos       = {os.path.splitext(f)[0]:f for f in os.listdir(video_folder)} # Trick to avoid dealoing with file extensions
        self.video_paths = [os.path.join(video_folder, all_videos[f]) for f in self.video_ids]
        
        self.clip_iterator, self.video_durations, self.video_fps = \
                    self._setup_clip_sampler(annotations_file, use_motion_vectors, use_residuals)
        self.clip_indices = torch.arange(len(self.clip_iterator))

        self._denormalize_video = self._build_denormalize_video()
        self.annos       = self._compute_annotaions(annotations_file)

        super().__init__(video_paths=self.video_paths, **kwargs)

        if self.use_motion_vectors:
            self.transform_map['motion_vectors'] = self._build_motion_vectors_transform()

        if self.use_residuals:
            self.transform_map['residuals'] = self._build_residuals_transform()
     
    def __len__(self) -> int:
        return len(self.clip_indices)

    def _build_denormalize_video(self):
        return torchvision.transforms.Compose(
            [NormalizeVideo(mean=[0., 0., 0.], std=[1/(0.26862954), 1/(0.26130258), 1/(0.27577711)]),
            NormalizeVideo(mean=[-0.48145466, -0.4578275, -0.40821073], std=[1., 1., 1.]),
            ]
            )

    def _build_motion_vectors_transform(self):
        return torchvision.transforms.Compose([
                        torchvision.transforms.Resize(size=(224, 224), antialias=True),
                        torchvision.transforms.CenterCrop(size=(224, 224)),
                    ])

    def _build_residuals_transform(self):
        return torchvision.transforms.Compose([
                        torchvision.transforms.Resize(size=(224, 224), antialias=True),
                        torchvision.transforms.CenterCrop(size=(224, 224)),
                    ])
          
    def _get_annotation(self, sentence_idx: int) -> dict:  
        return self.annos[sentence_idx]    
      
    def _get_clip_location(self, video_idx: int) -> str:
        video_id, video_index, clip_index = self.clip_iterator.get_clip_location(video_idx)
        video_id2 = Path(self.clip_iterator.metadata['video_paths'][video_index]).stem
        assert video_id == video_id2

        if isinstance(clip_index, torch.Tensor):
            clip_index = clip_index.item()
        return video_id, clip_index
    
    def _get_val_item(self, video_idx: int) -> Mapping[str, Union[torch.Tensor, str, T]]:
        video_idx = self.clip_indices[video_idx].item()                #Trick to deal with the duration agnostic setup
        frames, compressed_info, valid_frames = self.clip_iterator.get_clip(video_idx)

        dataset_output = {
                self.target_key_name: 'empty query', #Not used
                "video_id": video_idx,
                "valid_frames": valid_frames,
                "video":  self.transform_map['video'](frames)
            }

        if self.use_motion_vectors:
            dataset_output['motion_vectors'] = self.transform_map['motion_vectors'](compressed_info['motion_vectors'])

        if self.use_residuals:
            dataset_output['residuals'] = self.transform_map['residuals'](compressed_info['residuals'])

        return dataset_output
    
    @functools.lru_cache(maxsize=None)
    def _cached_get_item(self, video_idx: int) -> Mapping[str, Union[torch.Tensor, str, T]]:
        if self.train == True:
            raise ValueError('Not implemented yet.')
        else:
            return self._get_val_item(video_idx)
        
    @overrides
    def _get_target(self, video_idx: int) -> str:
        return ''  # No idea what this is for
    
    def _get_video_ids(self, annotations_file):
        annos = json.load(open(annotations_file,'r'))
        video_ids = list(annos.keys())
        return video_ids

    def _compute_annotaions(self, annotations_file: str) -> list:
        '''
            Parse annotation file and produce annotations list.
        '''
        annos = json.load(open(annotations_file,'r'))
        formatted_annotations = []
        cnt = 0
        num_invalid_timestamps = 0
        for vid, anno in annos.items():
            video_duration = self.video_durations[vid]
            
            # Produce annotations
            for timestamp, sentence in zip(anno['timestamps'], anno['sentences']):
                if timestamp[1] <= timestamp[0]:
                    print(f"Invalid timestamps: {timestamp}")
                    timestamp[1] = timestamp[0]+ 0.1 # hack for feature extractions
                    num_invalid_timestamps += 1

                moment = torch.tensor(timestamp).clamp(0, video_duration)
                formatted_annotations.append(
                        {
                            'video_id': vid,
                            'sentence_id': cnt,
                            'moment'  : moment,
                            'query'   : sentence,
                            'duration': video_duration,
                        }
                    )
                cnt += 1
        print(f'Number of invalid timestamps: {num_invalid_timestamps}/ {len(formatted_annotations)}')
        return formatted_annotations 
    
    def _setup_clip_sampler(self, annotations_file, use_motion_vectors, use_residuals):
        if self.train == True:
            raise ValueError('Not implemented yet.')
        else:
            split = Path(annotations_file).stem.split('_')[-1]
            root = Path(annotations_file).parent.parent
            video_info_file = os.path.join(root, f'{split}_video_info.json')

            precomputed_compressed_information=''
            if use_motion_vectors and not use_residuals:
                if 'activitynet_captions' in annotations_file and (split=='val' or split=='test'):
                    precomputed_compressed_information = \
                        os.path.join(root, f'motion_vectors_val.h5')
                else:
                    precomputed_compressed_information = \
                        os.path.join(root, f'motion_vectors_{split}.h5')

            clip_iterator = ClipIteratorAnet(
                                video_paths=self.video_paths, 
                                clip_length_in_frames=self.clip_length_in_frames,
                                frames_between_clips=self.frames_between_clips, 
                                frame_rate=self.frame_rate, 
                                video_info_file=video_info_file,
                                use_motion_vectors=use_motion_vectors,
                                use_residuals=use_residuals,
                                precomputed_compressed_information=precomputed_compressed_information
                                )
            video_fps = clip_iterator.get_video_fps()
            video_durations = clip_iterator.get_video_durations()
        return clip_iterator, video_durations, video_fps


class ActivityNetCaptionsDataModule(VideoTextDataModule):  # noqa
    def __init__(self, base_path: TYPE_PATH, clip_length_in_frames: int, frames_between_clips:int, 
                frame_rate: int, dataset_name: str, use_motion_vectors: bool = False,
                 use_residuals: bool = False,**kwargs) -> None:
        super().__init__(**kwargs)
        base_path = cached_path(base_path)
        # self.videos_folder = os.path.join(base_path, "videos/")
        self.videos_folder = os.path.join(base_path, "frames/")
        self.train_annotations_path = os.path.join(base_path, "annotations/train.json")
        self.val_annotations_path   = os.path.join(base_path, "annotations/val.json")
        self.test_annotations_path  = os.path.join(base_path, "annotations/test.json")
        self.clip_length_in_frames  = clip_length_in_frames
        self.frames_between_clips   = frames_between_clips
        self.use_motion_vectors     = use_motion_vectors
        self.use_residuals          = use_residuals
        self.frame_rate = frame_rate
        
    def _dataset(self, annotations_file: TYPE_PATH, train: bool) -> VideoDataset:
        return ActivityNetCaptions(video_folder=self.videos_folder, 
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
        # self.dataset = self._dataset(annotations_file=self.val_annotations_path, train=False)
        self.dataset = self._dataset(annotations_file=self.test_annotations_path, train=False)
        return self._create_dataloader(self.dataset, train=False)

    @overrides
    def test_dataloader(self) -> DataLoader:
        self.dataset = self._dataset(annotations_file=self.test_annotations_path, train=False)
        return self._create_dataloader(self.dataset, train=False)
