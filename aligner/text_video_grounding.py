# Copyright 2025 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.

import os
import math
import pandas as pd
from collections import OrderedDict
from hydra.utils import to_absolute_path

from typing import  Any, Optional, Sequence, Tuple, Union, Mapping, MutableMapping

import torch
from torch.nn import functional 
import torch.distributed.nn
from overrides import overrides

import pytorch_lightning as pl
from torch import nn

try:
    import wandb
except ImportError:
    wandb = None

from aligner.encoder.video_text_encoder import TYPE_OUTPUT, VideoTextEncoder
from aligner.utils.tensor_utils import all_gather
from aligner.utils.video_utils import _iou
from aligner.utils.post_processing import SimpleWatershed, SimpleMovingAverage

import logging

LOGGER = logging.getLogger(__name__)
TYPE_INPUT = MutableMapping[str, Any]

class TextVideoGroundingLightningModule(pl.LightningModule):  
    def __init__(self, encoder: VideoTextEncoder, 
                 model_name: str, dataset_name: str = None, compute_rank: bool = False,
                 datamodule = None, output_dir = None, evaluation_epoch: str = None, 
                 enable_wandb: bool= False, watershed_threshold: float = 1.0, 
                 watershed_scoring: str = 'max', patch_drop:float = 0.0, 
                 patch_merge:int = 0, smoothing_function=None, smoothing_window_size=1, 
                 init_temperature: float = 0.05, min_temperature: float = 0.001, 
                 fit_temperature: bool = True, **kwargs) -> None:

        super().__init__()
        
        self.encoder = encoder

        # Use the temperature as in CLIP: save it in log-space and fit it along with the model.
        self.logit_scale = nn.Parameter(torch.tensor([- math.log(init_temperature)]), requires_grad=fit_temperature)
        # The following constant is set also as a parameter, so it's moved to the correct device automatically.
        self.max_logit_scale = nn.Parameter(torch.tensor([- math.log(min_temperature)]), requires_grad=False)

        self.dataset_name  = dataset_name
        self.datamodule    = datamodule
        
        # Define metric parameters
        self.iou_metrics = torch.tensor([0.3, 0.5, 0.7])
        self.recall_metrics = torch.tensor([1, 5])
        self.enable_wandb = enable_wandb

        # Logging
        self.output_dir = os.path.join(output_dir, dataset_name, \
                                    model_name.replace('/', '_') + \
                                    f'_epoch_{evaluation_epoch}' + \
                                    f'__INPUT__frame_rate_{datamodule.frame_rate}'+ \
                                    f'_patch_drop_{patch_drop}'+ \
                                    f'_patch_merge_{patch_merge}'+ \
                                    f'_N_{datamodule.clip_length_in_frames-1:02d}'+ \
                                    f'__WATERSHED__threshold_{watershed_threshold}_scoring_{watershed_scoring}' 
                                    )
        if not os.path.isdir(to_absolute_path(self.output_dir)):
            os.makedirs(to_absolute_path(self.output_dir))
        
        # Post-processing
        self.watershed = SimpleWatershed(watershed_threshold, watershed_scoring)

        self.smoothing_function = None
        if smoothing_function is not None:
            if smoothing_function == 'SMA':
                smoothing_params = {'window_size': smoothing_window_size}
                self.smoothing_function = SimpleMovingAverage(**smoothing_params)
            else:
                raise ValueError('Unknown smoothing function.')
        
    @overrides(check_signature=False)
    def forward(self, batch: TYPE_INPUT) -> torch.Tensor:
        return self.encoder.encode_video(**batch)

    def _step(self, batch: TYPE_INPUT) -> TYPE_OUTPUT:
        return self(batch)

    @overrides(check_signature=False)
    def validation_step(self, batch: TYPE_INPUT, batch_idx: int = 0,
                        dataloader_idx: Optional[int] = None) -> Tuple[TYPE_OUTPUT, Optional[int]]:
        
        clip_ids = batch['video_id']
        valid_frames = batch['valid_frames'].squeeze(1)
        batch = {k: v for k, v in batch.items() if k in {"video", "motion_vectors"}}
        output = self._step(batch)
        
        output = output["encoded_images"]
        return (output, clip_ids, valid_frames)

    def _pack_batch_emebeddings(self, encoded_video, clip_ids, valid_frames):
        # Unpack all video snippets/frames embeddings and pack into dictionaries
        video_embeddings = {}
        for e, c, v in zip(encoded_video, clip_ids, valid_frames):
            video_id, clip_idx = self.datamodule.dataset._get_clip_location(c.to('cpu'))
            if video_id not in video_embeddings:
                video_embeddings[video_id] = {clip_idx:e[v.bool()].to('cpu')}
            else:
                video_embeddings[video_id][clip_idx] = e[v.bool()].to('cpu')
        return video_embeddings
    
    @overrides(check_signature=False)
    def validation_step_end(self, output: Tuple[TYPE_OUTPUT, int]) -> TYPE_OUTPUT:
        return self._pack_batch_emebeddings(*all_gather(self, output))

    def _pack_multiple_batches_emebeddings(self, outputs: Sequence[TYPE_OUTPUT]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Create a dictinary containing all text embeddings, where the key is the sentence id in the dataset
        # Also create a dictionary with video embeddings, where the key is the video id and the value is a tensor with all 
        # frame/clips embeddings

        video_embeddings = {}
        for output in outputs:
            for video_id, bv in output.items():
                for clip_idx, e in bv.items():
                    if video_id not in video_embeddings:
                        video_embeddings[video_id] = {clip_idx:e}
                    else:
                        video_embeddings[video_id][clip_idx] = e

        for video_id, video_embeddings_dict in video_embeddings.items():
            sorted_frames = sorted(list(video_embeddings_dict.keys()))
            if len(video_embeddings_dict[sorted_frames[0]].shape)==1:
                video_embeddings[video_id] = torch.stack([video_embeddings_dict[clip_idx] for clip_idx in sorted_frames])
            else:
                video_embeddings[video_id] = torch.cat([video_embeddings_dict[clip_idx] for clip_idx in sorted_frames])

        return video_embeddings

    def _compute_text_embeddings(self, annotations):
        sentences = {a['sentence_id']: a['query'] for a in annotations}
        batch_size = 256
        num_batches = math.ceil(len(sentences)/batch_size)
        text_embeddings = {}
        tokenizer = self.datamodule.dataset.tokenizer

        for i in range(num_batches):
            keys = list(sentences.keys())[i*batch_size:(i+1)*batch_size]
            batch_sentences = list(sentences.values())[i*batch_size:(i+1)*batch_size]
            batch_sentences = tokenizer(batch_sentences)
            batch_sentences['input_ids'] = batch_sentences['input_ids'].to('cuda')
            encoded_text = self.encoder.encode_text(batch_sentences).to('cpu')
            for k, e in zip(keys, encoded_text):
                text_embeddings[k] = e
        return text_embeddings
        
    def _compute_metrics(self, annotations, text_embeddings, video_embeddings, frame_rate):        
        # Setup metrics 
        max_recall = self.recall_metrics.max()
        num_iou_metrics = len(self.iou_metrics)
        recall_x_iou = torch.zeros((len(self.recall_metrics),len(self.iou_metrics)))

        # Loop over predictions
        LOGGER.info("Compute metrics.")
        for anno in annotations:
            # Get movie features and sentence features
            gt_moment = anno['moment']
            l_feat = text_embeddings[anno['sentence_id']][None,:]
            v_feat = video_embeddings[anno['video_id']]

            sim = functional.cosine_similarity(l_feat, v_feat, dim=1)   
            if self.smoothing_function:
                sim = self.smoothing_function.smooth(sim)  
            best_moments, best_scores = self.watershed(sim)
            best_moments = best_moments / frame_rate
            
            if len(best_moments) < max_recall:
                tmp_moments = torch.zeros((max_recall, 2))
                tmp_moments[:,1] = anno['duration']
                if len(best_moments) > 0:
                    tmp_moments[:len(best_moments)] = best_moments
                best_moments = tmp_moments

            mious = _iou(best_moments[:max_recall], gt_moment)
            mious = mious[:,None].expand(max_recall, num_iou_metrics)
            bools = mious > self.iou_metrics
            for i, r in enumerate(self.recall_metrics):
                recall_x_iou[i] += bools[:r].any(dim=0)

        recall_x_iou /= len(annotations)

        metrics_dict = {}
        for i, recall in enumerate(self.recall_metrics):
            for j, iou in enumerate(self.iou_metrics):
                metrics_dict[f'R@{recall}-IoU={iou.item():.1f}'] = round(recall_x_iou[i, j].item()*100, 2)

        return metrics_dict
    
    def _validate_dataset(self, outputs: Sequence[TYPE_OUTPUT], dataset_name: Optional[str] = None) -> None:
        # Gather across multiple batches. 
        LOGGER.info("Postprocessing visual predictions.")
        video_embeddings = self._pack_multiple_batches_emebeddings(outputs)
        path = to_absolute_path(os.path.join(self.output_dir, 'video_embeddings.pt'))
        torch.save(video_embeddings, path)

        # Get annotations from dataset
        annotations = self.datamodule.dataset.annos
        frame_rate  = self.datamodule.dataset.frame_rate
        path = to_absolute_path(os.path.join(self.output_dir, 'annotations.pt'))
        torch.save(annotations, path)

        # Compute sentence embeddings
        LOGGER.info("Compute sentence embeddings.")
        text_embeddings = self._compute_text_embeddings(annotations)
        path = to_absolute_path(os.path.join(self.output_dir, 'text_embeddings.pt'))
        torch.save(text_embeddings, path)

        # Compute metrics
        metrics_dict = self._compute_metrics(annotations, text_embeddings, video_embeddings, frame_rate)
        return metrics_dict

    @overrides(check_signature=False)
    def validation_epoch_end(self, outputs: Union[Sequence[TYPE_OUTPUT], Sequence[Sequence[TYPE_OUTPUT]]]) -> None:
        metrics = self._validate_dataset(outputs)

        if wandb and self.enable_wandb:
            wandb.log(metrics)
        
        # Output CSV
        metrics_df = pd.DataFrame(metrics, index=[0])
        metrics_df.to_csv(to_absolute_path(os.path.join(self.output_dir, 'results.csv')), index=False)
        LOGGER.info(f"Results: {metrics}.")
        
    @overrides(check_signature=False)
    def predict_step(self, batch: TYPE_INPUT, batch_idx: int = 0) -> Mapping[str, torch.Tensor]:
        encoded_video, encoded_text = self._step(batch, batch_idx)
        return {
            "encoded_videos": encoded_video,
            "encoded_texts": encoded_text,
            "video_ids": batch["video_id"]
        }

    @overrides
    def load_state_dict(self, state_dict: "OrderedDict[str, torch.Tensor]", strict: bool = True):
        # If it's exactly this class, then ignore any teacher-related thing.
        # We do it here, so we can control it more, and avoid bugs with a general solution.
        if type(self) is TextVideoGroundingLightningModule:
            incompatible_keys = super().load_state_dict(state_dict, strict=False)

            unexpected_keys = set(incompatible_keys.unexpected_keys)
            for key in incompatible_keys.unexpected_keys:
                if key.startswith("teacher"):
                    unexpected_keys.remove(key)

            # We then do as in super:

            if strict:
                error_msgs = []

                if unexpected_keys:
                    unexpected_key_str = ", ".join(f'"{k}"' for k in unexpected_keys)
                    error_msgs.append(f"Unexpected key(s) in state_dict: {unexpected_key_str}. ")
                if incompatible_keys.missing_keys:
                    missing_keys_str = ', '.join(f'"{k}"' for k in incompatible_keys.missing_keys)
                    error_msgs.append(f"Missing key(s) in state_dict: {missing_keys_str}. ")

                if error_msgs:
                    error_msgs_str = "\n\t".join(error_msgs)
                    raise RuntimeError(f"Error(s) in loading state_dict for {self.__class__.__name__}:\n\t"
                                       f"{error_msgs_str}")

            return incompatible_keys
        else:
            return super().load_state_dict(state_dict, strict)