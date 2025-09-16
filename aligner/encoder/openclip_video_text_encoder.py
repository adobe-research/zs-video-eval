# Copyright 2025 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.

from typing import Iterable, Iterator

import torch
from open_clip.factory import create_model_and_transforms
from open_clip import tokenizer
from open_clip.model import CLIP
from overrides import overrides
from torch import nn
from torchvision import transforms as T

from aligner.data.frame_sampler import FrameSampler, RandomFromUniformIntervalsFrameSampler, UniformFrameSampler, UniformFixedFPSFrameSampler
from aligner.encoder.video_encoder import TYPE_TRANSFORM, float_standard_denormalize
from aligner.encoder.video_text_encoder import TYPE_TEXT_INPUT, TYPE_TOKENIZER, TYPE_VIDEO_INPUT, VideoTextEncoder
from aligner.utils.transforms import ConvertBHWCtoBCHW, RandomResizedCropWithRandomInterpolation


# Necessary to use from Hydra so to get the first element of the tuple from `clip.load`.
# It also does more stuff than `clip.load`.
def load_openclip_model(name: str, *args, **kwargs) -> nn.Module:
    model, _, _ = create_model_and_transforms(
        name,
        pretrained=kwargs['pretrained'],
        precision=kwargs['precision'],
        device=kwargs['device'],
        output_dict=kwargs['output_dict'],
        force_quick_gelu=kwargs['force_quick_gelu'],
        force_output_dim=kwargs['force_output_dim'],
        force_patch_dropout=kwargs['force_patch_dropout'],
        force_inference_patch_dropout=kwargs['force_inference_patch_dropout'],
        inference_patch_dropout_mode=kwargs['inference_patch_dropout_mode'],
        use_residual_token=kwargs['use_residual_feat'],                                            
        residual_token_dim=kwargs['residual_token_dim'],                                           
        residual_token_projection_num_layers=kwargs['residual_token_projection_num_layers'],      
        use_patch_merging=kwargs['use_patch_merging'],
        patch_merging_r=kwargs['patch_merging_r'],
        force_image_size=kwargs['force_image_size'],
    )
    if 'distill_model' in kwargs:
        if kwargs['distill_model'] is not None:
            dist_model, _, _ = create_model_and_transforms(
                kwargs['distill_model'],
                pretrained=kwargs['distill_pretrained'],
                precision=kwargs['precision'],
                device=kwargs['device'],
                output_dict=kwargs['output_dict'],
                force_quick_gelu=kwargs['force_quick_gelu'],
                force_patch_dropout=kwargs['distill_force_patch_dropout'],
                force_inference_patch_dropout=kwargs['distill_force_inference_patch_dropout'],
                inference_patch_dropout_mode=kwargs['distill_inference_patch_dropout_mode'],
                force_image_size=kwargs['distill_force_image_size'],
                )
            return (model, dist_model)
    return model


def _tokenize(texts: Iterable[str]) -> TYPE_TEXT_INPUT:
    return {"input_ids": tokenizer.tokenize(texts)}  # noqa


class OpenClipVideoTextEncoder(VideoTextEncoder):
    def __init__(self, model: CLIP, num_frames: int = 4, remove_avg: bool = True, 
                 eval_sampler = 'constFPS', target_video_fps=1.0, 
                 normalize_first: bool = False, 
                 use_residual_feat = True) -> None: 
        super().__init__()
        if type(model) is tuple:
            self.model = model[0]
            self.dist_model = model[1]
        else:
            self.model = model
            self.dist_model = False
        self.normalize = T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                                     std=(0.26862954, 0.26130258, 0.27577711))
        self.num_frames = num_frames
        self.remove_avg = remove_avg
        self.eval_sampler = eval_sampler
        self.target_video_fps = target_video_fps
        self.normalize_first = normalize_first
        self.use_residual_feat = use_residual_feat

        # Indirectly unregister the param as we don't use it and would otherwise give problems while training.
        if hasattr(self.model, "logit_scale"):
            delattr(self.model, "logit_scale")
        if self.dist_model is not False and hasattr(self.dist_model, "logit_scale"):
            delattr(self.dist_model, "logit_scale")

    @overrides
    def encode_video(self, video: TYPE_VIDEO_INPUT, motion_vectors=None, residuals=None) -> torch.Tensor:
        # video shape: [batch_size, number_samples_per_video, number_channels, height/width, width/height]
        batch_size = video.shape[0]

        if self.dist_model is False:
            images = video.view(-1, *video.shape[2:])
            encoded_video = self.model.encode_image(images, motion_vectors=motion_vectors, 
                                            residuals=residuals, residual_feat=None, normalize=False)[0]
            encoded_video = encoded_video / encoded_video.norm(dim=-1, keepdim=True)

            if self.remove_avg:
                return {"encoded_images": encoded_video.view(batch_size, -1, *encoded_video.shape[1:])}
            return {"encoded_images": encoded_video.view(batch_size, -1, *encoded_video.shape[1:]).mean(dim=1)}

        else:
            B, F, C, H, W = video.shape
            anchor_frame = video[:, 0, :, :, :]
            anchor_motion_vectors = motion_vectors[:, 0, :, :] if motion_vectors is not None else None
            anchor_residuals = residuals[:, 0, :, :] if residuals is not None else None

            encoded_anchor_images, _, anchor_cls_tokens = self.dist_model.encode_image(
                                            anchor_frame, 
                                            motion_vectors=anchor_motion_vectors, 
                                            residuals=anchor_residuals,
                                            residual_feat=None, normalize=False)

            encoded_anchor_images = encoded_anchor_images / encoded_anchor_images.norm(dim=-1, keepdim=True)

            if anchor_cls_tokens is not None:
                anchor_cls_tokens = torch.stack(anchor_cls_tokens)
                anchor_cls_tokens = anchor_cls_tokens.unsqueeze(dim=2)
                anchor_cls_tokens = anchor_cls_tokens.expand(-1, -1, F-1, -1)
                anchor_cls_tokens = anchor_cls_tokens.reshape(anchor_cls_tokens.shape[0], -1, anchor_cls_tokens.shape[-1])

            residual_feat=None
            if self.use_residual_feat:
                num_frames = video.shape[1]
                residual_feat = encoded_anchor_images.repeat_interleave(num_frames-1, dim=0)

            redundant_frames = video[:, 1:, :, :, :].reshape(-1, C, H, W)
            redundant_motion_vectors = motion_vectors[:, 1:, :, :] if motion_vectors is not None else None
            redundant_residuals = residuals[:, 1:, :, :].unsqueeze(1) if residuals is not None else None

            if self.image_size[0] < redundant_frames.shape[-1]:
                redundant_frames = torch.nn.functional.interpolate(
                        redundant_frames, size=self.image_size, 
                        mode='bilinear', align_corners=False
                    )

                if redundant_motion_vectors is not None or redundant_residuals is not None:
                    raise ValueError('We did not handle lower resolution and token dropping.')

            elif self.image_size[0] > redundant_frames.shape[-1]:
                raise ValueError('This case does not interest us.')

            encoded_images, _, _ = self.model.encode_image(
                                        redundant_frames, 
                                        motion_vectors=redundant_motion_vectors, 
                                        residuals=redundant_residuals,
                                        residual_feat=residual_feat, normalize=False)
                    
            encoded_images = encoded_images / encoded_images.norm(dim=-1, keepdim=True)

            encoded_anchor_images = encoded_anchor_images.unsqueeze(dim=1)
            encoded_images = encoded_images.reshape(B, F-1, -1) 
            encoded_images = torch.cat((encoded_anchor_images, encoded_images), 1)

            if self.remove_avg:
                output_dict = {
                    "encoded_anchor_images" : encoded_anchor_images,
                    "encoded_images": encoded_images
                }
                return output_dict
            else:
                output_dict = {
                    "encoded_anchor_images" : encoded_anchor_images,
                    "encoded_images": encoded_images.mean(dim=1)
                }

        
    @overrides
    def encode_text(self, text: TYPE_TEXT_INPUT) -> torch.Tensor:
        encoded_texts = self.model.encode_text(text["input_ids"], normalize=False)
        return encoded_texts / encoded_texts.norm(dim=-1, keepdim=True) 

    @overrides
    def get_tokenizer(self) -> TYPE_TOKENIZER:
        return _tokenize

    @overrides
    def decode_text(self, text: TYPE_TEXT_INPUT) -> Iterator[str]:
        for text_instance in text:
            yield tokenizer._tokenizer.decode(text_instance["input_ids"])

    @overrides
    def get_train_frame_sampler(self) -> FrameSampler:
        return RandomFromUniformIntervalsFrameSampler(self.num_frames)

    @overrides
    def get_eval_frame_sampler(self) -> FrameSampler:
        if self.eval_sampler == 'uniform':
            return UniformFrameSampler(self.num_frames)
        
        elif self.eval_sampler == 'constFPS':
            return UniformFixedFPSFrameSampler(self.num_frames, self.target_video_fps)
        
        else:
            raise ValueError('Select available sampler.')

    @overrides
    def get_train_transform(self, dtype: torch.dtype) -> TYPE_TRANSFORM:
        self.image_size = self.model.visual.image_size
        size = self.image_size

        if self.dist_model is not False:
            self.dist_model_size = self.dist_model.visual.image_size
            if self.dist_model_size > size:
                size = self.dist_model_size

            if self.image_size > self.dist_model_size:
                raise ValueError('This case does not interest us.')

        return T.Compose([
            ConvertBHWCtoBCHW(),
            T.ConvertImageDtype(dtype),
            RandomResizedCropWithRandomInterpolation(size, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            self.normalize,
        ])

    @overrides
    def get_eval_transform(self, dtype: torch.dtype) -> TYPE_TRANSFORM:
        self.image_size = self.model.visual.image_size
        size = self.image_size

        if self.dist_model is not False:
            self.dist_model_size = self.dist_model.visual.image_size
            if self.dist_model_size > size:
                size = self.dist_model_size

            if self.image_size > self.dist_model_size:
                raise ValueError('This case does not interest us.')

        return T.Compose([
            ConvertBHWCtoBCHW(),
            T.ConvertImageDtype(dtype),
            T.Resize(size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(size),
            self.normalize,
        ])

    @property
    @overrides
    def should_pad_batch(self) -> bool:
        return True

    @overrides
    def to_bchw(self, t: torch.Tensor) -> torch.Tensor:
        return t

    @overrides
    def denormalize_video_tensor(self, video: TYPE_VIDEO_INPUT) -> torch.Tensor:
        return float_standard_denormalize(video, mean=self.normalize.mean, std=self.normalize.std)
