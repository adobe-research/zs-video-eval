# Copyright 2025 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.

import os
import torch
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

from torchvision.datasets.video_utils import VideoClips

from aligner.utils.typing_utils import TYPE_PATH

# From https://en.wikipedia.org/wiki/Video_file_format
VIDEO_FILE_EXTENSIONS = (".3g2", ".3gp", ".amv", ".asf", ".avi", ".drc", ".f4a", ".f4b", ".f4p", ".f4v", ".flv",
                         ".gif", ".gifv", ".m2ts", ".m2v", ".m4p", ".m4v", ".mkv", ".mng", ".mov", ".mp2", ".mp4",
                         ".mpe", ".mpeg", ".mpg", ".mpv", ".mts", ".mxf", ".nsv", ".ogg", ".ogv", ".qt", ".rm",
                         ".rmvb", ".roq", ".svi", ".ts", ".viv", ".vob", ".webm", ".wmv", ".yuv")


def get_videos_in_folder(path: TYPE_PATH,
                         extensions: Optional[Iterable[str]] = VIDEO_FILE_EXTENSIONS) -> Iterator[str]:
    extensions = None if extensions is None else tuple(extensions)
    for folder, _, filenames in os.walk(path, followlinks=True):
        for filename in filenames:
            if os.path.isfile(full_path := os.path.join(folder, filename)) \
                    and (not extensions or filename.lower().endswith(extensions)):
                yield full_path


def get_sorted_videos_in_folder(path: TYPE_PATH,
                                extensions: Optional[Iterable[str]] = VIDEO_FILE_EXTENSIONS,
                                key: Optional[Callable[[str], Any]] = None, reverse: bool = False) -> Iterator[str]:
    """Returns a sorted version of `get_videos_in_folder`.

    Even though this can be simply applied by the caller, the fact that the main use case of `get_videos_in_folder`
    is from a video dataset and that its order should be deterministic (but that `get_videos_in_folder` doesn't
    guarantee it) makes this function handy and a wake-up call for this issue.

    The videos in a PyTorch `Dataset` need to be deterministic e.g. for a distributed setting, when e.g. using
    `DistributedSampler` for it to guarantee each data sample is used once and only once between all processes.
    """
    return sorted(get_videos_in_folder(path, extensions), key=key, reverse=reverse)


def resample(num_frames: int, original_fps: float, new_fps: float) -> Sequence[int]:
    """Returns essentially the same as `VideoClips._resample_video_idx`. Unlike it, it always checks for the max frames
    (the mentioned function doesn't do it when it returns a `slice`)."""
    indices = VideoClips._resample_video_idx(num_frames, original_fps, new_fps)

    if isinstance(indices, slice) and indices.stop is None:
        indices = range(*indices.indices((indices.start or 0) + num_frames * indices.step))

    return indices


def _iou(candidates, gt):
    start, end = candidates[:,0].float(), candidates[:,1].float()
    s, e = gt[0].float(), gt[1].float()
    inter = end.min(e) - start.max(s)
    union = end.max(e) - start.min(s)
    return inter.clamp(min=0) / union


class NormalizeVideo:
    """
    Normalize the video clip by mean subtraction and division by standard deviation
    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def _is_tensor_video_clip(self, clip):
        if not torch.is_tensor(clip):
            raise TypeError("clip should be Tensor. Got %s" % type(clip))

        if not clip.ndimension() == 4:
            raise ValueError("clip should be 4D. Got %dD" % clip.dim())

        return True

    def normalize(self, clip, mean, std, inplace=False):
        """
        Args:
            clip (torch.tensor): Video clip to be normalized. Size is (C, T, H, W)
            mean (tuple): pixel RGB mean. Size is (3)
            std (tuple): pixel standard deviation. Size is (3)
        Returns:
            normalized clip (torch.tensor): Size is (C, T, H, W)
        """
        
        if not self._is_tensor_video_clip(clip):
            raise ValueError("clip should be a 4D torch.tensor")
        if not inplace:
            clip = clip.clone()
        mean = torch.as_tensor(mean, dtype=clip.dtype, device=clip.device)
        std = torch.as_tensor(std, dtype=clip.dtype, device=clip.device)
        clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
        return clip

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): video clip to be normalized. Size is (C, T, H, W)
        """
        return self.normalize(clip, self.mean, self.std, self.inplace)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, inplace={self.inplace})"