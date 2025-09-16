# Copyright 2025 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.

import numpy as np
import itertools
from abc import ABC, abstractmethod
from typing import Optional, Sequence

import torch
from overrides import overrides

from aligner.utils.iter_utils import pairwise
from aligner.utils.video_utils import resample


class FrameSampler(ABC):
    """Returns the frame indices to seek for the given clip start and end frame indices."""

    @abstractmethod
    def __call__(self, start_frame: int, end_frame: int, fps: float) -> Sequence[int]:
        raise NotImplementedError


class RandomFromUniformIntervalsFrameSampler(FrameSampler):
    def __init__(self, max_frames: int) -> None:
        super().__init__()
        self.max_frames = max_frames

    @overrides
    def __call__(self, start_frame: int, end_frame: int, fps: float) -> Sequence[int]:
        num_frames = min(self.max_frames, end_frame - start_frame + 1)
        ticks = torch.linspace(start=start_frame, end=end_frame, steps=num_frames + 1, dtype=torch.int)
        return [torch.randint(a, b + 1, size=()) for a, b in pairwise(ticks)]


class UniformFrameSampler(FrameSampler):
    def __init__(self, max_frames: int) -> None:
        super().__init__()
        self.max_frames = max_frames

    @overrides
    def __call__(self, start_frame: int, end_frame: int, fps: float) -> Sequence[int]:
        if self.max_frames == 1:
            index = (end_frame - start_frame) // 2
            return [torch.tensor(index).to(torch.int)]
        else:
            num_frames = min(self.max_frames, end_frame - start_frame + 1)
            ticks = torch.linspace(start=start_frame, end=end_frame, steps=num_frames + 1, dtype=torch.int)
            return [torch.round((a + b) / 2).to(torch.int) for a, b in pairwise(ticks)]
        
class UniformFixedFPSFrameSampler(FrameSampler):
    def __init__(self, max_frames: int, target_video_fps) -> None:
        super().__init__()
        self.max_frames = max_frames
        self.target_video_fps = target_video_fps

    @overrides
    def __call__(self, start_frame: int, end_frame: int, fps: float) -> Sequence[int]:
        target_step_size = int(np.ceil(fps / self.target_video_fps))
        n_frames = end_frame - start_frame
        center_frame_index = start_frame + n_frames // 2  # Adjusted to be within the given range
        
        if target_step_size * (self.max_frames - 1) > n_frames:
            target_step_size = int(np.ceil(n_frames / (self.max_frames - 1)))
            selected_indices = np.linspace(
                                    center_frame_index - (target_step_size * 1.5), 
                                    center_frame_index + (target_step_size * 1.5), 
                                    self.max_frames)
        else:
            selected_indices = []
            if self.max_frames > 1:
                for i in range(self.max_frames // 2):
                    selected_indices.append(int(center_frame_index - (i + 0.5) * target_step_size))
                    selected_indices.append(int(center_frame_index + (i + 0.5) * target_step_size))
            elif self.max_frames == 1:
                selected_indices = [center_frame_index]
            else:
                raise ValueError('Use a number of frames > 0.') 
        
        # Ensure indices are within bounds
        selected_indices = [max(start_frame, min(int(idx), end_frame)) for idx in selected_indices]

        return sorted(selected_indices)


class ConsecutiveFrameSampler(FrameSampler):
    def __init__(self, max_frames: int, fps: Optional[int] = None) -> None:
        super().__init__()
        self.max_frames = max_frames
        self.fps = fps

    @overrides
    def __call__(self, start_frame: int, end_frame: int, fps: float) -> Sequence[int]:
        if self.fps:
            indices = resample(num_frames=self.max_frames, original_fps=fps, new_fps=self.fps)
        else:
            indices = range(self.max_frames)

        smallest_possible_end = min(end_frame, start_frame + indices[-1])

        if isinstance(smallest_possible_end, torch.Tensor):
            smallest_possible_end = smallest_possible_end.item()  # To avoid a warning in the floor division.
        start = start_frame + (end_frame - smallest_possible_end) // 2

        return list(itertools.takewhile(lambda i: i <= end_frame, (start + i for i in indices)))
