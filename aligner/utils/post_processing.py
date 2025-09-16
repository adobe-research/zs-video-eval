# Copyright 2025 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.

import torch
import numpy as np
class SimpleMovingAverage():
    def __init__(self, window_size):
        self.window_size = window_size
    
    def smooth(self, signal):
        if len(signal) <= self.window_size:
            return signal
        # Ensuring the signal is a PyTorch tensor
        if not torch.is_tensor(signal):
            raise TypeError("Expected input is a PyTorch tensor")
        
        half_window = self.window_size // 2
        
        # Calculating initial moving average values
        weights = torch.ones(self.window_size) / self.window_size
        result = torch.conv1d(signal.unsqueeze(0).unsqueeze(0), weights.unsqueeze(0).unsqueeze(0), padding=half_window).squeeze()
        
        # Edge handling
        for i in range(half_window):
            result[i] = torch.mean(signal[:i + half_window + 1])
            result[-i - 1] = torch.mean(signal[-i - half_window - 1:])
        
        return result

class SimpleWatershed():
    def __init__(self, threshold_shift: float = 1.0, watershed_scoring: str = 'max') -> None:
        self.threshold_shift = threshold_shift
        self.scoring_function = self._get_scoring_function(watershed_scoring)
    
    def _get_scoring_function(self, watershed_scoring):
        if watershed_scoring == 'mean':
            return np.mean
        elif watershed_scoring == 'max':
            return np.max
        else:
            raise ValueError(f'Unknown watershed scoring function: {watershed_scoring}')
    
    def __call__(self, scores):
        # Compute gamma as the average of the scores
        gamma = torch.mean(scores)*self.threshold_shift

        # Generate segments as a sequence from 0 to the length of scores
        segments = torch.arange(len(scores))

        merged_segments = []
        merged_scores = []

        current_segment_start = None
        similarity_scores = []

        for segment, score in zip(segments, scores):
            if score >= gamma:
                # If the score is above the threshold, add the segment to the current group
                if current_segment_start is None:
                    current_segment_start = segment.item()
                similarity_scores.append(score.item())
            else:
                if current_segment_start is not None:
                    # If the current group is not empty, add it to the output lists
                    merged_segments.append((current_segment_start, segment.item()))
                    merged_scores.append(self.scoring_function(similarity_scores))
                    # Reset the current group and max score
                    current_segment_start = None
                    similarity_scores = []

        # Add the last group to the output lists if it is not empty
        if current_segment_start is not None:
            merged_segments.append((current_segment_start, segments[-1].item()))
            merged_scores.append(self.scoring_function(similarity_scores))

        # Convert the lists to tensors
        merged_segments_tensor = torch.tensor(merged_segments, dtype=torch.long)
        merged_scores_tensor = torch.tensor(merged_scores)

        # Sort the segments and scores by scores in descending order
        sorted_indices = torch.argsort(merged_scores_tensor, descending=True)
        merged_segments_tensor = merged_segments_tensor[sorted_indices]
        merged_scores_tensor = merged_scores_tensor[sorted_indices]

        return merged_segments_tensor, merged_scores_tensor
       