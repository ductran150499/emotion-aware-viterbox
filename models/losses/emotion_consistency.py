"""
Emotion Consistency Loss

Encourages generated speech to maintain consistent
emotional characteristics across utterances.
"""

import torch
import torch.nn as nn

class EmotionConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, emotion_repr):
        """
        Args:
            emotion_repr: internal emotional representation
        Returns:
            scalar loss
        """
        # TODO: define consistency constraint
        return torch.mean(emotion_repr ** 2)
