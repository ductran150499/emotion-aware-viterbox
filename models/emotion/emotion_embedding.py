"""
Emotion Embedding Module

This module defines a learnable emotion embedding
that maps discrete emotion labels to continuous vectors.

Contribution #1 of the thesis.
"""

import torch
import torch.nn as nn

class EmotionEmbedding(nn.Module):
    def __init__(self, num_emotions: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(num_emotions, d_model)

    def forward(self, emotion_ids):
        """
        Args:
            emotion_ids (LongTensor): shape (B,)
        Returns:
            Tensor: emotion embeddings of shape (B, d_model)
        """
        return self.embedding(emotion_ids)
