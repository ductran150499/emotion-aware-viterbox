"""
Wrapper around Viterbox for Emotion-Aware TTS

This module integrates an external Viterbox instance and
allows injection of learned emotion representations into
the text encoding stage (T3).
"""

import torch
import torch.nn as nn

# import backbone class
from models.backbone.models.backbone.viterbox import viterbox

from models.emotion.emotion_embedding import EmotionEmbedding


class EmotionAwareViterbox(nn.Module):
    def __init__(self, config):
        super().__init__()

        # load vanilla viterbox backbone
        self.backbone = Viterbox.from_pretrained(
            config["backbone_device"]
        )

        # emotion embedding table
        self.emotion_emb = EmotionEmbedding(
            num_emotions=config["num_emotions"],
            d_model=self.backbone.model_dim  # hidden size of T3
        )

        # control freeze/unfreeze
        self.freeze_backbone(config["freeze_backbone"])

    def freeze_backbone(self, freeze=True):
        for p in self.backbone.parameters():
            p.requires_grad = not freeze

    def forward(self, text, emotion_ids=None, **kwargs):
        """
        1. Tokenize text
        2. Add emotion embedding
        3. Pass text + emotion to T3 encoder
        4. Continue with backbone generation
        """
        # TODO: implement emotion injection
        raise NotImplementedError
