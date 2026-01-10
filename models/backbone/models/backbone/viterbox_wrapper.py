import torch
import torch.nn as nn

# Viterbox official class
from models.backbone.models.backbone.viterbox.viterbox.tts import Viterbox


class EmotionAwareViterbox(nn.Module):
    """
    Emotion-aware wrapper on top of pretrained Viterbox.
    Only emotion embedding + optional last T3 blocks are trainable.
    """

    def __init__(self, device="cpu", num_emotions=4, t3_unfreeze_blocks=0):
        super().__init__()

        # ✅ Load pretrained Viterbox from HuggingFace
        self.backbone = Viterbox.from_pretrained(device)
        self.backbone.to(device)

        # Freeze all backbone params
        for p in self.backbone.parameters():
            p.requires_grad = False

        # ===== Emotion embedding =====
        d_model = self.backbone.t3.model_dim
        self.emotion_emb = nn.Embedding(num_emotions, d_model)

        # ===== Optional: unfreeze last T3 blocks =====
        if t3_unfreeze_blocks > 0:
            blocks = self.backbone.t3.blocks
            for block in blocks[-t3_unfreeze_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True

    def forward(self, text_tokens, emotion_ids):
        """
        Forward only until T3 text encoding stage.
        This is enough for emotion embedding fine-tune.
        """

        # Text encoding
        text_emb = self.backbone.t3.embed(text_tokens)

        # Emotion embedding
        emo_emb = self.emotion_emb(emotion_ids).unsqueeze(1)

        # Inject emotion (simple additive)
        text_emb = text_emb + emo_emb

        # Pass through T3 encoder blocks
        for block in self.backbone.t3.blocks:
            text_emb = block(text_emb)

        return text_emb
