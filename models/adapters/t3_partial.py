"""
Partial Fine-Tuning Utilities for T3 Encoder

This module controls which layers of the TTS backbone
are frozen or trainable during emotion adaptation.
"""

def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze_last_layers(model, n_layers: int):
    """
    Unfreeze the last N layers of the T3 encoder
    """
    for layer in model.layers[-n_layers:]:
        for p in layer.parameters():
            p.requires_grad = True
