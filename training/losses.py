import torch
import torch.nn.functional as F

def emotion_consistency_loss(latents, emotion_ids):
    """
    latents: (B, D)
    emotion_ids: (B,)
    """
    loss = 0.0
    count = 0

    for i in range(len(latents)):
        for j in range(i + 1, len(latents)):
            if emotion_ids[i] == emotion_ids[j]:
                loss += F.mse_loss(latents[i], latents[j])
                count += 1

    if count > 0:
        loss /= count

    return loss
