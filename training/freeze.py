def freeze_backbone(model):
    for p in model.backbone.parameters():
        p.requires_grad = False

def unfreeze_t3_last_blocks(model, n_blocks=2):
    blocks = model.backbone.t3.transformer.blocks
    for block in blocks[-n_blocks:]:
        for p in block.parameters():
            p.requires_grad = True

def unfreeze_emotion_embedding(model):
    for p in model.emotion_emb.parameters():
        p.requires_grad = True
