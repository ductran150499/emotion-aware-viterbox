from models.backbone.models.backbone.viterbox_wrapper import EmotionAwareViterbox


def load_viterbox(model_cfg, device):
    return EmotionAwareViterbox(
        device=device,
        num_emotions=model_cfg["num_emotions"],
        t3_unfreeze_blocks=model_cfg.get("t3_unfreeze_blocks", 0),
    )
