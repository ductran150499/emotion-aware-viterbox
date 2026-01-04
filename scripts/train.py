def main():
    # 1. Load config
    cfg = load_config()

    # 2. Load dataset
    dataset = EmotionTTSDataset(cfg.data)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size)

    # 3. Load backbone
    viterbox = load_viterbox(cfg.model)

    # 4. Wrap model
    model = EmotionAwareViterbox(viterbox, cfg.num_emotions)

    # 5. Freeze logic
    freeze_backbone(model)
    unfreeze_emotion_embedding(model)
    unfreeze_t3_last_blocks(model, cfg.t3_unfreeze_blocks)

    # 6. Optimizer
    optimizer = build_optimizer(model, cfg.optim)

    # 7. Trainer
    trainer = EmotionAwareTrainer(
        model, optimizer, dataloader,
        device=cfg.device,
        lambda_emo=cfg.lambda_emo
    )

    # 8. Train loop
    for epoch in range(cfg.epochs):
        loss = trainer.train_one_epoch()
        print(f"Epoch {epoch}: loss = {loss:.4f}")
