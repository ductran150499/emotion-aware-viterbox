class EmotionAwareTrainer:
    def __init__(self, model, optimizer, dataloader, device, lambda_emo):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        self.lambda_emo = lambda_emo

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0

        for batch in self.dataloader:
            text = batch["text"]
            emotion_ids = batch["emotion_id"].to(self.device)

            outputs = self.model(
                text_inputs=text,
                emotion_ids=emotion_ids
            )

            tts_loss = outputs["tts_loss"]
            latents = outputs["latents"]

            emo_loss = emotion_consistency_loss(latents, emotion_ids)

            loss = tts_loss + self.lambda_emo * emo_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.dataloader)
