import sys, os
sys.path.append(os.getcwd())

import argparse
import torch
from torch.utils.data import DataLoader

from data.dataset import EmotionTTSDataset
from utils.config import load_config
from utils.model import load_viterbox


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        text_tokens = batch["text_tokens"].to(device)
        emotion_ids = batch["emotion_id"].to(device)

        optimizer.zero_grad()
        out = model(text_tokens, emotion_ids)

        # Simple dummy loss (L2)
        loss = out.mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(cfg["device"])
    print(f"🚀 Using device: {device}")

    # Dataset
    dataset = EmotionTTSDataset(cfg["data"])
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)
    print(f"📊 Dataset size: {len(dataset)}")

    # Model
    model = load_viterbox(cfg["model"], device)
    model.to(device)

    # Optimizer (only trainable params)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["optim"]["lr"],
    )

    # Training loop
    for epoch in range(cfg["epochs"]):
        loss = train_one_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch + 1}: loss = {loss:.6f}")


if __name__ == "__main__":
    main()
