import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim

from dataprocessor import data_generator, load_data, load_test_data
from model import LSTM


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAVE_DIR = BASE_DIR / "save"
SAVE_DIR.mkdir(exist_ok=True)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this machine.")
        return torch.device("cuda")
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available on this machine.")
        return torch.device("mps")
    if device_name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_model(model, data_dict, batch_size, max_len, criterion, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in data_generator(data_dict, batch_size=batch_size, max_len=max_len):
            x_tensor = torch.from_numpy(x).long().transpose(1, 0).contiguous().to(device)
            y_tensor = (
                torch.from_numpy(y).long().transpose(1, 0).contiguous().to(device)
            )
            output = model(x_tensor)
            loss = criterion(output, y_tensor.view(-1))
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


def main():
    parser = argparse.ArgumentParser(description="Train an LSTM couplet model")
    parser.add_argument("--epochs", type=int, default=10, help="training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size")
    parser.add_argument("--embedding-dim", type=int, default=128, help="embedding dim")
    parser.add_argument("--hidden-dim", type=int, default=512, help="hidden dim")
    parser.add_argument("--num-layers", type=int, default=3, help="LSTM layers")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="training device",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    train_dict, vocab_size, idx2word, word2idx, max_len = load_data(
        str(DATA_DIR / "train_in.txt"), str(DATA_DIR / "train_out.txt")
    )
    test_dict = load_test_data(
        str(DATA_DIR / "test_in.txt"), str(DATA_DIR / "test_out.txt"), word2idx
    )

    model = LSTM(
        vocab_size=vocab_size,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_state_dict = None
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        valid_losses = []

        for x, y in data_generator(train_dict, batch_size=args.batch_size, max_len=max_len):
            if len(x) < 2:
                continue

            valid_size = max(int(len(x) * 0.1), 1)
            idxs = np.random.choice(len(x), size=valid_size, replace=False)

            valid_x = torch.from_numpy(x.copy()[idxs]).long().transpose(1, 0).contiguous()
            valid_y = (
                torch.from_numpy(y.copy()[idxs]).long().transpose(1, 0).contiguous()
            )

            train_x = torch.from_numpy(np.delete(x, idxs, 0)).long().transpose(1, 0).contiguous()
            train_y = (
                torch.from_numpy(np.delete(y, idxs, 0)).long().transpose(1, 0).contiguous()
            )

            train_x = train_x.to(device)
            train_y = train_y.to(device)
            valid_x = valid_x.to(device)
            valid_y = valid_y.to(device)

            optimizer.zero_grad()
            output = model(train_x)
            loss = criterion(output, train_y.view(-1))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            model.eval()
            with torch.no_grad():
                valid_output = model(valid_x)
                valid_loss = criterion(valid_output, valid_y.view(-1))
                valid_losses.append(valid_loss.item())
            model.train()

        train_loss_mean = float(np.mean(train_losses)) if train_losses else 0.0
        valid_loss_mean = float(np.mean(valid_losses)) if valid_losses else 0.0
        test_loss = evaluate_model(
            model, test_dict, args.batch_size, max_len, criterion, device
        )

        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Train Loss: {train_loss_mean:.6f}")
        print(f"Valid Loss: {valid_loss_mean:.6f}")
        print(f"Test Loss : {test_loss:.6f}")

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_mean,
                "valid_loss": valid_loss_mean,
                "test_loss": test_loss,
            }
        )

        if valid_loss_mean < best_val_loss:
            best_val_loss = valid_loss_mean
            best_state_dict = {
                "model_state_dict": model.state_dict(),
                "config": {
                    "vocab_size": vocab_size,
                    "embedding_dim": args.embedding_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                },
                "best_val_loss": best_val_loss,
            }

    if best_state_dict is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    torch.save(best_state_dict, SAVE_DIR / "best_model.pt")
    with (SAVE_DIR / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)

    plt.figure(figsize=(8, 5))
    plt.plot([item["epoch"] for item in history], [item["train_loss"] for item in history], label="train_loss")
    plt.plot([item["epoch"] for item in history], [item["valid_loss"] for item in history], label="valid_loss")
    plt.plot([item["epoch"] for item in history], [item["test_loss"] for item in history], label="test_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM Couplet Training Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "loss.png", dpi=200)
    plt.close()

    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoint saved to: {SAVE_DIR / 'best_model.pt'}")


if __name__ == "__main__":
    main()
