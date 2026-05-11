import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dataprocessor import data_generator, load_test_data
from model import LSTM


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAVE_DIR = BASE_DIR / "save"


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


def generate_couplet(model, text, word2idx, idx2word, device):
    ids = [word2idx.get(ch, word2idx["UNK"]) for ch in text]
    x = torch.tensor(ids, dtype=torch.long).view(-1, 1).to(device)
    with torch.no_grad():
        output = model(x).argmax(dim=1).cpu().tolist()
    result = []
    for idx in output:
        word = idx2word.get(str(idx), "UNK")
        if word == "UNK":
            word = "春"
        result.append(word)
    return "".join(result)


def main():
    parser = argparse.ArgumentParser(description="Test an LSTM couplet model")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(SAVE_DIR / "best_model.pt"),
        help="checkpoint path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="inference device",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="batch size")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    with (SAVE_DIR / "word2idx.json").open("r", encoding="utf-8") as file:
        word2idx = json.load(file)
    with (SAVE_DIR / "idx2word.json").open("r", encoding="utf-8") as file:
        idx2word = json.load(file)

    checkpoint = torch.load(args.weights, map_location=device)
    config = checkpoint["config"]
    model = LSTM(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_dict = load_test_data(
        str(DATA_DIR / "test_in.txt"), str(DATA_DIR / "test_out.txt"), word2idx
    )
    criterion = nn.CrossEntropyLoss()
    losses = []
    max_len = max(test_dict.keys())

    with torch.no_grad():
        for x, y in data_generator(test_dict, batch_size=args.batch_size, max_len=max_len):
            x_tensor = torch.from_numpy(x).long().transpose(1, 0).contiguous().to(device)
            y_tensor = (
                torch.from_numpy(y).long().transpose(1, 0).contiguous().to(device)
            )
            output = model(x_tensor)
            loss = criterion(output, y_tensor.view(-1))
            losses.append(loss.item())

    print(f"Test Loss: {float(np.mean(losses)):.6f}")
    examples = ["春回大地", "福满人间", "岁岁平安"]
    print("Sample Predictions:")
    for text in examples:
        print(f"上联：{text}")
        print(f"下联：{generate_couplet(model, text, word2idx, idx2word, device)}")


if __name__ == "__main__":
    main()
