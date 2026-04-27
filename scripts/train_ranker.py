from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split


REPO_ROOT = Path(__file__).resolve().parents[1]

EMB_PATH = REPO_ROOT / "data" / "combined_card_embeddings_norm.npy"
IDS_PATH = REPO_ROOT / "data" / "combined_card_ids_norm.npy"
PAIRS_PATH = REPO_ROOT / "data" / "ranking_pairs.csv"

MODEL_DIR = REPO_ROOT / "models"
PLOTS_DIR = REPO_ROOT / "outputs" / "training_plots"

MODEL_OUT = MODEL_DIR / "ranker.pt"
METRICS_OUT = MODEL_DIR / "training_metrics.csv"


class RankingDataset(Dataset):
    def __init__(self, pairs, id_to_emb):
        self.rows = pairs.to_dict("records")
        self.id_to_emb = id_to_emb

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        q = torch.tensor(self.id_to_emb[row["query_card_id"]], dtype=torch.float32)
        good = torch.tensor(self.id_to_emb[row["preferred_card_id"]], dtype=torch.float32)
        bad = torch.tensor(self.id_to_emb[row["worse_card_id"]], dtype=torch.float32)

        return q, good, bad


class RankerMLP(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        input_dim = emb_dim * 4

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 1),
        )

    def make_features(self, q, c):
        return torch.cat([q, c, torch.abs(q - c), q * c], dim=1)

    def forward(self, q, c):
        x = self.make_features(q, c)
        return self.net(x).squeeze(1)


def ranking_loss(good_score, bad_score):
    return -torch.log(torch.sigmoid(good_score - bad_score) + 1e-8).mean()


def evaluate(model, loader, device):
    model.eval()
    losses = []
    correct = 0
    total = 0
    margins = []

    with torch.no_grad():
        for q, good, bad in loader:
            q = q.to(device)
            good = good.to(device)
            bad = bad.to(device)

            good_score = model(q, good)
            bad_score = model(q, bad)

            loss = ranking_loss(good_score, bad_score)
            losses.append(loss.item())

            margin = good_score - bad_score
            margins.extend(margin.cpu().numpy().tolist())

            correct += (margin > 0).sum().item()
            total += len(margin)

    return {
        "loss": float(np.mean(losses)),
        "accuracy": correct / total if total else 0.0,
        "margin_mean": float(np.mean(margins)),
    }


def save_training_plots(metrics_df):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Loss plot
    plt.figure()
    plt.plot(metrics_df["epoch"], metrics_df["train_loss"], label="train loss")
    plt.plot(metrics_df["epoch"], metrics_df["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Ranking loss")
    plt.title("Training vs Validation Ranking Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "loss_curve.png", dpi=200)
    plt.close()

    # Accuracy plot
    plt.figure()
    plt.plot(metrics_df["epoch"], metrics_df["train_accuracy"], label="train accuracy")
    plt.plot(metrics_df["epoch"], metrics_df["val_accuracy"], label="val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Pairwise accuracy")
    plt.title("Training vs Validation Pairwise Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_curve.png", dpi=200)
    plt.close()

    # Margin plot
    plt.figure()
    plt.plot(metrics_df["epoch"], metrics_df["train_margin_mean"], label="train margin")
    plt.plot(metrics_df["epoch"], metrics_df["val_margin_mean"], label="val margin")
    plt.xlabel("Epoch")
    plt.ylabel("Mean score margin")
    plt.title("Preferred Score - Worse Score Margin")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "margin_curve.png", dpi=200)
    plt.close()


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(EMB_PATH)
    card_ids = np.load(IDS_PATH, allow_pickle=True).astype(str)
    pairs = pd.read_csv(PAIRS_PATH)

    id_to_emb = {cid: emb for cid, emb in zip(card_ids, embeddings)}

    # Keep only rows where all ids exist in embeddings.
    pairs = pairs[
        pairs["query_card_id"].astype(str).isin(id_to_emb)
        & pairs["preferred_card_id"].astype(str).isin(id_to_emb)
        & pairs["worse_card_id"].astype(str).isin(id_to_emb)
    ].copy()

    print(f"Training pairs after filtering: {len(pairs)}")

    dataset = RankingDataset(pairs, id_to_emb)

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    emb_dim = embeddings.shape[1]
    model = RankerMLP(emb_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 50
    metrics = []

    best_val_acc = -1
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for q, good, bad in train_loader:
            q = q.to(device)
            good = good.to(device)
            bad = bad.to(device)

            good_score = model(q, good)
            bad_score = model(q, bad)

            loss = ranking_loss(good_score, bad_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_margin_mean": train_metrics["margin_mean"],
            "val_margin_mean": val_metrics["margin_mean"],
        }
        metrics.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={row['train_loss']:.4f}, acc={row['train_accuracy']:.3f} | "
            f"val loss={row['val_loss']:.4f}, acc={row['val_accuracy']:.3f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = {
                "model_state_dict": model.state_dict(),
                "emb_dim": emb_dim,
                "input_dim": emb_dim * 4,
                "best_val_accuracy": best_val_acc,
            }

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(METRICS_OUT, index=False)
    save_training_plots(metrics_df)

    torch.save(best_state, MODEL_OUT)

    print()
    print(f"Saved model to: {MODEL_OUT}")
    print(f"Saved metrics to: {METRICS_OUT}")
    print(f"Saved plots to: {PLOTS_DIR}")
    print(f"Best validation accuracy: {best_val_acc:.3f}")


if __name__ == "__main__":
    main()