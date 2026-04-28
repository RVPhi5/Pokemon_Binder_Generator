import numpy as np
import torch
from torch import nn


class RankerMLP(nn.Module):
    """Same architecture as scripts/train_ranker.py. Duplicated here so this file is self-contained."""

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


def compute_page_embeddings(card_embeddings):
    """Compute page embedding p from card embeddings z1..zn"""

    if len(card_embeddings) == 0:
        raise ValueError("Page has 0 cards. Add cards to compute page embeddings")
    return np.mean(card_embeddings, axis=0)


def page_score(card_embeddings, model):
    """Computes overall page aesthetic score; averages compatibility of each card with
    the page embedding.

    Expects `model` to take two batched tensors (q, c) of shape (B, emb_dim) and return
    a (B,) tensor of scores. Compatible with RankerMLP from scripts/train_ranker.py.
    """

    p = compute_page_embeddings(card_embeddings)
    p_t = torch.tensor(p, dtype=torch.float32).unsqueeze(0)

    scores = []
    model.eval()
    with torch.no_grad():
        for z in card_embeddings:
            z_t = torch.tensor(z, dtype=torch.float32).unsqueeze(0)
            score = model(p_t, z_t).item()
            scores.append(score)

    return float(np.mean(scores))


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Score a page of n cards using the trained RankerMLP."
    )
    parser.add_argument("cards", nargs="+", help="Card IDs to include on the page.")
    args = parser.parse_args()

    REPO_ROOT = Path(__file__).resolve().parents[1]
    EMB_PATH = REPO_ROOT / "data" / "combined_card_embeddings_norm.npy"
    IDS_PATH = REPO_ROOT / "data" / "combined_card_ids_norm.npy"
    MODEL_PATH = REPO_ROOT / "models" / "ranker.pt"

    embeddings = np.load(EMB_PATH)
    card_ids = np.load(IDS_PATH, allow_pickle=True).astype(str)
    id_to_idx = {cid: i for i, cid in enumerate(card_ids)}

    missing = [c for c in args.cards if c not in id_to_idx]
    if missing:
        raise SystemExit(f"Unknown card ids (no embedding found): {missing}")

    page_emb = np.stack([embeddings[id_to_idx[c]] for c in args.cards])

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model = RankerMLP(checkpoint["emb_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])

    score = page_score(page_emb, model)
    print(f"Page cards: {args.cards}")
    print(f"Page score: {score:.4f}")
