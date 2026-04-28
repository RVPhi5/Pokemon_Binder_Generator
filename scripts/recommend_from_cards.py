from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]

EMB_PATH = REPO_ROOT / "data" / "combined_card_embeddings_norm.npy"
IDS_PATH = REPO_ROOT / "data" / "combined_card_ids_norm.npy"
CSV_PATH = REPO_ROOT / "data" / "pokemon-cards-clean.csv"
MODEL_PATH = REPO_ROOT / "models" / "ranker.pt"

ART_DIR = REPO_ROOT / "data" / "art_224"
OUT_DIR = REPO_ROOT / "outputs" / "page_recommendations"


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
        return self.net(self.make_features(q, c)).squeeze(1)


def plot_page(query_ids, rec_ids, scores, df, out_path):
    all_ids = query_ids + rec_ids
    total = len(all_ids)

    cols = 3
    rows = int(np.ceil(total / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(9, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for i, cid in enumerate(all_ids):
        img = Image.open(ART_DIR / f"{cid}.png")
        axes[i].imshow(img)

        name = df.loc[cid, "name"] if cid in df.index else cid

        if cid in query_ids:
            title = f"USER\n{name}"
        else:
            rank = rec_ids.index(cid) + 1
            score = scores[rank - 1]
            title = f"REC #{rank}\n{name}\nscore={score:.3f}"

        axes[i].set_title(title, fontsize=8)
        axes[i].axis("off")

        # Red border for user-provided cards
        if cid in query_ids:
            for spine in axes[i].spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(4)
                spine.set_visible(True)

    for j in range(total, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queries",
        nargs="+",
        required=True,
        help="One or more card IDs, e.g. --queries sm9-18 xy2-14 swsh3-5",
    )
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    df["id"] = df["id"].astype(str)
    df = df.set_index("id")

    embeddings = np.load(EMB_PATH)
    card_ids = np.load(IDS_PATH, allow_pickle=True).astype(str)

    id_to_idx = {cid: i for i, cid in enumerate(card_ids)}

    query_ids = [str(q) for q in args.queries]

    for q in query_ids:
        if q not in id_to_idx:
            raise ValueError(f"Query card {q} not found in embeddings.")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    emb_dim = checkpoint["emb_dim"]

    model = RankerMLP(emb_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Page embedding = average of query card embeddings
    query_indices = [id_to_idx[q] for q in query_ids]
    page_emb = embeddings[query_indices].mean(axis=0)
    page_emb = page_emb / np.linalg.norm(page_emb)

    # Score all candidates except already-selected query cards
    candidate_indices = np.array([
        i for i, cid in enumerate(card_ids)
        if cid not in query_ids
    ])

    q_batch = torch.tensor(
        np.repeat(page_emb[None, :], len(candidate_indices), axis=0),
        dtype=torch.float32,
    )
    c_batch = torch.tensor(embeddings[candidate_indices], dtype=torch.float32)

    with torch.no_grad():
        scores = model(q_batch, c_batch).numpy()

    top_order = np.argsort(scores)[::-1][:args.k]
    top_indices = candidate_indices[top_order]

    rec_ids = [str(card_ids[i]) for i in top_indices]
    rec_scores = [float(scores[i]) for i in top_order]

    print("\nUser-provided cards:")
    for cid in query_ids:
        print(f"- {cid} | {df.loc[cid, 'name']}")

    print("\nRecommendations:")
    for rank, (cid, score) in enumerate(zip(rec_ids, rec_scores), start=1):
        print(f"{rank}. {cid} | {df.loc[cid, 'name']} | score={score:.4f}")

    safe_name = "_".join(query_ids).replace("/", "_")
    out_path = OUT_DIR / f"page_recommendations_{safe_name}.png"

    plot_page(query_ids, rec_ids, rec_scores, df, out_path)

    print(f"\nSaved visualization to: {out_path}")


if __name__ == "__main__":
    main()