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
OUT_DIR = REPO_ROOT / "outputs" / "recommendations"


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


def find_card_id(query, df, card_ids):
    query = str(query)

    # Direct ID match
    if query in set(card_ids):
        return query

    # Name search
    matches = df[df["name"].str.contains(query, case=False, na=False)]

    if len(matches) == 0:
        raise ValueError(f"No card found matching: {query}")

    print("\nMatches:")
    for i, (_, row) in enumerate(matches.iterrows()):
        print(f"{i}: {row['id']} | {row['name']} | {row.get('set_name', '')}")

    chosen = input("\nChoose match number: ").strip()
    chosen_i = int(chosen)

    return str(matches.iloc[chosen_i]["id"])


def plot_recommendations(query_id, rec_ids, scores, df, out_path):
    n = len(rec_ids)
    cols = min(5, n + 1)
    rows = int(np.ceil((n + 1) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    query_img = Image.open(ART_DIR / f"{query_id}.png")
    axes[0].imshow(query_img)
    q_name = df.loc[query_id, "name"] if query_id in df.index else query_id
    axes[0].set_title(f"QUERY\n{q_name}", fontsize=9)
    axes[0].axis("off")

    for i, (cid, score) in enumerate(zip(rec_ids, scores), start=1):
        img = Image.open(ART_DIR / f"{cid}.png")
        axes[i].imshow(img)
        name = df.loc[cid, "name"] if cid in df.index else cid
        axes[i].set_title(f"#{i}\n{name}\nscore={score:.3f}", fontsize=8)
        axes[i].axis("off")

    for j in range(n + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        required=True,
        help="Card id or partial card name, e.g. 'sm9-18' or 'Gengar'",
    )
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument(
        "--top_pool",
        type=int,
        default=None,
        help="Optional: only rerank top N CLIP/color similar cards. Default uses all cards.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    df["id"] = df["id"].astype(str)
    df = df.set_index("id")

    embeddings = np.load(EMB_PATH)
    card_ids = np.load(IDS_PATH, allow_pickle=True).astype(str)

    id_to_idx = {cid: i for i, cid in enumerate(card_ids)}

    query_id = find_card_id(args.query, df.reset_index(), card_ids)

    if query_id not in id_to_idx:
        raise ValueError(f"Query id {query_id} does not have an embedding.")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    emb_dim = checkpoint["emb_dim"]

    model = RankerMLP(emb_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    query_idx = id_to_idx[query_id]
    q = embeddings[query_idx]

    candidate_indices = np.arange(len(card_ids))

    candidate_indices = candidate_indices[candidate_indices != query_idx]

    if args.top_pool is not None:
        sims = embeddings @ q
        sorted_idxs = np.argsort(sims)[::-1]
        sorted_idxs = [i for i in sorted_idxs if i != query_idx]
        candidate_indices = np.array(sorted_idxs[: args.top_pool])

    q_batch = torch.tensor(
        np.repeat(q[None, :], len(candidate_indices), axis=0),
        dtype=torch.float32,
    )
    c_batch = torch.tensor(embeddings[candidate_indices], dtype=torch.float32)

    with torch.no_grad():
        scores = model(q_batch, c_batch).numpy()

    top_order = np.argsort(scores)[::-1][: args.k]
    top_indices = candidate_indices[top_order]

    rec_ids = [str(card_ids[i]) for i in top_indices]
    rec_scores = [float(scores[i]) for i in top_order]

    print(f"\nQuery: {query_id} | {df.loc[query_id, 'name']}")
    print("\nRecommendations:")
    for rank, (cid, score) in enumerate(zip(rec_ids, rec_scores), start=1):
        name = df.loc[cid, "name"] if cid in df.index else cid
        set_name = df.loc[cid, "set_name"] if cid in df.index else ""
        print(f"{rank}. {cid} | {name} | {set_name} | score={score:.4f}")

    safe_query = query_id.replace("/", "_")
    out_path = OUT_DIR / f"recommendations_{safe_query}.png"

    plot_recommendations(query_id, rec_ids, rec_scores, df, out_path)

    print(f"\nSaved visualization to: {out_path}")


if __name__ == "__main__":
    main()