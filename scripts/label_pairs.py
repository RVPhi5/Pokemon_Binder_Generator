from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = REPO_ROOT / "data" / "pokemon-cards-clean.csv"
EMB_PATH = REPO_ROOT / "data" / "card_embeddings.npy"
IDS_PATH = REPO_ROOT / "data" / "card_ids.npy"
ART_DIR = REPO_ROOT / "data" / "art_224"
LABELS_PATH = REPO_ROOT / "data" / "labels.csv"


def main() -> int:
    df = pd.read_csv(CSV_PATH)
    df = df.set_index("id")

    embeddings = np.load(EMB_PATH)
    card_ids = np.load(IDS_PATH, allow_pickle=True)

    labels = []
    if LABELS_PATH.exists():
        labels = pd.read_csv(LABELS_PATH).to_dict("records")

    print("Manual labeling mode.")
    print("For each query card, choose the BEST and WORST aesthetic match.")
    print("Type q anytime to quit.\n")

    while True:
        query_i = random.randrange(len(card_ids))
        query_id = str(card_ids[query_i])
        query_emb = embeddings[query_i]

        sims = embeddings @ query_emb
        similar_idxs = np.argsort(sims)[::-1]

        similar_candidates = []
        for idx in similar_idxs:
            cid = str(card_ids[idx])
            if cid != query_id:
                similar_candidates.append(idx)
            if len(similar_candidates) == 3:
                break

        random_candidates = []
        while len(random_candidates) < 2:
            idx = random.randrange(len(card_ids))
            cid = str(card_ids[idx])
            if cid != query_id and idx not in similar_candidates:
                random_candidates.append(idx)

        candidate_idxs = similar_candidates + random_candidates
        random.shuffle(candidate_idxs)

        fig, axes = plt.subplots(1, 6, figsize=(15, 4))

        # Query card
        query_img = Image.open(ART_DIR / f"{query_id}.png")
        axes[0].imshow(query_img)
        axes[0].set_title(f"QUERY\n{df.loc[query_id, 'name']}", fontsize=8)
        axes[0].axis("off")

        # Candidates
        for j, idx in enumerate(candidate_idxs):
            cid = str(card_ids[idx])
            img = Image.open(ART_DIR / f"{cid}.png")
            axes[j + 1].imshow(img)
            axes[j + 1].set_title(f"{j}\n{df.loc[cid, 'name']}", fontsize=8)
            axes[j + 1].axis("off")

        plt.tight_layout()
        plt.show()

        best = input("Best match option (0-4), or q: ").strip()
        if best.lower() == "q":
            break

        worst = input("Worst match option (0-4), or q: ").strip()
        if worst.lower() == "q":
            break

        try:
            best_i = int(best)
            worst_i = int(worst)
            assert 0 <= best_i <= 4
            assert 0 <= worst_i <= 4
            assert best_i != worst_i
        except Exception:
            print("Invalid input. Try again.\n")
            continue

        good_id = str(card_ids[candidate_idxs[best_i]])
        bad_id = str(card_ids[candidate_idxs[worst_i]])

        labels.append({
            "query_card_id": query_id,
            "good_card_id": good_id,
            "bad_card_id": bad_id,
            "candidate_card_ids": "|".join(str(card_ids[i]) for i in candidate_idxs),
        })

        pd.DataFrame(labels).to_csv(LABELS_PATH, index=False)

        print(f"Saved: query={query_id}, good={good_id}, bad={bad_id}")
        print(f"Total labels: {len(labels)}\n")

    pd.DataFrame(labels).to_csv(LABELS_PATH, index=False)
    print(f"Saved labels to {LABELS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())