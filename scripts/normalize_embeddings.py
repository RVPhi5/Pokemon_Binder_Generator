from pathlib import Path
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

INPUT_EMB = REPO_ROOT / "data" / "combined_card_embeddings.npy"
INPUT_IDS = REPO_ROOT / "data" / "combined_card_ids.npy"

OUTPUT_EMB = REPO_ROOT / "data" / "combined_card_embeddings_norm.npy"
OUTPUT_IDS = REPO_ROOT / "data" / "combined_card_ids_norm.npy"


def main():
    z = np.load(INPUT_EMB)
    ids = np.load(INPUT_IDS, allow_pickle=True)

    print(f"Loaded embeddings: {z.shape}")

    # CLIP is 512-dim
    e = z[:, :512]
    m = z[:, 512:]

    print(f"CLIP shape: {e.shape}")
    print(f"Color shape: {m.shape}")

    e = e / np.linalg.norm(e, axis=1, keepdims=True)

    mean = m.mean(axis=0, keepdims=True)
    std = m.std(axis=0, keepdims=True) + 1e-6
    m = (m - mean) / std

    #Optional: scale color features
    alpha = 0.3  # tune this later if need be
    m = alpha * m

    z_norm = np.concatenate([e, m], axis=1)

    z_norm = z_norm / np.linalg.norm(z_norm, axis=1, keepdims=True)

    np.save(OUTPUT_EMB, z_norm)
    np.save(OUTPUT_IDS, ids)

    print(f"Saved normalized embeddings: {OUTPUT_EMB}")
    print(f"Shape: {z_norm.shape}")


if __name__ == "__main__":
    main()