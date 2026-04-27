from pathlib import Path 
import numpy as np 
import pandas as pd 

"""Combines CLIP Embeddings and color features z = [e||m]"""

REPO_ROOT= Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT/"data"/"pokemon-cards-clean.csv"
CLIP_EMB_PATH = REPO_ROOT / "data"/"card_embeddings.npy"
CARD_IDS_PATH = REPO_ROOT/"data"/"card_ids.npy"
FEATURES_PATH = REPO_ROOT/"data"/"color_features.csv"

OUTPUT_EMB_PATH = REPO_ROOT/"data"/"combined_card_embeddings.npy"
OUTPUT_IDS_PATH = REPO_ROOT/"data"/"combined_card_ids.npy"

def main():
    df = pd.read_csv(FEATURES_PATH)
    clip_emb = np.load(CLIP_EMB_PATH)
    clip_ids = np.load(CARD_IDS_PATH, allow_pickle=True)

    clip_map = {id:emb for id, emb in zip(clip_ids, clip_emb)}
    color_feature_cols = [cf for cf in df.columns if cf.startswith("cf_")]

    comb_embeddings = []
    comb_ids = []

    for _, row in df.iterrows():
        card_id = row["image_id"]
        if card_id not in clip_map:
            continue
        e = clip_map[card_id]
        m = row[color_feature_cols].values.astype(np.float32)
        z = np.concatenate([e,m])
        
        comb_embeddings.append(z)
        comb_ids.append(card_id)
    comb_embeddings = np.array(comb_embeddings)
    comb_ids = np.array(comb_ids)

    np.save(OUTPUT_EMB_PATH, comb_embeddings)
    np.save(OUTPUT_IDS_PATH, comb_ids)

    print(f"Saved combined embeddings to : {OUTPUT_EMB_PATH}")
    print(f"Shape: {comb_embeddings.shape}")

if __name__ == "__main__":
    raise SystemExit(main())
          

