from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


REPO_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = REPO_ROOT / "data" / "pokemon-cards-clean.csv"
ART_DIR = REPO_ROOT / "data" / "art_224"

EMB_OUT = REPO_ROOT / "data" / "card_embeddings.npy"
IDS_OUT = REPO_ROOT / "data" / "card_ids.npy"


BATCH_SIZE = 64   # increase if you have more RAM / GPU


def load_batch(image_paths):
    images = []
    valid_ids = []

    for cid, path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
            valid_ids.append(cid)
        except Exception:
            continue

    return images, valid_ids


def main():
    df = pd.read_csv(CSV_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    all_embeddings = []
    all_ids = []

    # Build list of (id, path)
    items = []
    for _, row in df.iterrows():
        cid = str(row["id"])
        path = ART_DIR / f"{cid}.png"
        if path.exists():
            items.append((cid, path))

    print(f"Processing {len(items)} images...")

    for i in tqdm(range(0, len(items), BATCH_SIZE)):
        batch_items = items[i:i + BATCH_SIZE]

        images, ids = load_batch(batch_items)
        if len(images) == 0:
            continue

        inputs = processor(images=images, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.vision_model(pixel_values=inputs["pixel_values"])
            emb = outputs.pooler_output
            emb = model.visual_projection(emb)

        emb = emb / emb.norm(dim=-1, keepdim=True)

        all_embeddings.append(emb.cpu().numpy())
        all_ids.extend(ids)

    embeddings = np.concatenate(all_embeddings, axis=0)
    card_ids = np.array(all_ids)

    np.save(EMB_OUT, embeddings)
    np.save(IDS_OUT, card_ids)

    print(f"\nSaved embeddings: {EMB_OUT}")
    print(f"Shape: {embeddings.shape}")
    print(f"Saved ids: {IDS_OUT}")
    print(f"Total cards embedded: {len(card_ids)}")


if __name__ == "__main__":
    main()