from __future__ import annotations

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


def main() -> int:
    df = pd.read_csv(CSV_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    embeddings = []
    card_ids = []

    for _, row in tqdm(df.iterrows(), total=len(df), unit="card"):
        card_id = str(row["id"])
        image_path = ART_DIR / f"{card_id}.png"

        if not image_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.vision_model(pixel_values=inputs["pixel_values"])
            emb = outputs.pooler_output
            emb = model.visual_projection(emb)

        # normalize embedding
        emb = emb / emb.norm(dim=-1, keepdim=True)

        embeddings.append(emb.cpu().numpy()[0])
        card_ids.append(card_id)

    embeddings = np.array(embeddings, dtype=np.float32)
    card_ids = np.array(card_ids)

    np.save(EMB_OUT, embeddings)
    np.save(IDS_OUT, card_ids)

    print(f"Saved embeddings: {EMB_OUT} shape={embeddings.shape}")
    print(f"Saved ids:        {IDS_OUT} shape={card_ids.shape}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())