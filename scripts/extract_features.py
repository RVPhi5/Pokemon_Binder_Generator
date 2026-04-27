from pathlib import Path 
import numpy as np 
import pandas as pd 
from PIL import Image 
from tqdm import tqdm 
from sklearn.cluster import KMeans
import colorsys 

"""Extracts color features from the cropped out art from the card including 
dominant colors, average rgb + hsv, and std rgb + hsv"""

REPO_ROOT = Path(__file__).resolve().parents[1]
ART_DIR = REPO_ROOT / "data" / "art_224"
OUTPUT_PATH = REPO_ROOT / "data" / "color_features.csv"


def load_image(path:Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32)

def rgb_to_hsv(pixels:np.ndarray)->np.ndarray:

    hsv = [colorsys.rgb_to_hsv(*p) for p in pixels]
    return np.array(hsv)

def extract_features (img: np.ndarray, k=3):

    pixels = img.reshape(-1,3)
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)
    dom_colors = kmeans.cluster_centers_.flatten()


    mean_rgb = pixels.mean(axis=0)
    #gives variation in colors (how simple/noisy it is)
    std_rgb = pixels.std(axis=0)

    hsv = rgb_to_hsv(pixels)
    mean_hsv = hsv.mean(axis=0)
    std_hsv = hsv.mean(axis=0)

    features = np.concatenate([dom_colors, mean_rgb, std_rgb, mean_hsv, std_hsv])

    return features

def main():
    art_paths = sorted(ART_DIR.glob("*.png"))

    print(f"Extracting color features from {len(art_paths)} images")
    rows = [] 

    for path in tqdm(art_paths):
        try: 
            img = load_image(path)
            features = extract_features(img)
            single_row  = {"image_id": path.stem}

            for i, feature_val in enumerate(features):
                single_row[f"cf_{i}"]= feature_val
            rows.append(single_row)
            
        except Exception as e: 
            print(f"Skipping {path.name}: {e}")
            
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved color features to: {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")

if __name__ == "__main__":
    main()



