"""Produce a cleaned metadata CSV keyed only on cards whose images we have.

Reads:  data/pokemon-cards.csv, data/images/<id>.png
Writes: data/pokemon-cards-clean.csv

Additions vs. the raw CSV:
    image_path    -- relative path to the downloaded image ("data/images/<safe>.png")
    hp            -- parsed to nullable int (from the raw `hp` column; falls back to caption regex)
    subtype       -- e.g. "Basic", "Stage 1", "Trainer", parsed from `caption`
    card_types    -- "|"-joined list of energy types parsed from `caption`
    rarity        -- e.g. "Rare Holo", parsed from `caption`
Rows whose image file is missing are dropped.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_IN = REPO_ROOT / "data" / "pokemon-cards.csv"
CSV_OUT = REPO_ROOT / "data" / "pokemon-cards-clean.csv"
IMAGES_DIR = REPO_ROOT / "data" / "images"

_WIN_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

_RE_SUBTYPE = re.compile(r"^A ([A-Za-z0-9 +\-]+?) Pokemon Card", re.IGNORECASE)
_RE_TRAINER = re.compile(r"^A ([A-Za-z ]+?) Trainer Card", re.IGNORECASE)
_RE_TYPES = re.compile(r"of type ([A-Za-z, ]+?) with the title", re.IGNORECASE)
_RE_HP_CAP = re.compile(r"and (\d+)\s*HP", re.IGNORECASE)
_RE_RARITY = re.compile(r"of rarity ([A-Za-z0-9 \-'/]+?)(?: evolved from| from the set)", re.IGNORECASE)


def sanitize_id(card_id: str) -> str:
    return _WIN_INVALID.sub("_", card_id)


def parse_caption(caption: str) -> dict[str, object]:
    if not isinstance(caption, str) or not caption.strip():
        return {"subtype": None, "card_types": None, "rarity": None, "hp_from_caption": None}

    subtype = None
    m = _RE_SUBTYPE.match(caption)
    if m:
        subtype = m.group(1).strip()
    else:
        m = _RE_TRAINER.match(caption)
        if m:
            subtype = f"Trainer: {m.group(1).strip()}"

    card_types = None
    m = _RE_TYPES.search(caption)
    if m:
        raw = m.group(1).strip()
        parts = [p.strip() for p in re.split(r",| and ", raw) if p.strip()]
        card_types = "|".join(parts) if parts else None

    rarity = None
    m = _RE_RARITY.search(caption)
    if m:
        rarity = m.group(1).strip()

    hp_from_caption: int | None = None
    m = _RE_HP_CAP.search(caption)
    if m:
        try:
            hp_from_caption = int(m.group(1))
        except ValueError:
            hp_from_caption = None

    return {
        "subtype": subtype,
        "card_types": card_types,
        "rarity": rarity,
        "hp_from_caption": hp_from_caption,
    }


def main() -> int:
    if not CSV_IN.exists():
        print(f"ERROR: raw CSV not found at {CSV_IN}", file=sys.stderr)
        return 1

    df = pd.read_csv(CSV_IN)
    n_raw = len(df)

    df["safe_id"] = df["id"].astype(str).map(sanitize_id)
    df["image_path"] = df["safe_id"].map(lambda s: f"data/images/{s}.png")
    df["has_image"] = df["safe_id"].map(lambda s: (IMAGES_DIR / f"{s}.png").exists())

    n_missing = int((~df["has_image"]).sum())
    df = df[df["has_image"]].copy()

    parsed = df["caption"].map(parse_caption).apply(pd.Series)
    df = pd.concat([df.drop(columns=["has_image"]), parsed], axis=1)

    df["hp"] = pd.to_numeric(df["hp"], errors="coerce")
    df["hp"] = df["hp"].fillna(df["hp_from_caption"])
    df["hp"] = df["hp"].astype("Int64")
    df = df.drop(columns=["hp_from_caption", "safe_id"])

    cols = [
        "id", "name", "hp", "subtype", "card_types", "rarity",
        "set_name", "image_url", "image_path", "caption",
    ]
    df = df[cols]

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUT, index=False)

    print(f"Input rows:     {n_raw}")
    print(f"Missing images: {n_missing} (dropped)")
    print(f"Output rows:    {len(df)}  ->  {CSV_OUT}")
    print()
    print("Coverage (non-null):")
    for col in ["hp", "subtype", "card_types", "rarity"]:
        pct = df[col].notna().mean() * 100
        print(f"  {col:12s} {pct:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
