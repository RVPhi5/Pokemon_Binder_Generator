"""Download Pokémon card images referenced by ``data/pokemon-cards.csv``.

Usage (from the repo root):

    python scripts/download_images.py                    # small images, all cards
    python scripts/download_images.py --hires            # hi-res (~15-25 GB)
    python scripts/download_images.py --limit 200        # quick sample
    python scripts/download_images.py --workers 16       # bump concurrency

Images land in ``data/images/<card_id>.png``. Already-downloaded files are
skipped, so the script is safe to re-run / resume.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "data" / "pokemon-cards.csv"
IMAGES_DIR = REPO_ROOT / "data" / "images"

_WIN_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize_id(card_id: str) -> str:
    """Map a card id to a filesystem-safe filename stem (Windows-safe)."""
    return _WIN_INVALID.sub("_", card_id)


def to_small_url(url: str) -> str:
    """Rewrite ``.../1_hires.png`` -> ``.../1.png`` (pokemontcg.io small variant)."""
    return url.replace("_hires.png", ".png")


def load_targets(limit: int | None, hires: bool) -> list[tuple[str, str]]:
    """Return (card_id, primary_url) tuples. ``primary_url`` respects ``--hires``."""
    rows: list[tuple[str, str]] = []
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            card_id = row["id"].strip()
            url = row["image_url"].strip()
            if not card_id or not url:
                continue
            if not hires:
                url = to_small_url(url)
            rows.append((card_id, url))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def download_one(
    card_id: str,
    url: str,
    out_dir: Path,
    session: requests.Session,
) -> tuple[str, str | None]:
    """Return (card_id, error_message_or_None).

    If the primary URL 404s and it's a rewritten small-variant URL, fall back to
    the hi-res URL (some older / promo sets only have ``_hires.png``).
    """
    safe_name = sanitize_id(card_id)
    out_path = out_dir / f"{safe_name}.png"
    if out_path.exists() and out_path.stat().st_size > 0:
        return card_id, None

    candidate_urls = [url]
    if url.endswith(".png") and not url.endswith("_hires.png"):
        candidate_urls.append(url.replace(".png", "_hires.png"))

    last_err: str | None = None
    for candidate in candidate_urls:
        try:
            resp = session.get(candidate, timeout=30)
            if resp.status_code == 404:
                last_err = f"HTTPError: 404 for {candidate}"
                continue
            resp.raise_for_status()
            tmp_path = out_path.with_suffix(".png.part")
            tmp_path.write_bytes(resp.content)
            tmp_path.replace(out_path)
            return card_id, None
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
    return card_id, last_err


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hires", action="store_true", help="Download hi-res PNGs (~15-25 GB instead of ~1 GB).")
    parser.add_argument("--limit", type=int, default=None, help="Only download the first N cards (for quick tests).")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent download workers (default: 8).")
    args = parser.parse_args()

    if not CSV_PATH.exists():
        print(f"ERROR: expected metadata CSV at {CSV_PATH}", file=sys.stderr)
        return 1

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    targets = load_targets(args.limit, args.hires)
    print(f"Downloading {len(targets)} images -> {IMAGES_DIR} "
          f"({'hires' if args.hires else 'small'}, {args.workers} workers)")

    failures: list[tuple[str, str]] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "PokemonBinderGenerator/0.1 (class project)"})

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, cid, url, IMAGES_DIR, session) for cid, url in targets]
        for fut in tqdm(as_completed(futures), total=len(futures), unit="img"):
            card_id, err = fut.result()
            if err is not None:
                failures.append((card_id, err))

    print(f"Done. ok={len(targets) - len(failures)}  failed={len(failures)}")
    if failures:
        fail_log = REPO_ROOT / "data" / "download_failures.txt"
        with fail_log.open("w", encoding="utf-8") as f:
            for card_id, err in failures:
                f.write(f"{card_id}\t{err}\n")
        print(f"Wrote failure list to {fail_log}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
