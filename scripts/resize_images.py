"""Resize every downloaded card image to a fixed square size.

Default: 224x224 (matches CLIP ViT-B/* and most ViT variants). Aspect ratio is
preserved by padding the card onto a black square *before* resizing, which
keeps the whole card visible (cards are portrait ~245x342).

Reads:  data/images/<id>.png
Writes: data/images_<size>/<id>.png   (default: data/images_224/)

Usage:
    python scripts/resize_images.py                 # 224 px, 8 workers
    python scripts/resize_images.py --size 336      # for CLIP ViT-L/14-336
    python scripts/resize_images.py --workers 16
    python scripts/resize_images.py --force         # overwrite existing
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
IN_DIR = REPO_ROOT / "data" / "images"


def resize_one(src: Path, dst: Path, size: int, force: bool) -> tuple[str, str | None]:
    if dst.exists() and not force:
        return src.stem, None
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            padded = ImageOps.pad(im, (size, size), method=Image.Resampling.LANCZOS, color=(0, 0, 0))
            tmp = dst.with_suffix(".png.part")
            padded.save(tmp, format="PNG", optimize=False)
            tmp.replace(dst)
        return src.stem, None
    except (UnidentifiedImageError, OSError) as e:
        return src.stem, f"{type(e).__name__}: {e}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=224, help="Output square size (default 224).")
    parser.add_argument("--workers", type=int, default=8, help="Thread pool size (default 8).")
    parser.add_argument("--force", action="store_true", help="Re-resize even if output exists.")
    args = parser.parse_args()

    if not IN_DIR.exists():
        print(f"ERROR: {IN_DIR} not found. Run scripts/download_images.py first.", file=sys.stderr)
        return 1

    out_dir = REPO_ROOT / "data" / f"images_{args.size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    srcs = sorted(IN_DIR.glob("*.png"))
    print(f"Resizing {len(srcs)} images to {args.size}x{args.size} -> {out_dir} (workers={args.workers})")

    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(resize_one, s, out_dir / s.name, args.size, args.force)
            for s in srcs
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), unit="img"):
            stem, err = fut.result()
            if err is not None:
                failures.append((stem, err))

    print(f"Done. ok={len(srcs) - len(failures)}  failed={len(failures)}")
    if failures:
        for stem, err in failures[:10]:
            print(f"  {stem}: {err}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
