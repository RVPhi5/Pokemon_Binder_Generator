"""Crop the artwork region from each card (best-effort, template-based).

Pokémon cards have a fairly consistent classic layout (portrait):

             left         right
    top   +------------------+
          |   name / HP      |
    ~9%   +------------------+
          |                  |
          |      ART         |
          |                  |
    ~55%  +------------------+
          |  attacks / etc.  |
          +------------------+
    bot

We crop the bounding box (x: 6%->94%, y: 9%->55% by default), pad to a square,
and resize to the requested size. Modern full-art / textured cards will still
get cropped this way -- they're a known limitation of the template approach
and can be handled later with a segmentation model if desired.

Reads:  data/images/<id>.png
Writes: data/art_<size>/<id>.png  (default: data/art_224/)
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


def crop_one(
    src: Path,
    dst: Path,
    size: int,
    box: tuple[float, float, float, float],
    force: bool,
) -> tuple[str, str | None]:
    if dst.exists() and not force:
        return src.stem, None
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            x0, y0, x1, y1 = box
            crop = im.crop((int(w * x0), int(h * y0), int(w * x1), int(h * y1)))
            padded = ImageOps.pad(crop, (size, size), method=Image.Resampling.LANCZOS, color=(0, 0, 0))
            tmp = dst.with_suffix(".png.part")
            padded.save(tmp, format="PNG", optimize=False)
            tmp.replace(dst)
        return src.stem, None
    except (UnidentifiedImageError, OSError) as e:
        return src.stem, f"{type(e).__name__}: {e}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--x0", type=float, default=0.06, help="Left edge of art region (frac of width).")
    parser.add_argument("--y0", type=float, default=0.09, help="Top edge of art region (frac of height).")
    parser.add_argument("--x1", type=float, default=0.94, help="Right edge of art region (frac of width).")
    parser.add_argument("--y1", type=float, default=0.55, help="Bottom edge of art region (frac of height).")
    args = parser.parse_args()

    if not IN_DIR.exists():
        print(f"ERROR: {IN_DIR} not found. Run scripts/download_images.py first.", file=sys.stderr)
        return 1

    box = (args.x0, args.y0, args.x1, args.y1)
    if not (0 <= args.x0 < args.x1 <= 1 and 0 <= args.y0 < args.y1 <= 1):
        print(f"ERROR: invalid crop box {box}", file=sys.stderr)
        return 1

    out_dir = REPO_ROOT / "data" / f"art_{args.size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    srcs = sorted(IN_DIR.glob("*.png"))
    print(
        f"Cropping art {box} and resizing to {args.size}x{args.size} "
        f"from {len(srcs)} images -> {out_dir} (workers={args.workers})"
    )

    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(crop_one, s, out_dir / s.name, args.size, box, args.force)
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
