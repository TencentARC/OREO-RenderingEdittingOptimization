#!/usr/bin/env python3
"""
Prepare GSO (Google Scanned Objects) Test data set.

Data source:
    Roldbach/google_scanned_objects (HuggingFace)
    - 1030 objects, each object 25 perspectives (Blender rendering,RGBA White background,512×512)
    - Camera pose is random MVS layout (non-fixed azimuth),world-to-camera Matrix storage .npy

Perspective selection strategy (alignment Zero123 Evaluation protocol):
    Roldbach of each object 25 The viewing angles are randomly distributed, so for each target azimuth
    Choose the perspective with the lowest score:score = |elev - target_elev| + 0.5 * |Δazim|(Unit: degree).
    Note:Blender Default"front view"(Numpad 1) camera in -Y axis,atan2(y,x)=-90°.
    Main reference image target:elev=30°, azim=-90°(front).

Output file naming:
    gso_{i:03d}.png              ← Main reference picture (azim ≈ 90°)
    gso_{i:03d}_az{target}.png   ← Additional contrast angle (--extra_azims Specify, default 45 135)
    Example:gso_000.png, gso_000_az045.png, gso_000_az135.png

Function:
1. from HuggingFace Download Roldbach/google_scanned_objects(single zip,~4.7 GB)
2. Unzip to /data/zhiyuan_ma/data/gso_extracted/
3. Print zip internal structure (for --inspect mode)
4. sampling 100 objects, each selects the perspective closest to the azimuth angle of each target (RGBA → white background RGBA)
5. Output to /data/zhiyuan_ma/data/gso_test/
6. Create in code directory dataset/gso_test soft link

Usage:
    # 1. Check the internal structure first (without extracting pictures)
    python scripts/download/prepare_gso_dataset.py --inspect

    # 2. Full run (download + Unzip + Extraction, including default comparison perspective)
    python scripts/download/prepare_gso_dataset.py

    # 3. Already zip Skip download when
    python scripts/download/prepare_gso_dataset.py --local_zip /data/zhiyuan_ma/data/gso/google_scanned_objects.zip

    # 4. Customize additional contrast azimuth (0=side,45,135,180=back)
    python scripts/download/prepare_gso_dataset.py --extra_azims 0 45 135 180
"""

import argparse
import os
import random
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


# =====================================================================
# Configuration
# =====================================================================

HF_REPO_ID = "Roldbach/google_scanned_objects"
HF_FILENAME = "google_scanned_objects.zip"

GSO_DOWNLOAD_DIR = Path("/data/zhiyuan_ma/data/gso")
GSO_EXTRACT_DIR = Path("/data/zhiyuan_ma/data/gso_extracted")
GSO_TEST_DIR = Path("/data/zhiyuan_ma/data/gso_test")

N_OBJECTS = 100
RANDOM_SEED = 42

# Main reference image target perspective
# Blender Default"front view"(Numpad 1) camera in -Y axis,atan2(y,x)=-90°
TARGET_ELEV = 30.0
TARGET_AZIM = -90.0

# Default additional comparison azimuth (±45° deflection, absolute value)
DEFAULT_EXTRA_AZIMS: list[int] = [45, 135]


# =====================================================================
# Utility function
# =====================================================================

def rgba_to_white_rgb(im: Image.Image) -> Image.Image:
    """will RGBA Picture synthesis white background, return RGB."""
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im.convert("RGB"), mask=im.split()[3])
        return bg
    return im.convert("RGB")


def find_object_dirs(extract_root: Path) -> list[Path]:
    """Look for the objects subdirectory in the unzipped directory.

    known Roldbach/google_scanned_objects zip The deep structure of:
        <extract_root>/.../google_scanned_blender_25_w2c/<object>/render_mvs_25/model/000.png

    Strategy:
    1. First look for the containing render_mvs_25 Directory level of subdirectory (object root directory)
    2. If not found, return to general logic (look for layers containing PNG the shallowest layer)
    """
    # --- method 1: Find render_mvs_25 mode(GSO specific structure)---
    # Find all render_mvs_25 Directory, its parent directory is the object directory
    obj_dirs = set()
    for mvs_dir in extract_root.rglob("render_mvs_25"):
        if mvs_dir.is_dir():
            obj_dirs.add(mvs_dir.parent)
    if obj_dirs:
        return sorted(obj_dirs)

    # --- method 2: Universal fallback —— Find contains PNG The shallowest subdirectory of ---
    top_dirs = sorted([d for d in extract_root.iterdir() if d.is_dir()])
    if not top_dirs:
        return [extract_root]

    # If there is an object directory directly under the root (each containing PNG)
    sample_pngs = list(top_dirs[0].rglob("*.png"))[:1]
    if sample_pngs:
        return top_dirs

    # One more level down
    nested = []
    for d in top_dirs:
        nested.extend(sorted([x for x in d.iterdir() if x.is_dir()]))
    return nested if nested else top_dirs


def _parse_w2c(npy_path: Path):
    """from .npy Load world-to-camera (3×4) matrix, return (elevation_deg, azimuth_deg).

    Calculation method:
        The position of the camera in the world coordinate system = -R^T @ t
        elevation = arcsin(z / r)
        azimuth   = atan2(y, x)   (Blender Coordinate system:Z upward,Y forward; front camera on azim≈90°)
    """
    m = np.load(str(npy_path))          # shape (3, 4)
    R, t = m[:3, :3], m[:3, 3]
    cam_pos = -R.T @ t                  # Camera position in world coordinate system
    r = float(np.linalg.norm(cam_pos))
    if r < 1e-6:
        return 0.0, 0.0
    elev = float(np.degrees(np.arcsin(np.clip(cam_pos[2] / r, -1, 1))))
    azim = float(np.degrees(np.arctan2(cam_pos[1], cam_pos[0])))
    return elev, azim


def pick_best_for_azim(
    obj_dir: Path,
    target_azim: float,
    target_elev: float = TARGET_ELEV,
) -> tuple[Optional[Path], float]:
    """Pick the object closest to the specified (elev, azim) Perspective view of the target.

    Roldbach GSO Structure:render_mvs_25/model/{000..024}.png + {000..024}.npy(w2c matrix)

    Rating = |elev - target_elev| + 0.5 * |Δazim|(Unit: degree), take the angle of view corresponding to the minimum value.
    The azimuth angle difference is folded to [-180°, 180°).

    If not .npy camera matrix (not GSO format), return to the smallest numbered PNG,score=inf.

    Return (png_path, best_score);returns when no pictures are found (None, inf).
    """
    model_dir = obj_dir / "render_mvs_25" / "model"

    if model_dir.is_dir():
        npys = sorted(model_dir.glob("*.npy"))
        if npys:
            best_png, best_score = None, float("inf")
            for npy_path in npys:
                png_path = npy_path.with_suffix(".png")
                if not png_path.exists():
                    continue
                try:
                    elev, azim = _parse_w2c(npy_path)
                except Exception:
                    continue
                azim_diff = abs(((azim - target_azim) + 180) % 360 - 180)
                score = abs(elev - target_elev) + 0.5 * azim_diff
                if score < best_score:
                    best_score, best_png = score, png_path
            if best_png is not None:
                return best_png, best_score

        pngs = sorted(model_dir.glob("*.png"))
        if pngs:
            return pngs[0], float("inf")

    # --- Universal fallback (not GSO format)---
    for sub in ["renders", "images", "render", "image"]:
        sub_dir = obj_dir / sub
        if sub_dir.is_dir():
            pngs = sorted(sub_dir.glob("*.png"))
            if pngs:
                return pngs[0], float("inf")

    candidates = sorted(p for p in obj_dir.rglob("*.png"))
    if candidates:
        return candidates[0], float("inf")
    return None, float("inf")


# =====================================================================
# main logic
# =====================================================================

def download_zip(local_zip: Optional[Path]) -> Path:
    """Download or use local zip, return zip path."""
    if local_zip is not None:
        local_zip = Path(local_zip)
        if not local_zip.exists():
            print(f"[ERROR] specified local zip does not exist: {local_zip}", file=sys.stderr)
            sys.exit(1)
        print(f"[INFO] Use local zip: {local_zip}")
        return local_zip

    GSO_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = GSO_DOWNLOAD_DIR / HF_FILENAME

    if target.exists():
        print(f"[INFO] zip Already exists, skip download: {target}")
        return target

    print(f"[INFO] start from HuggingFace Download {HF_REPO_ID}/{HF_FILENAME} ...")
    print("[INFO] approx. 4.7 GB, please wait patiently...")

    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            repo_type="dataset",
            local_dir=str(GSO_DOWNLOAD_DIR),
        )
        print(f"[INFO] Download completed: {downloaded}")
        return Path(downloaded)
    except Exception as e:
        print(f"[ERROR] Download failed: {e}", file=sys.stderr)
        print("[HINT] You can download it manually and use it later --local_zip Parameter specifies path", file=sys.stderr)
        sys.exit(1)


def inspect_zip(zip_path: Path, n: int = 30):
    """Print zip before n An internal path to understand the directory structure."""
    print(f"\n[INSPECT] {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
    print(f"[INSPECT] total {len(names)} items, before {n} Article:")
    for name in names[:n]:
        print(f"  {name}")
    if len(names) > n:
        print(f"  ... (total {len(names)} Article)")


def extract_zip(zip_path: Path, extract_dir: Path):
    """Unzip zip Arrive extract_dir(Skip if already exists)."""
    marker = extract_dir / ".extracted"
    if marker.exists():
        print(f"[INFO] Unzipped, skip: {extract_dir}")
        return

    print(f"[INFO] Unzipping to {extract_dir} ...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(extract_dir))
    marker.touch()
    print("[INFO] Decompression completed")


def _save_rgba(img_path: Path, out_path: Path) -> bool:
    """save RGBA Image (preserve transparency channel). Return on failure False."""
    try:
        im = Image.open(img_path).convert("RGBA")
        im.save(str(out_path))
        return True
    except Exception as e:
        print(f"[WARN] Processing failed {img_path}: {e}")
        return False


def prepare_test_set(
    extract_dir: Path,
    out_dir: Path,
    n: int,
    seed: int,
    extra_azims: list[int] | None = None,
):
    """Sample from unzipped directory n object, extract the reference image and compare the perspective to out_dir.

    Output file:
        gso_{i:03d}.png              ← Main reference picture (azim≈TARGET_AZIM=90°)
        gso_{i:03d}_az{target}.png   ← each extra_azims One for each target azimuth in
    """
    if extra_azims is None:
        extra_azims = DEFAULT_EXTRA_AZIMS

    out_dir.mkdir(parents=True, exist_ok=True)

    obj_dirs = find_object_dirs(extract_dir)
    print(f"[INFO] discover {len(obj_dirs)} object directory")

    if len(obj_dirs) < n:
        print(f"[WARN] Number of objects ({len(obj_dirs)}) Less than requested quantity ({n}), use all objects")
        n = len(obj_dirs)

    random.seed(seed)
    selected = random.sample(obj_dirs, n)
    selected.sort(key=lambda p: p.name)

    # Summarize all azimuths to be saved: main reference map + Additional comparison
    all_azims: list[tuple[float, str]] = [(TARGET_AZIM, "")]  # (targetazim, File name suffix)
    for az in extra_azims:
        all_azims.append((float(az), f"_az{az:03d}"))

    success_main, fail = 0, 0
    for i, obj_dir in enumerate(selected):
        obj_ok = False
        for target_azim, suffix in all_azims:
            img_path, score = pick_best_for_azim(obj_dir, target_azim)
            if img_path is None:
                if suffix == "":
                    print(f"[WARN] No picture found: {obj_dir.name}")
                continue
            out_path = out_dir / f"gso_{i:03d}{suffix}.png"
            ok = _save_rgba(img_path, out_path)
            if suffix == "":
                if ok:
                    success_main += 1
                    obj_ok = True
                else:
                    fail += 1
        if not obj_ok and not any(
            (out_dir / f"gso_{i:03d}.png").exists() for _ in [1]
        ):
            fail += 1

    total_extras = len(extra_azims)
    print(f"[INFO] Extraction completed: main reference picture {success_main} Zhang, failed {fail} Zhang")
    if total_extras:
        print(f"[INFO] Additional contrasting angles: maximum per object {total_extras} a"
              f"(Target azimuth {extra_azims}°)")
    print(f"[INFO] Output directory:{out_dir}")
    return success_main


def create_symlink(target: Path, link_name: Path):
    """Create a soft link in the code directory. If it already exists, skip it."""
    if link_name.exists() or link_name.is_symlink():
        print(f"[INFO] Soft link already exists, skip: {link_name}")
        return
    link_name.parent.mkdir(parents=True, exist_ok=True)
    link_name.symlink_to(target)
    print(f"[INFO] Create soft link: {link_name} -> {target}")


# =====================================================================
# CLI
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare GSO test data set")
    parser.add_argument(
        "--local_zip", type=str, default=None,
        help="local zip Path (skip download)",
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="print only zip Internal structure, do not extract images",
    )
    parser.add_argument(
        "--n", type=int, default=N_OBJECTS,
        help=f"Number of sampled objects (default {N_OBJECTS})",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed (default {RANDOM_SEED})",
    )
    parser.add_argument(
        "--extract_dir", type=str, default=str(GSO_EXTRACT_DIR),
        help=f"Unzip directory (default {GSO_EXTRACT_DIR})",
    )
    parser.add_argument(
        "--out_dir", type=str, default=str(GSO_TEST_DIR),
        help=f"Output test set directory (default {GSO_TEST_DIR})",
    )
    parser.add_argument(
        "--symlink", type=str,
        default=str(Path(__file__).resolve().parents[2] / "dataset" / "gso_test"),
        help="Soft link path created in the code directory",
    )
    parser.add_argument(
        "--extra_azims", type=int, nargs="*", default=DEFAULT_EXTRA_AZIMS,
        metavar="AZ",
        help=(
            f"List of additional saved comparison azimuths (absolute value, unit: degrees, default {DEFAULT_EXTRA_AZIMS})."
            " The main reference drawing is always azim=-90° save as gso_NNN.png,"
            " Comparison chart with gso_NNN_azXXX.png Save."
            " Pass an empty list (--extra_azims) to disable contrasting perspectives."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    zip_path = download_zip(args.local_zip)

    if args.inspect:
        inspect_zip(zip_path)
        return

    # Print the structure first for reference
    inspect_zip(zip_path, n=20)

    extract_dir = Path(args.extract_dir)
    extract_zip(zip_path, extract_dir)

    extra_azims: list[int] = args.extra_azims if args.extra_azims else []
    out_dir = Path(args.out_dir)
    n_success = prepare_test_set(
        extract_dir, out_dir, args.n, args.seed, extra_azims=extra_azims
    )

    if n_success > 0:
        create_symlink(out_dir, Path(args.symlink))
        print(f"\n[DONE] GSO Test set is ready:{out_dir}(Main reference picture {n_success} Zhang)")
        if extra_azims:
            names = ", ".join(f"gso_NNN_az{az:03d}.png" for az in extra_azims)
            print(f"[DONE] Comparative perspective:{names}")
        print(f"[DONE] Soft link:{args.symlink}")
    else:
        print("[ERROR] Failed to retrieve any images, please check zip content", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
