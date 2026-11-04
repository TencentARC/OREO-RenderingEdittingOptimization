#!/usr/bin/env python3
"""Prepare Toys4k single-view test set (GSO-aligned PNG layout) via Blender renders."""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# =====================================================================
# Path configuration
# =====================================================================

BLENDER_PATH = "/data/zhiyuan_ma/tools/blender-4.2.0-linux-x64/blender"
BLENDER_SCRIPT = str(
    Path(__file__).parents[2]
    / "_reference_codes/TRELLIS/dataset_toolkits/blender_script/render.py"
)

ZIP_PATH = Path("/data/zhiyuan_ma/code/flow_grpo_custom/dataset/Toys4k/raw/toys4k_blend_files.zip")
EXTRACT_DIR = Path("/data/zhiyuan_ma/code/flow_grpo_custom/dataset/Toys4k/raw/toys4k_blend_files")
OUTPUT_DIR = Path("/data/zhiyuan_ma/data/toys4k_test")
SYMLINK_DIR = Path("/data/zhiyuan_ma/code/flow_grpo_custom/dataset/toys4k_test")

# Camera parameters (alignment TRELLIS training; elevation angle 30°, azimuth angle 0°)
SINGLE_VIEW = [{"yaw": -math.pi / 2, "pitch": math.radians(30), "radius": 2.0, "fov": math.radians(40)}]

RENDER_RESOLUTION = 512

# =====================================================================
# Utility function
# =====================================================================

def extract_zip(zip_path: Path, extract_to: Path) -> None:
    print(f"Unzip {zip_path} → {extract_to} ...")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.endswith(".blend")]
        for member in tqdm(members, desc="Unzip .blend"):
            zf.extract(member, extract_to.parent)
    print(f"Decompression completed, total {len(members)} a .blend File")


def find_blend_files(extract_dir: Path) -> list[Path]:
    """Return the sorted .blend File list."""
    files = sorted(extract_dir.rglob("*.blend"))
    return files


def render_one(args_tuple):
    """Called in child process Blender Render a single image (for multiprocessing use)."""
    blend_file, tmp_dir, output_path, blender_path, blender_script = args_tuple

    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    views_json = json.dumps(SINGLE_VIEW)

    cmd = [
        blender_path,
        str(blend_file),          # Blender to .blend Open for project file
        "-b",
        "-P", blender_script,
        "--",
        "--views", views_json,
        "--object", str(blend_file),   # tells the script this is .blend, skip load_object
        "--resolution", str(RENDER_RESOLUTION),
        "--output_folder", str(tmp_dir),
        "--engine", "CYCLES",
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=300,
    )

    # When rendering is successful,Blender output 000.png
    rendered = tmp_dir / "000.png"
    if rendered.exists():
        shutil.move(str(rendered), str(output_path))
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return str(output_path), None
    else:
        err = result.stderr.decode(errors="replace")[-500:]
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None, f"{blend_file.name}: {err}"


# =====================================================================
# Main process
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare Toys4k Single view test data set")
    parser.add_argument("--zip_path", type=str, default=str(ZIP_PATH))
    parser.add_argument("--extract_dir", type=str, default=str(EXTRACT_DIR))
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--extract_only", action="store_true", help="Only decompress, not render")
    parser.add_argument("--skip_extract", action="store_true", help="Skip decompression (already decompressed)")
    parser.add_argument("--max_objects", type=int, default=None, help="before rendering at most N objects (debugging)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel Blender Number of processes")
    parser.add_argument("--resume", action="store_true", help="Skip existing output files")
    opt = parser.parse_args()

    zip_path = Path(opt.zip_path)
    extract_dir = Path(opt.extract_dir)
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Unzip
    if not opt.skip_extract:
        if not extract_dir.exists() or not any(extract_dir.rglob("*.blend")):
            extract_zip(zip_path, extract_dir)
        else:
            print(f"Unzipped directory found {extract_dir}, skip decompression")
    else:
        print(f"--skip_extract: Skip decompression, use {extract_dir}")

    if opt.extract_only:
        print("--extract_only: Decompression completed, exit")
        return

    # Step 2: collect .blend File
    blend_files = find_blend_files(extract_dir)
    print(f"found {len(blend_files)} a .blend File")

    if opt.max_objects is not None:
        blend_files = blend_files[: opt.max_objects]
        print(f"limit to ex {len(blend_files)} (--max_objects)")

    # Step 3: Build a task list
    tasks = []
    for idx, blend_file in enumerate(blend_files):
        out_name = f"toys4k_{idx:04d}.png"
        out_path = output_dir / out_name
        if opt.resume and out_path.exists():
            continue
        tmp_dir = output_dir / f".tmp_{idx:04d}"
        tasks.append((blend_file, tmp_dir, out_path, BLENDER_PATH, BLENDER_SCRIPT))

    if not tasks:
        print("All files are rendered and no need to be reprocessed (--resume)")
    else:
        print(f"To be rendered:{len(tasks)} (workers={opt.workers})")

    # Step 4: Parallel rendering
    success, errors = 0, []
    with ProcessPoolExecutor(max_workers=opt.workers) as executor:
        futures = {executor.submit(render_one, t): t for t in tasks}
        with tqdm(total=len(tasks), desc="rendering") as pbar:
            for future in as_completed(futures):
                out_path, err = future.result()
                if out_path:
                    success += 1
                else:
                    errors.append(err)
                pbar.update(1)
                pbar.set_postfix(ok=success, fail=len(errors))

    print(f"\nRendering completed: successful {success}, failed {len(errors)}")
    if errors:
        error_log = output_dir / "render_errors.txt"
        with open(error_log, "w") as f:
            f.write("\n".join(errors))
        print(f"For error details, see {error_log}")

    # Step 5: save index CSV
    all_pngs = sorted(output_dir.glob("toys4k_*.png"))
    records = []
    for png in all_pngs:
        idx = int(re.search(r"(\d+)", png.stem).group(1))
        blend_file = blend_files[idx] if idx < len(blend_files) else None
        category = blend_file.parts[-3] if blend_file else ""
        instance = blend_file.stem if blend_file else ""
        records.append({"filename": png.name, "category": category, "instance": instance})
    csv_path = output_dir / "index.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"Index saved to {csv_path}")

    # Step 6: Create soft link
    symlink = SYMLINK_DIR
    if not symlink.exists():
        symlink.symlink_to(output_dir)
        print(f"Soft link:{symlink} → {output_dir}")
    else:
        print(f"Soft link already exists:{symlink}")

    print(f"\nDone! total {len(all_pngs)} images saved in {output_dir}")


if __name__ == "__main__":
    main()
