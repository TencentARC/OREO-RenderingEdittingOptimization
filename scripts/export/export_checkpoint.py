#!/usr/bin/env python
"""Export a training Accelerate checkpoint into a TRELLIS-inference-compatible weight directory."""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export training checkpoint to TRELLIS inference format")
    parser.add_argument("--checkpoint", required=True, help="training checkpoint Directory")
    parser.add_argument("--output", required=True, help="Export target directory")
    parser.add_argument("--pretrained", default="pretrained_weights/TRELLIS-image-large",
                        help="Original pre-trained weight directory (used to complete untrained models)")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint)
    export_dir = Path(args.output)
    pretrained_dir = Path(args.pretrained)

    # ---- Verification ----
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint Directory does not exist: {ckpt_dir}")
    if not (pretrained_dir / "pipeline.json").exists():
        raise FileNotFoundError(f"Pre-training directory is missing pipeline.json: {pretrained_dir}")

    # ---- read pipeline.json Get model name mapping ----
    with open(pretrained_dir / "pipeline.json") as f:
        pipeline_cfg = json.load(f)

    # internal name → TRELLIS ckpt relative path
    # e.g. {"slat_flow_model": "ckpts/slat_flow_img_dit_L_64l8p2_fp16", ...}
    model_mapping = pipeline_cfg["args"]["models"]

    # ---- Prepare export directory ----
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)
    (export_dir / "ckpts").mkdir()

    # Copy pipeline.json
    shutil.copy2(pretrained_dir / "pipeline.json", export_dir / "pipeline.json")
    logger.info(f"[EXPORT] pipeline.json → {export_dir / 'pipeline.json'}")

    # ---- Model-by-model processing ----
    for internal_name, ckpt_rel_path in model_mapping.items():
        # Copy config json(Model structure definition)
        src_json = pretrained_dir / f"{ckpt_rel_path}.json"
        dst_json = export_dir / f"{ckpt_rel_path}.json"
        if src_json.exists():
            shutil.copy2(src_json, dst_json)

        # Check checkpoint Are there training weights for this model?
        # accelerator.save_state() save as model.safetensors(single model)
        # or model_0.safetensors, model_1.safetensors(Multiple models, press prepare Registration order)
        trained_weights = ckpt_dir / "model.safetensors"
        src_weights = pretrained_dir / f"{ckpt_rel_path}.safetensors"
        dst_weights = export_dir / f"{ckpt_rel_path}.safetensors"

        if internal_name == "slat_flow_model" and trained_weights.exists():
            # ✅ Replace with the trained weights
            shutil.copy2(trained_weights, dst_weights)
            logger.info(f"[EXPORT] {internal_name}: ✅ Use training weights ← {trained_weights}")
        elif src_weights.exists():
            # 📋 Use original pre-trained weights
            shutil.copy2(src_weights, dst_weights)
            logger.info(f"[EXPORT] {internal_name}: 📋 Use original weights ← {src_weights}")
        else:
            logger.warning(f"[EXPORT] {internal_name}: ⚠️ No weight file!(checkpoint and pretrained None)")

    # ---- Copy meta.json ----
    meta_src = ckpt_dir / "meta.json"
    if meta_src.exists():
        shutil.copy2(meta_src, export_dir / "meta.json")
        with open(meta_src) as f:
            meta = json.load(f)
        logger.info(f"[EXPORT] Source checkpoint: epoch={meta['epoch']}, step={meta['global_step']}")

    logger.info(f"[EXPORT] ✅ Export completed → {export_dir}")
    logger.info(f"[EXPORT] Usage: TrellisImageTo3DPipeline.from_pretrained('{export_dir}')")


if __name__ == "__main__":
    main()
