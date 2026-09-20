"""
Dense FlowEdit Evaluation script.

Comparing the rendering results of the two passes CLIP / DINO Similarity:
  Teacher (pretrained): Dense ODE → decode_to_coords → Sparse ODE
  Student (finetuned) : Dense ODE → Dense FlowEdit → decode_to_coords → Sparse ODE

with universal teacher/student Evaluation script differences:
  - student The path is in Stage 1 additional execution Dense FlowEdit
  - FlowEdit parameters passed --config.rollout.flowedit.* Override (or use default)
  - Not dependent on new config files, directly to existing config Inject flowedit subfield

Usage (single card):
  python scripts/eval/eval_trellis_dense_flowedit.py \
    --config=config/trellis_oreo.py \
    --config.checkpoint=/path/to/ckpt \
    --config.rollout.flowedit.n_max=9 \
    --config.rollout.flowedit.cfg_scale_tgt=3.0 \
    --config.rollout.flowedit.cfg_scale_src=-3.0

Usage (DDP Multi-card):
  accelerate launch scripts/eval/eval_trellis_dense_flowedit.py \
    --config=config/trellis_oreo.py ...
"""

import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

repo_root = os.path.abspath(os.getcwd())
trellis_ref_root = os.path.join(repo_root, "_reference_codes", "TRELLIS")
if trellis_ref_root not in sys.path:
    sys.path.insert(0, trellis_ref_root)
triposf_ref_root = os.path.join(repo_root, "_reference_codes", "TripoSF")
if triposf_ref_root not in sys.path:
    sys.path.insert(0, triposf_ref_root)

import numpy as np
import torch
from contextlib import nullcontext
from PIL import Image
from tqdm import tqdm
from absl import app
from ml_collections import config_flags
import ml_collections
from accelerate import Accelerator

from oreo.systems.trellis.system import (
    build_system, trellis_forward, _CONFIG,
)
from oreo.systems.base import setup_env_and_seed, EvalModeGuard
from oreo.datasets.trellis import (
    TrellisCameraTrainConfig,
    TrellisCameraEvalConfig,
    TrellisDataConfig,
    TrellisDataModule,
)
from oreo.generators.trellis.state import TrellisState
from oreo.generators.trellis.rollout import rollout_sparse, rollout_dense
from oreo.generators.trellis.rollout.flowedit import rollout_dense_flowedit
from oreo.systems.trellis.forward import decode_and_render_gs, decode_and_render_mesh
from oreo.systems.trellis.checkpoint import (
    load_full_dual_checkpoint,
    validate_student_diff,
)
from oreo.guidance import create_guidance
from oreo.systems.utils import composite_alpha_to_white
from oreo.guidance.metric.clip import CLIPMetric
from oreo.guidance.metric.dino import DINOMetric

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =====================================================================
# FlowEdit Default parameters (available via --config.rollout.flowedit.* coverage)
# =====================================================================

_FE_DEFAULTS = {
    "steps": 12,
    "n_max": 9,
    "cfg_scale_tgt": 3.0,
    "cfg_scale_src": -3.0,
}


def _ensure_flowedit_cfg(cfg: ml_collections.ConfigDict) -> None:
    """If cfg.rollout.flowedit If it does not exist, a default value will be injected.

    ml_collections ConfigDict in absl is locked after parsing, must first unlock Add new key.
    """
    if not hasattr(cfg.rollout, "flowedit"):
        fe = ml_collections.ConfigDict({
            "steps": _FE_DEFAULTS["steps"],
            "n_max": _FE_DEFAULTS["n_max"],
            "cfg_scale_tgt": _FE_DEFAULTS["cfg_scale_tgt"],
            "cfg_scale_src": _FE_DEFAULTS["cfg_scale_src"],
        })
        with cfg.rollout.unlocked():
            cfg.rollout.flowedit = fe


# =====================================================================
# Utility functions (and general teacher/student assessment remains consistent)
# =====================================================================

def build_eval_dataloader(cfg, accelerator: Accelerator):
    train_cam_cfg = TrellisCameraTrainConfig(
        n_view=cfg.data.train.n_view,
        yaw_range=list(cfg.data.train.yaw_range),
        pitch_range=list(cfg.data.train.pitch_range),
        r_range=list(cfg.data.train.r_range),
        fov_range=list(cfg.data.train.fov_range),
        adaptive_distance=cfg.data.train.adaptive_distance,
    )
    eval_cam_cfg = TrellisCameraEvalConfig(
        n_view=cfg.data.eval.n_view,
        yaw_range=list(cfg.data.eval.yaw_range),
        pitch_range=list(cfg.data.eval.pitch_range),
        r_range=list(cfg.data.eval.r_range),
        fov_range=list(cfg.data.eval.fov_range),
        adaptive_distance=cfg.data.eval.adaptive_distance,
    )
    dm_cfg = TrellisDataConfig(
        batch_size=cfg.data.train.batch_size,
        eval_batch_size=cfg.data.eval.batch_size,
        width=cfg.renderer.resolution,
        height=cfg.renderer.resolution,
        image_dataset_dir=cfg.data.eval.dir,
        eval_image_path=cfg.data.eval.dir,
        train=train_cam_cfg,
        eval=eval_cam_cfg,
    )
    dm = TrellisDataModule(
        dm_cfg,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
    )
    dm.setup(stage="test")
    return dm.eval_dataloader()


def _to_pil(t: torch.Tensor) -> Image.Image:
    if t.dim() == 4:
        t = t.squeeze(0)
    if t.dim() == 3 and t.shape[0] in (1, 3, 4):
        t = t.permute(1, 2, 0)
    return Image.fromarray(
        (t.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    )


def _similarity(metric, rendered_pil: Image.Image, target_pil: Image.Image) -> float:
    with torch.no_grad():
        loss = metric.compute_from_pil([rendered_pil], [target_pil])
    return 1.0 - loss.item()


def _save_images(
    images_dir: Path,
    name: str,
    cond_pil: Image.Image,
    stu_pil: Image.Image,
    tea_pil: Image.Image,
    v: int,
) -> None:
    d = images_dir / name
    d.mkdir(parents=True, exist_ok=True)
    if v == 0:
        cond_pil.save(d / "condition.png")
    stu_pil.save(d / f"v{v}_student.png")
    tea_pil.save(d / f"v{v}_teacher.png")
    margin = 12
    h = stu_pil.height
    c = cond_pil.copy()
    if c.height != h:
        s = h / c.height
        c = c.resize((max(1, int(c.width * s)), h), Image.LANCZOS)
    imgs = [c, tea_pil, stu_pil]
    total_w = sum(im.width for im in imgs) + margin * (len(imgs) + 1)
    grid = Image.new("RGB", (total_w, h + margin * 2), (255, 255, 255))
    x = margin
    for im in imgs:
        grid.paste(im, (x, margin))
        x += im.width + margin
    grid.save(d / f"v{v}_grid.png")


def _load_existing_eval_rows(
    csv_path: Path,
    fields: List[str],
    expected_views: int,
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Read existing CSV, returns the written rows and view Count the complete sample name."""
    if not csv_path.exists():
        return [], set()

    rows_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    views_by_name: Dict[str, Set[str]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name", "")
            view = row.get("view", "")
            if not name or name == "AVERAGE" or view in ("", "-"):
                continue

            clean_row = {field: row.get(field, "") for field in fields}
            key = (name, str(view))
            rows_by_key[key] = clean_row
            views_by_name.setdefault(name, set()).add(str(view))

    completed = {
        name
        for name, views in views_by_name.items()
        if len(views) >= expected_views
    }
    return list(rows_by_key.values()), completed


# =====================================================================
# EvalMetricLogger(with universal teacher/student assessment remains consistent)
# =====================================================================

class EvalMetricLogger:
    def __init__(
        self,
        out_dir: Path,
        keys: List[str],
        accelerator: Accelerator,
        existing_rows: Optional[List[Dict[str, Any]]] = None,
    ):
        self.out_dir = out_dir
        self.keys = keys
        self.fields = ["name", "view"] + keys
        self.rows: List[Dict[str, Any]] = []
        self._all_rows: List[Dict[str, Any]] = list(existing_rows or [])
        self._seen_keys = {
            (str(r.get("name", "")), str(r.get("view", "")))
            for r in self._all_rows
        }
        self._acc = accelerator
        self._is_main = accelerator.is_main_process
        self._ddp = accelerator.num_processes > 1
        self.csv_path = out_dir / "teacher_student_similarity.csv"
        self._f = None
        if self._is_main:
            mode = "a" if self._all_rows else "w"
            self._f = open(self.csv_path, mode, newline="", encoding="utf-8")
            self._w = csv.DictWriter(self._f, fieldnames=self.fields)
            if not self._all_rows:
                self._w.writeheader()
            self._f.flush()

    def log(self, row: Dict[str, Any]) -> None:
        self.log_many([row])

    def log_many(self, rows: List[Dict[str, Any]]) -> None:
        self.rows.extend(rows)
        if self._ddp:
            import torch.distributed as dist
            buf: List[Any] = [None] * self._acc.num_processes
            dist.all_gather_object(buf, rows)
            if self._is_main:
                for rank_rows in buf:
                    for r in rank_rows:
                        self._write_row(r)
                self._f.flush()
        else:
            if self._is_main:
                for row in rows:
                    self._write_row(row)
                self._f.flush()

    def _write_row(self, row: Dict[str, Any]) -> None:
        key = (str(row.get("name", "")), str(row.get("view", "")))
        if key in self._seen_keys:
            return
        self._seen_keys.add(key)
        self._all_rows.append(row)
        if self._is_main:
            self._w.writerow(row)

    def finalize(self) -> Optional[Dict[str, float]]:
        if self._f is not None and not self._f.closed:
            self._f.close()
        if not self._is_main or not self._all_rows:
            return None
        avg = {
            k: round(float(np.mean([float(r[k]) for r in self._all_rows])), 4)
            for k in self.keys
        }
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fields)
            w.writeheader()
            for r in self._all_rows:
                w.writerow(r)
            w.writerow({"name": "AVERAGE", "view": "-", **avg})
        json_path = self.out_dir / "teacher_student_similarity.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"samples": self._all_rows, "average": avg}, f, indent=2, ensure_ascii=False)
        return avg


# =====================================================================
# Student Forward:Dense ODE + Dense FlowEdit + Sparse ODE
# =====================================================================

def trellis_forward_dense_flowedit(
    system,
    state: TrellisState,
    cfg: ml_collections.ConfigDict,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Student Dedicated forward:Dense ODE → Dense FlowEdit → decode_to_coords → Sparse ODE → Render

    FlowEdit Parameters from cfg.rollout.flowedit Read.
    """
    pipeline = system.pipeline
    _seed = int(cfg.seed)

    # ---- Stage 1: Dense ODE + FlowEdit ----
    # rollout_dense_flowedit Run internally first ODE(CPU gen), run again FlowEdit(CUDA gen, seeds+1)
    gen_dense = torch.Generator(device="cpu").manual_seed(_seed)
    with torch.no_grad():
        rollout_dense_flowedit(state, cfg, system, device, generator=gen_dense)
    # state.stage1.z0 = FlowEdit edited dense latent

    batch_size = state.stage1.z0.shape[0]
    state.coords = pipeline.dense.decode_to_coords(state.stage1.z0, batch_size=batch_size)

    # ---- Stage 2: Sparse ODE ----
    gen_sparse = torch.Generator(device=device).manual_seed(_seed)
    with torch.no_grad():
        rollout_sparse(state, cfg, system, device, generator=gen_sparse, is_training=False)
    latents = state.stage2.z0

    torch.cuda.empty_cache()

    # ---- Decode & Render ----
    renderer_type = cfg.renderer.type
    renderer = system.renderers[renderer_type]

    if renderer_type == "gs":
        render_out = decode_and_render_gs(latents, state.cameras, pipeline, renderer, device)
    else:
        render_out = decode_and_render_mesh(latents, state.cameras, pipeline, renderer, device)
        render_out["color"] = render_out["normal"]

    state.views_generated.image_tensor = render_out["color"]
    return render_out


# =====================================================================
# DDP barrier
# =====================================================================

def _dist_barrier(local_rank: int) -> None:
    import torch.distributed as dist
    if dist.is_initialized():
        t = torch.zeros(1, device=f"cuda:{local_rank}")
        dist.all_reduce(t, op=dist.ReduceOp.SUM)


# =====================================================================
# Main process
# =====================================================================

def main(argv) -> None:
    del argv
    cfg = _CONFIG.value
    cfg.eval_only = True

    # Inject FlowEdit Default configuration (if config file is not defined)
    _ensure_flowedit_cfg(cfg)

    # Dense FlowEdit eval Fixed use 360 degree 8 Perspectives to avoid relying on sharing config the default eval Number of viewing angles.
    cfg.data.eval.n_view = 8
    cfg.data.eval.yaw_range = [0.0, 360.0]
    cfg.data.eval.pitch_range = [0.0, 0.0]

    setup_env_and_seed(cfg)
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    os.environ.setdefault("NCCL_TIMEOUT", "1800")

    accelerator = Accelerator(mixed_precision="no", kwargs_handlers=[])
    local_rank = accelerator.local_process_index
    torch.cuda.set_device(local_rank)
    device = accelerator.device
    is_main = accelerator.is_main_process

    fe_cfg = cfg.rollout.flowedit
    logger.info(
        f"[Rank {accelerator.process_index}] device={device} | "
        f"FlowEdit steps={fe_cfg.steps} n_max={fe_cfg.n_max} "
        f"cfg_tgt={fe_cfg.cfg_scale_tgt} cfg_src={fe_cfg.cfg_scale_src}"
    )
    logger.info(
        f"[Rank {accelerator.process_index}] Eval camera | "
        f"n_view={cfg.data.eval.n_view} "
        f"yaw_range={list(cfg.data.eval.yaw_range)} "
        f"pitch_range={list(cfg.data.eval.pitch_range)}"
    )

    # ---- data ----
    eval_loader = build_eval_dataloader(cfg, accelerator)

    # ---- Build system ----
    system = build_system(cfg, accelerator, guidance_factory=create_guidance)
    system = system.prepare_lora(cfg, adapter="base", load_path=None, clone_from=None)

    # ---- Load finetuned checkpoint ----
    ckpt_path = cfg.get("checkpoint", "")
    checkpoint_load_report: Optional[Dict[str, Any]] = None
    parameter_diff_reports: List[Dict[str, Any]] = []
    if ckpt_path:
        pipe_models = system.pipeline.pipe.models
        checkpoint_load_report = load_full_dual_checkpoint(
            ckpt_path,
            slat_flow_model=pipe_models["slat_flow_model"],
            sparse_structure_flow_model=pipe_models["sparse_structure_flow_model"],
        )
        for stage_report in checkpoint_load_report["stages"]:
            logger.info("[Checkpoint] %s", json.dumps(stage_report, sort_keys=True))

        sparse_teacher = getattr(system.strategy, "_sparse_teacher", None)
        dense_teacher = getattr(system.strategy, "_dense_teacher", None)
        if sparse_teacher is None or dense_teacher is None:
            raise RuntimeError(
                "full Dual-Stage checkpoint validation requires both sparse and dense "
                "base/teacher models"
            )
        parameter_diff_reports = [
            validate_student_diff(
                pipe_models["slat_flow_model"],
                sparse_teacher,
                target="slat_flow_model",
            ),
            validate_student_diff(
                pipe_models["sparse_structure_flow_model"],
                dense_teacher,
                target="sparse_structure_flow_model",
            ),
        ]
        for diff_report in parameter_diff_reports:
            logger.info("[ParamCheck] %s", json.dumps(diff_report, sort_keys=True))
    else:
        logger.warning("[Checkpoint] unspecified checkpoint,student Use pretrained weight (with teacher same)")

    # ---- Output directory ----
    run_root = Path(cfg.logdir) / (cfg.run_name or "run")
    ckpt_tag = Path(str(ckpt_path).rstrip("/")).name if ckpt_path else "pretrained_baseline"
    # Distinguish this script from the standard eval
    ckpt_tag = ckpt_tag + "_dense_flowedit"
    out_dir = run_root / "eval_teacher_student" / ckpt_tag
    images_dir = out_dir / "images"
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        # record FlowEdit Operating parameters
        with open(out_dir / "flowedit_params.json", "w") as f:
            json.dump({
                "steps": int(fe_cfg.steps),
                "n_max": int(fe_cfg.n_max),
                "cfg_scale_tgt": float(fe_cfg.cfg_scale_tgt),
                "cfg_scale_src": float(fe_cfg.cfg_scale_src),
                "checkpoint": ckpt_path,
                "has_dense_ckpt": checkpoint_load_report is not None,
                "checkpoint_load_report": checkpoint_load_report,
                "parameter_diff_reports": parameter_diff_reports,
            }, f, indent=2)
    _dist_barrier(local_rank)

    # ---- Indicator model (lazy initialization)----
    clip_m: Optional[CLIPMetric] = None
    dino_m: Optional[DINOMetric] = None

    # ---- Metrics recorder ----
    metric_keys = [
        "clip_teacher", "clip_student", "clip_delta",
        "dino_teacher", "dino_student", "dino_delta",
    ]
    existing_rows, completed_names = _load_existing_eval_rows(
        out_dir / "teacher_student_similarity.csv",
        ["name", "view"] + metric_keys,
        int(cfg.data.eval.n_view),
    )
    if is_main and existing_rows:
        logger.info(
            f"[Resume] Found that there is already CSV rows={len(existing_rows)},"
            f"full sample={len(completed_names)}, the complete sample will be skipped and the make-up run will continue."
        )
    el = EvalMetricLogger(out_dir, metric_keys, accelerator, existing_rows=existing_rows)

    # ---- evaluation loop ----
    pipe_models = system.pipeline.pipe.models
    inference_ctx = system.strategy.inference_context() if system.strategy else nullcontext()

    with inference_ctx, EvalModeGuard(
        pipe_models["slat_flow_model"],
        pipe_models["sparse_structure_flow_model"],
        pipe_models["slat_decoder_mesh"],
        pipe_models["slat_decoder_gs"],
    ):
        loader = tqdm(eval_loader, desc="Eval") if is_main else eval_loader
        for batch_idx, batch in enumerate(loader):
            batch_paths = batch.get("paths", [])
            if isinstance(batch_paths, str):
                batch_paths = [batch_paths]
            batch_names = [
                os.path.splitext(os.path.basename(str(p)))[0]
                for p in batch_paths
            ]
            if batch_names and all(name in completed_names for name in batch_names):
                el.log_many([])
                continue

            with torch.no_grad():
                # === Student: Dense ODE + FlowEdit → Sparse ODE ===
                state_stu = TrellisState()
                state_stu.attach_batch(batch, pipeline=system.pipeline)
                render_stu = trellis_forward_dense_flowedit(
                    system, state_stu, cfg, device,
                )
                comp_rgb_stu = render_stu["color"]  # (B,V,H,W,C)

                # === Teacher: Standard Dense ODE → Sparse ODE(pretrained weight)===
                with system.strategy.sparse_teacher_context(), \
                     system.strategy.dense_teacher_context():
                    state_tea = TrellisState()
                    state_tea.attach_batch(batch, pipeline=system.pipeline)
                    # Teacher always use your own dense model Regenerate coords(do not share student coords)
                    render_tea = trellis_forward(
                        system, state_tea, cfg, device,
                        global_step=0, is_training=False,
                    )
                comp_rgb_tea = render_tea["color"]  # (B,V,H,W,C)

            # ---- Lazy initialization indicator model ----
            if clip_m is None:
                clip_m = CLIPMetric(weight=1.0, device=device)
                dino_m = DINOMetric(weight=1.0, device=device)

            # ---- Calculate indicators on a sample-by-sample basis + save image ----
            B, V = comp_rgb_stu.shape[:2]
            h, w = comp_rgb_stu.shape[2], comp_rgb_stu.shape[3]
            batch_rows: List[Dict[str, Any]] = []

            for b in range(B):
                name = os.path.splitext(
                    os.path.basename(state_stu.views_conditioned.paths[b])
                )[0]
                cond_pil = composite_alpha_to_white(
                    state_stu.views_conditioned.image_pils[b]
                ).resize((w, h), Image.LANCZOS)

                for v in range(V):
                    stu_pil = _to_pil(comp_rgb_stu[b, v])
                    tea_pil = _to_pil(comp_rgb_tea[b, v])

                    _save_images(images_dir, name, cond_pil, stu_pil, tea_pil, v)

                    cs = _similarity(clip_m, stu_pil, cond_pil)
                    ct = _similarity(clip_m, tea_pil, cond_pil)
                    ds = _similarity(dino_m, stu_pil, cond_pil)
                    dt = _similarity(dino_m, tea_pil, cond_pil)

                    batch_rows.append({
                        "name": name,
                        "view": v,
                        "clip_teacher": round(ct, 4),
                        "clip_student": round(cs, 4),
                        "clip_delta": round(cs - ct, 4),
                        "dino_teacher": round(dt, 4),
                        "dino_student": round(ds, 4),
                        "dino_delta": round(ds - dt, 4),
                    })

                    if is_main:
                        logger.info(
                            f"[{name} v{v}] "
                            f"CLIP tea={ct:.4f} stu={cs:.4f} Δ{cs - ct:+.4f} | "
                            f"DINO tea={dt:.4f} stu={ds:.4f} Δ{ds - dt:+.4f}"
                        )

            el.log_many(batch_rows)

            del state_stu, state_tea, render_stu, render_tea
            del comp_rgb_stu, comp_rgb_tea
            torch.cuda.empty_cache()

    # ---- Summary ----
    _dist_barrier(local_rank)
    avg = el.finalize()

    if is_main and avg:
        logger.info("=" * 60)
        logger.info(f"[Dense FlowEdit Eval] steps={fe_cfg.steps} n_max={fe_cfg.n_max} "
                    f"cfg_tgt={fe_cfg.cfg_scale_tgt} cfg_src={fe_cfg.cfg_scale_src}")
        logger.info(
            f"CLIP:  teacher={avg['clip_teacher']:.4f}  "
            f"student={avg['clip_student']:.4f}  Δ={avg['clip_delta']:+.4f}"
        )
        logger.info(
            f"DINO:  teacher={avg['dino_teacher']:.4f}  "
            f"student={avg['dino_student']:.4f}  Δ={avg['dino_delta']:+.4f}"
        )
        logger.info(f"CSV:  {el.csv_path}")
        logger.info(f"JSON: {out_dir / 'teacher_student_similarity.json'}")
        logger.info("=" * 60)

    for m in [clip_m, dino_m]:
        if m is not None:
            m.cleanup()

    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    app.run(main)
