"""TRELLIS Dual-Stage Contrastive FlowEdit training configuration — Sparse + Dense Two-stage contrastive learning.

Corresponding entrance: oreo.systems.trellis.entries.oreo_autograd

Training process:
  ── Sparse Stage ──
  Pretrained Rollout (frozen) → clean z₀ (sparse)
  → Noise → Student velocity → ẑ₀ → Render → FlowEdit → c_src / c_tgt
  → Contrastive loss → slat_flow_model.grad

  ── Dense Stage(Reuse c_src / c_tgt)──
  Dense Rollout (frozen) → clean z_s (dense)
  → Noise → Student velocity → ẑ_s₀
  → Teacher denoise with c_tgt / c_src → positive / negative
  → Contrastive loss → sparse_structure_flow_model.grad

Configuration structure:
    cfg.guidance              → Guidance initialize(FlowEdit Model loading)
    cfg.train.guidance        → FlowEdit runtime parameters (prompt / cfg scale / loss weight)
    cfg.train.noise           → Noise time step sampling configuration (Sparse + Dense share)
    cfg.train.loss.contrastive → Contrast loss Parameters (ada, eps, dense_weight)
    cfg.renderer              → Renderer configuration (only gs)
    cfg.train                 → Training hyperparameters (mode, optimizer, loss, gradient_accumulation_steps)
"""
import ml_collections


# =====================================================================
# Auxiliary configuration function
# =====================================================================

def _flowedit_init_config(g: ml_collections.ConfigDict):
    """FlowEdit Sampling parameters (construct Pipeline read once)."""
    g.flowedit = ml_collections.ConfigDict()
    g.flowedit.steps = 12       # Total time steps
    g.flowedit.n_max = 9        # actual execution at the end n_max step
    g.flowedit.noise_mode = "aligned"  # Noise mode (aligned / random)
    g.flowedit.csd_pos_mode = "cfg"
    g.flowedit.csd_neg_mode = "uncond"
    g.flowedit.remove_tgt_neg = True


def _flowedit_runtime_config():
    """FlowEdit runtime parameters (per-call, pass in compute_guidance)."""
    cfg = ml_collections.ConfigDict()

    cfg.seed = 0
    cfg.bg_color = [1.0, 1.0, 1.0]

    # Target branch
    cfg.true_cfg_scale_tgt = 4
    cfg.target_prompt = "Rotate the camera. White background."
    cfg.negative_prompt_tgt = " "

    # Source branch
    cfg.true_cfg_scale_src = -1 * cfg.true_cfg_scale_tgt  # reverse boot
    cfg.source_prompt = cfg.target_prompt
    cfg.negative_prompt_src = cfg.negative_prompt_tgt

    # multiple steps Loss Aggregation (only compute_guidance path usage,edit() Path is not required)
    cfg.reduce_mode = "final"
    cfg.ada_normalize = False
    cfg.ada_eps = 1e-4

    return cfg


def _adaptive_distance_config(cfg: ml_collections.ConfigDict):
    """for train/eval Data addition adaptive_distance configuration."""
    cfg.data.train.adaptive_distance = ml_collections.ConfigDict()
    cfg.data.train.adaptive_distance.enabled = True
    cfg.data.train.adaptive_distance.fill_ratio = 0.9

    cfg.data.eval.adaptive_distance = ml_collections.ConfigDict()
    cfg.data.eval.adaptive_distance.enabled = True
    cfg.data.eval.adaptive_distance.fill_ratio = 0.9


# =====================================================================
# main configuration
# =====================================================================

def get_config():
    """TRELLIS Dual-Stage Contrastive FlowEdit training configuration (Sparse + Dense Dual-stage contrastive learning)."""
    cfg = ml_collections.ConfigDict()

    # === General ===
    cfg.run_name = "trellis_oreo"
    cfg.use_wandb = False
    cfg.seed = 42
    cfg.logdir = "logs"
    cfg.num_epochs = 500
    cfg.checkpoint = ""
    cfg.eval_only = False
    # Keep the training precision explicit so it is persisted with each run.
    cfg.mixed_precision = "no"

    # === frequency control ===
    cfg.freq = ml_collections.ConfigDict()
    cfg.freq.save = ml_collections.ConfigDict()
    cfg.freq.save.visual = 1
    cfg.freq.save.ckpt = 1
    cfg.freq.save.progress_samples = 4
    cfg.freq.eval = 1

    # === Data configuration ===
    cfg.data = ml_collections.ConfigDict()

    cfg.data.train = ml_collections.ConfigDict()
    cfg.data.train.dir = "dataset/alphaimages_v3/train"
    cfg.data.train.batch_size = 1
    cfg.data.train.n_view = 1
    cfg.data.train.yaw_range = [0.0, 360.0]
    cfg.data.train.pitch_range = [0.0, 0.0]
    cfg.data.train.r_range = [2.0, 2.0]
    cfg.data.train.fov_range = [40.0, 40.0]

    cfg.data.eval = ml_collections.ConfigDict()
    cfg.data.eval.dir = "dataset/alphaimages_v3/test"
    cfg.data.eval.batch_size = 1
    cfg.data.eval.n_view = 3
    cfg.data.eval.yaw_range = [90.0, 270.0]
    cfg.data.eval.pitch_range = [0.0, 0.0]
    cfg.data.eval.r_range = [2.0, 2.0]
    cfg.data.eval.fov_range = [40.0, 40.0]
    _adaptive_distance_config(cfg)

    # === Pre-trained weights ===
    cfg.pretrained = ml_collections.ConfigDict()
    cfg.pretrained.model = "./pretrained_weights/TRELLIS-image-large"

    # === Renderer configure(★ only GS Color rendering) ===
    cfg.renderer = ml_collections.ConfigDict()
    cfg.renderer.resolution = 1024
    cfg.renderer.type = "gs"
    cfg.renderer.ssaa = 1

    cfg.renderer.gs = ml_collections.ConfigDict()
    cfg.renderer.gs.near = 0.8
    cfg.renderer.gs.far = 1.6
    cfg.renderer.gs.bg_color = [1.0, 1.0, 1.0]

    # === Training hyperparameters ===
    cfg.train = tr = ml_collections.ConfigDict()

    tr.mode = "full"  # Contrastive Use full parameter fine-tuning

    # Rollout mode: "pretrained" (off-policy) | "student" (on-policy)
    tr.rollout_mode = "student"

    # Student Whether to use single-step denoising CFG(Control simultaneously P3.5 reg teacher)
    # True = keep pipeline Default, False = skip uncond forward pass
    tr.student_denoise_cfg = True

    tr.gradient_accumulation_steps = 1
    tr.optimizer = ml_collections.ConfigDict()
    tr.optimizer.type = "adan"
    tr.optimizer.lr = 1e-4
    tr.optimizer.weight_decay = 0.0
    tr.optimizer.eps = 1e-4

    # ★ enable Dense Stage optimizer(Dual-Stage training required)
    tr.dense_optimizer = True

    # === Rollout Configuration ===
    cfg.rollout = ml_collections.ConfigDict()
    cfg.rollout.type = "ode"
    cfg.rollout.reg = ml_collections.ConfigDict()
    cfg.rollout.reg.type = "none"  # Contrastive No need rollout regularization

    # === ★ Noisy time step sampling configuration ===
    tr.noise = ml_collections.ConfigDict()
    tr.noise.t_min = 0.02
    tr.noise.t_max = 0.98

    # === Guidance Initialization configuration (FlowEdit Model loading) ===
    cfg.guidance = g = ml_collections.ConfigDict()
    g.type = "flowedit"
    g.model_path = "Qwen/Qwen-Image-Edit-2511"
    g.edit_resolution = 1024
    _flowedit_init_config(g)

    # === Guidance Runtime configuration (FlowEdit) ===
    tr.guidance = _flowedit_runtime_config()
    tr.guidance.bg_color = cfg.renderer.gs.bg_color

    # === Loss Configuration ===
    tr.loss = ml_collections.ConfigDict()
    tr.loss.guidance = 1.0      # Contrastive loss weight
    tr.loss.reg = 1.0           # velocity Regularized weight
    tr.loss.reg_type = "v"      # Regularization type: "v" | "x0" | "x1"

    # ★ Contrastive loss Configuration
    tr.loss.contrastive = ml_collections.ConfigDict()
    tr.loss.contrastive.ada = True
    tr.loss.contrastive.eps = 0.1
    tr.loss.contrastive.adaptive_swap = False
    # Contrast teacher Right/Whether to use negative samples CFG(independent of student_denoise_cfg)
    # True = teacher keep CFG(zeros uncond),False = teacher No need CFG
    tr.loss.contrastive.teacher_cfg = True

    return cfg
