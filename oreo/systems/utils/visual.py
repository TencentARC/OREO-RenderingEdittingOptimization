"""VisualIO — save training / eval visualization grids and dumps."""
import os
import numpy as np
from PIL import Image
from accelerate import Accelerator
from pathlib import Path

from .mixins import WandbMixin


# =====================================================================
# General image processing tools
# =====================================================================

def composite_alpha_to_black(img: Image.Image) -> Image.Image:
    """
    will have Alpha The image of the channel is composited onto a black background and converted to RGB.
    If the image does not Alpha channel, directly converted to RGB.

    with TRELLIS preprocess_image of Alpha Premultiplication processing remains consistent:
    output = output[:, :, :3] * output[:, :, 3:4]
    """
    if img.mode == 'RGBA':
        background = Image.new('RGBA', img.size, (0, 0, 0, 255))  # black opaque background
        combined = Image.alpha_composite(background, img)
        return combined.convert('RGB')
    else:
        return img.convert('RGB')


def composite_alpha(img: Image.Image, bg_color=(1.0, 1.0, 1.0)) -> Image.Image:
    """
    will have Alpha The image of the channel is synthesized onto the specified background color and converted to RGB.

    Args:
        img: PIL Image(can be RGBA or RGB)
        bg_color: background color,float [0, 1] of (R, G, B)

    Returns:
        RGB PIL Image
    """
    if img.mode == 'RGBA':
        bg = tuple(int(c * 255) for c in bg_color)
        background = Image.new('RGBA', img.size, (*bg, 255))
        return Image.alpha_composite(background, img).convert('RGB')
    return img.convert('RGB')


def composite_alpha_to_white(img: Image.Image) -> Image.Image:
    """
    will have Alpha The image of the channel is composited onto a white background and converted to RGB.
    If the image does not Alpha channel, directly converted to RGB.
    """
    if img.mode == 'RGBA':
        background = Image.new('RGBA', img.size, (255, 255, 255, 255))  # White opaque background
        combined = Image.alpha_composite(background, img)
        return combined.convert('RGB')
    else:
        return img.convert('RGB')


# =====================================================================
# VisualIO - training/Evaluate visualization saves (using WandbMixin)
# =====================================================================


class VisualIO(WandbMixin):
    """
    Visualization tool base class (state irrelevant).

    Provides common tool methods (to_pil / make_grid / save_pil / resize / get_names)
    and general save_batch_eval.

    ★ Not included save_batch_train: Training visualization depends on specific State The field structure of
      By subclass (TrellisVisualIO / Trellis2VisualIO) is realized.
    """

    def __init__(
        self,
        root: Path,
        target_h: int = 512,
        vis_freq: int = 100,
        accelerator: Accelerator = None,
        max_wandb_samples: int = 4,
    ):
        self.root = root
        self.target_h = target_h
        self.vis_freq = vis_freq
        self.accelerator = accelerator
        self.max_wandb_samples = max_wandb_samples

    # ===== Tool method =====

    @staticmethod
    def to_pil(x) -> Image.Image:
        """Tensor/ndarray -> PIL"""
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        return Image.fromarray((x * 255).clip(0, 255).astype(np.uint8))

    def resize(self, img: Image.Image) -> Image.Image:
        """press target_h proportional scaling"""
        w, h = img.size
        scale = self.target_h / max(1, h)
        return img.resize((max(1, int(w * scale)), self.target_h), Image.Resampling.LANCZOS)

    @staticmethod
    def get_names(state) -> list:
        """from state Extract a list of image names"""
        return [os.path.splitext(os.path.basename(p))[0] for p in state.views_conditioned.paths]

    def make_grid(self, images: list) -> Image.Image:
        """will PIL Lists organized into a horizontal grid"""
        imgs = [self.resize(im) for im in images]
        margin = 12
        total_w = sum(im.width for im in imgs) + margin * (len(imgs) + 1)
        total_h = max(im.height for im in imgs) + margin * 2
        canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        x = margin
        for im in imgs:
            canvas.paste(im, (x, margin))
            x += im.width + margin
        return canvas

    def save_pil(self, pil: Image.Image, path: Path) -> None:
        """save PIL to file (automatically creates directory)"""
        path.parent.mkdir(parents=True, exist_ok=True)
        pil.save(path)

    # ===== Assessment method =====

    def save_batch_eval(self, state, epoch: int, render_out: dict = None, pipeline=None, export_mesh: bool = False) -> None:
        """
        Evaluation mode: Save renderings + Optional mesh Export.

        Directory structure: root/epoch_{N}/{name}/color_v0.png, color_v1.png, ..., mesh.obj
        """
        names = self.get_names(state)
        out_dir = self.root / f"epoch_{epoch}"
        gens = state.views_generated.image_tensor  # (B, V, H, W, C)
        meshes = (render_out or {}).get("meshes", [])
        num_views = gens.shape[1]  # Number of viewing angles V

        wandb_images = {}
        for b, name in enumerate(names):
            sample_dir = out_dir / name

            # Save all perspectives color
            for v in range(num_views):
                color_pil = self.to_pil(gens[b, v])  # (H, W, C)
                self.save_pil(color_pil, sample_dir / f"color_v{v}.png")
                if b < self.max_wandb_samples and v == 0:
                    wandb_images[f"eval/{name}"] = color_pil

            # Export mesh
            if export_mesh and pipeline and b < len(meshes):
                pipeline.export_mesh_obj(meshes[b], str(sample_dir / "mesh.obj"))

        self.log_images(wandb_images, step=epoch, prefix="")


# =====================================================================
# TrellisVisualIO - Trellis Dedicated visualization (supports Hybrid Dual-pass rendering)
# =====================================================================


class TrellisVisualIO(VisualIO):
    """
    Trellis Dedicated visualization save (TrellisState perception).

    Applicable scenarios:
    - One-way training (save_batch_train): [cond | gen | edit | progress] → {name}.png
    - Hybrid two way training:
      - save_normal_train: [cond | normal | edit | progress] → {name}_normal.png
      - save_color_train:  [cond | color | edit | progress]  → {name}_color.png

    TrellisState Field mapping:
    - views_generated: image_tensor(GS Color),normal_tensor(Mesh Normal)
    - views_edited:    color_tensor / color_trackers,normal_tensor / normal_trackers

    evaluation mode (save_batch_eval overwrite):
    - normal.png(normal_tensor (when valuable)
    - color.png(image_tensor (when valuable)
    """

    def _save_stage_train(
        self, state, epoch: int, step: int,
        render_tensor,       # (B, V, H, W, C) Rendering results
        suffix: str,         # File suffix, such as "_normal" or "_color"
        edit_tensor=None,    # (B, V, C, H, W) edit Result (passed in explicitly)
        trackers=None,       # List[StateTracker](passed in explicitly)
        pipe=None,           # guidance pipeline(used to generate progress grid)
        n_progress_samples: int = 0,
        wandb_prefix: str = "train",
    ) -> None:
        """
        Internal method: Saves a training visualization mesh for a single render pass.

        grid content: [cond | render | edit | progress]
          - edit / progress Only displayed if the corresponding parameter has a value
        Directory structure: root/epoch_{N}/step_{M}/{name}_{suffix}.png
        """
        names = self.get_names(state)
        out_dir = self.root / f"epoch_{epoch}" / f"step_{step}"
        conds = state.views_conditioned.image_pils

        wandb_images = {}
        for b, name in enumerate(names):
            imgs = [composite_alpha_to_white(conds[b])]

            if render_tensor is not None:
                imgs.append(self.to_pil(render_tensor[b, 0]))

            if edit_tensor is not None:
                imgs.append(self.to_pil(edit_tensor[b, 0].permute(1, 2, 0)))

            if trackers and pipe and n_progress_samples > 0:
                imgs.append(trackers[b].get_progress_grid(pipe, n_progress_samples))

            grid = self.make_grid(imgs)
            self.save_pil(grid, out_dir / f"{name}{suffix}.png")

            if b < self.max_wandb_samples:
                wandb_images[f"{wandb_prefix}/{name}{suffix}"] = grid

        self.log_images(wandb_images, step=step, prefix="")

    # ===== One way training =====

    def save_batch_train(
        self, state, epoch: int, step: int,
        pipe=None, n_progress_samples: int = 0,
    ) -> None:
        """
        Single-channel training visualization: [cond | gen | edit | progress] → {name}.png

        read views_generated.image_tensor and views_edited.color_tensor.
        """
        self._save_stage_train(
            state, epoch, step,
            render_tensor=state.views_generated.image_tensor,
            edit_tensor=state.views_edited.color_tensor,
            trackers=state.views_edited.color_trackers,
            suffix="",
            pipe=pipe,
            n_progress_samples=n_progress_samples,
        )

    # ===== Hybrid two way training =====

    def save_normal_train(
        self, state, epoch: int, step: int,
        pipe=None, n_progress_samples: int = 0,
    ) -> None:
        """Mesh Normal Render visualization: [cond | normal | edit | progress] → {name}_normal.png"""
        self._save_stage_train(
            state, epoch, step,
            render_tensor=state.views_generated.normal_tensor,
            edit_tensor=state.views_edited.normal_tensor,
            trackers=state.views_edited.normal_trackers,
            suffix="_normal",
            pipe=pipe,
            n_progress_samples=n_progress_samples,
        )

    def save_color_train(
        self, state, epoch: int, step: int,
        pipe=None, n_progress_samples: int = 0,
    ) -> None:
        """GS Color Render visualization: [cond | color | edit | progress] → {name}_color.png"""
        self._save_stage_train(
            state, epoch, step,
            render_tensor=state.views_generated.image_tensor,
            edit_tensor=state.views_edited.color_tensor,
            trackers=state.views_edited.color_trackers,
            suffix="_color",
            pipe=pipe,
            n_progress_samples=n_progress_samples,
        )

    # ===== Contrastive training =====

    def save_contrastive_train(
        self, state, epoch: int, step: int,
        pipe=None, n_progress_samples: int = 0,
    ) -> None:
        """Contrastive Visualization: [cond | src_render | tgt_edit | progress] → {name}_contrastive.png"""
        self._save_stage_train(
            state, epoch, step,
            render_tensor=state.views_generated.image_tensor,
            edit_tensor=state.views_edited.image_tensor,
            trackers=state.trackers.guidance,
            suffix="_contrastive",
            pipe=pipe,
            n_progress_samples=n_progress_samples,
        )

    def save_batch_eval(self, state, epoch: int, render_out: dict = None, pipeline=None, export_mesh: bool = False) -> None:
        """
        Evaluation mode: Save separately by renderer (supports multiple perspectives).

        Directory structure:
            root/epoch_{N}/{name}/
            ├── normal_v0.png, normal_v1.png, ...   # normal_tensor Save when there is value
            ├── color_v0.png, color_v1.png, ...     # image_tensor Save when there is value
        """
        names = self.get_names(state)
        out_dir = self.root / f"epoch_{epoch}"
        vg = state.views_generated

        wandb_images = {}
        for b, name in enumerate(names):
            sample_dir = out_dir / name

            if vg.normal_tensor is not None:
                num_views = vg.normal_tensor.shape[1]  # V
                for v in range(num_views):
                    normal_pil = self.to_pil(vg.normal_tensor[b, v])  # (H, W, C)
                    self.save_pil(normal_pil, sample_dir / f"normal_v{v}.png")
                    if b < self.max_wandb_samples and v == 0:
                        wandb_images[f"eval/{name}/normal"] = normal_pil

            if vg.image_tensor is not None:
                num_views = vg.image_tensor.shape[1]  # V
                for v in range(num_views):
                    color_pil = self.to_pil(vg.image_tensor[b, v])  # (H, W, C)
                    self.save_pil(color_pil, sample_dir / f"color_v{v}.png")
                    if b < self.max_wandb_samples and v == 0:
                        wandb_images[f"eval/{name}/color"] = color_pil

        self.log_images(wandb_images, step=epoch, prefix="")


# =====================================================================
# Trellis2VisualIO - Trellis2 Dedicated visualization
# =====================================================================


class Trellis2VisualIO(VisualIO):
    """
    Trellis2 Dedicated visualization saving (automatic adaptation Shape / Tex / Shape+Tex).

    with VisualIO The difference:
    - Trellis2 of ViewsGenerated contains shape_tensor(normal) and pbr_tensor(RGB),
      rather than a single image_tensor.
    - Training visualizations are saved separately by stage:
      - save_shape_train: [cond | normal | edited_normal]  → {name}_shape.png
      - save_tex_train:   [cond | rgb   | edited_rgb]      → {name}_tex.png
    - Save the evaluation visualization into files:
      - normal.png(shape_tensor (when valuable)
      - color.png(pbr_tensor (when valuable)
      - mesh.obj(export_mesh=True time)

    How to use:
        # shape-only:
        visual_io.save_shape_train(state, epoch, step)

        # tex-only:
        visual_io.save_tex_train(state, epoch, step)

        # shape+tex:
        visual_io.save_shape_train(state, epoch, step)  # shape guidance after
        # ... tex forward + tex guidance ...
        visual_io.save_tex_train(state, epoch, step)     # tex guidance after
    """

    def _save_stage_train(
        self, state, epoch: int, step: int,
        render_tensor,  # (B, V, H, W, C) Rendering results
        suffix: str,    # File suffix, such as "_shape" or "_tex"
        wandb_prefix: str = "train",
    ) -> None:
        """
        Internal method: Saves the training visualization grid for a single stage.

        grid content: [cond | render | edit]
        Directory structure: root/epoch_{N}/step_{M}/{name}_{suffix}.png
        """
        names = self.get_names(state)
        out_dir = self.root / f"epoch_{epoch}" / f"step_{step}"

        conds = state.views_conditioned.image_pils
        edits = state.views_edited.color_tensor  # current stage edit(The timing of the call determines the content)

        wandb_images = {}
        for b, name in enumerate(names):
            imgs = [composite_alpha_to_white(conds[b])]

            # Rendering results
            if render_tensor is not None:
                imgs.append(self.to_pil(render_tensor[b, 0]))

            # Edited image (guidance edit)
            if edits is not None:
                imgs.append(self.to_pil(edits[b, 0].permute(1, 2, 0)))

            grid = self.make_grid(imgs)
            self.save_pil(grid, out_dir / f"{name}{suffix}.png")

            if b < self.max_wandb_samples:
                wandb_images[f"{wandb_prefix}/{name}{suffix}"] = grid

        self.log_images(wandb_images, step=step, prefix="")

    def save_shape_train(self, state, epoch: int, step: int) -> None:
        """
        Shape Stage training visualization: [cond | normal | edited_normal]

        must be in shape guidance After that,tex forward before calling,
        Otherwise state.views_edited will be tex guidance Cover.
        """
        self._save_stage_train(
            state, epoch, step,
            render_tensor=state.views_generated.shape_tensor,
            suffix="_shape",
        )

    def save_tex_train(self, state, epoch: int, step: int) -> None:
        """
        Tex Stage training visualization: [cond | rgb | edited_rgb]

        in tex guidance Called afterwards.
        """
        self._save_stage_train(
            state, epoch, step,
            render_tensor=state.views_generated.pbr_tensor,
            suffix="_tex",
        )

    def save_batch_eval(self, state, epoch: int, render_out: dict = None, pipeline=None, export_mesh: bool = False) -> None:
        """
        Evaluation mode: Save renderings separately by stage + Optional mesh Export (supports multiple perspectives).

        Directory structure:
            root/epoch_{N}/{name}/
            ├── normal_v0.png, normal_v1.png, ...   # shape_tensor Save when there is value
            ├── color_v0.png, color_v1.png, ...     # pbr_tensor Save when there is value
            └── mesh.obj                            # export_mesh=True Save when
        """
        names = self.get_names(state)
        out_dir = self.root / f"epoch_{epoch}"
        vg = state.views_generated
        meshes = (render_out or {}).get("meshes", [])

        wandb_images = {}
        for b, name in enumerate(names):
            sample_dir = out_dir / name

            # Normal Figure (Shape stage)
            if vg.shape_tensor is not None:
                num_views = vg.shape_tensor.shape[1]  # V
                for v in range(num_views):
                    normal_pil = self.to_pil(vg.shape_tensor[b, v])  # (H, W, C)
                    self.save_pil(normal_pil, sample_dir / f"normal_v{v}.png")
                    if b < self.max_wandb_samples and v == 0:
                        wandb_images[f"eval/{name}/normal"] = normal_pil

            # RGB Figure (Tex stage)
            if vg.pbr_tensor is not None:
                num_views = vg.pbr_tensor.shape[1]  # V
                for v in range(num_views):
                    color_pil = self.to_pil(vg.pbr_tensor[b, v])  # (H, W, C)
                    self.save_pil(color_pil, sample_dir / f"color_v{v}.png")
                    if b < self.max_wandb_samples and v == 0:
                        wandb_images[f"eval/{name}/color"] = color_pil

            # Mesh Export
            if export_mesh and pipeline and b < len(meshes):
                pipeline.export_mesh_obj(meshes[b], str(sample_dir / "mesh.obj"))

        self.log_images(wandb_images, step=epoch, prefix="")

