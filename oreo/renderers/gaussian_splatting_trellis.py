"""
Gaussian Splatting renderer for Trellis (BaseRenderer 7-stage pipeline).

OREO default training renderer (cfg.renderer type gs).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import math
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict

# from reference code TRELLIS import Gaussian class
from trellis.representations.gaussian import Gaussian
from trellis.renderers.sh_utils import eval_sh

from oreo.renderers.base_renderer import (
    BaseRenderer,
    RenderConfig,
    CameraData,
    RasterOutput,
    RenderOutput,
    intrinsics_to_projection,
)


# ============================================================================
# data structure
# ============================================================================

@dataclass
class GaussianPipeConfig:
    """3DGS pipeline knobs (kernel size, etc.)."""
    kernel_size: float = 0.1
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    scale_modifier: float = 1.0
    debug: bool = False


@dataclass
class GaussianCameraData(CameraData):
    """CameraData extended with 3DGS view/proj fields."""
    image_height: int = 512
    image_width: int = 512
    world_view_transform: torch.Tensor = None  # (4, 4) transposed view matrix
    full_proj_transform: torch.Tensor = None   # (4, 4) transposed MVP matrix


@dataclass
class GaussianRasterResult:
    """Raster outputs: rendered_image (3,H,W), radii, etc."""
    rendered_image: torch.Tensor      # (3, H, W)
    viewspace_points: torch.Tensor    # (N, 3)
    visibility_filter: torch.Tensor   # (N,)
    radii: torch.Tensor               # (N,)


# ============================================================================
# Core rasterization functions
# ============================================================================

def gaussian_rasterize(
    camera_data: GaussianCameraData,
    gaussian: Gaussian,
    pipe_config: GaussianPipeConfig,
    bg_color: torch.Tensor,
    override_color: torch.Tensor = None,
) -> GaussianRasterResult:
    """Core 3DGS rasterization call."""
    # Lazy import
    from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings

    # Created to get 2D zero tensor of gradient
    screenspace_points = torch.zeros_like(
        gaussian.get_xyz,
        dtype=gaussian.get_xyz.dtype,
        requires_grad=True,
        device="cuda"
    ) + 0  # (N, 3)
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Calculate tan(fov/2)
    tanfovx = math.tan(camera_data.fov[0] * 0.5)
    tanfovy = math.tan(camera_data.fov[1] * 0.5)

    # Subpixel offset
    subpixel_offset = torch.zeros(
        (camera_data.image_height, camera_data.image_width, 2),
        dtype=torch.float32,
        device="cuda"
    )  # (H, W, 2)

    # Rasterization settings
    raster_settings = GaussianRasterizationSettings(
        image_height=camera_data.image_height,
        image_width=camera_data.image_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=pipe_config.kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=pipe_config.scale_modifier,
        viewmatrix=camera_data.world_view_transform,
        projmatrix=camera_data.full_proj_transform,
        sh_degree=gaussian.active_sh_degree,
        campos=camera_data.camera_center,
        prefiltered=False,
        debug=pipe_config.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Extract Gaussian Properties
    means3D = gaussian.get_xyz  # (N, 3)
    means2D = screenspace_points  # (N, 3)
    opacity = gaussian.get_opacity  # (N, 1)

    # covariance
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe_config.compute_cov3D_python:
        cov3D_precomp = gaussian.get_covariance(pipe_config.scale_modifier)
    else:
        scales = gaussian.get_scaling
        rotations = gaussian.get_rotation

    # color
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe_config.convert_SHs_python:
            shs_view = gaussian.get_features.transpose(1, 2).view(
                -1, 3, (gaussian.max_sh_degree + 1) ** 2
            )  # (N, 3, D^2)
            dir_pp = gaussian.get_xyz - camera_data.camera_center.repeat(
                gaussian.get_features.shape[0], 1
            )  # (N, 3)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)  # (N, 3)
            sh2rgb = eval_sh(gaussian.active_sh_degree, shs_view, dir_pp_normalized)  # (N, 3)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)  # (N, 3)
        else:
            shs = gaussian.get_features
    else:
        colors_precomp = override_color  # (N, 3)

    # Perform rasterization
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )  # rendered_image: (3, H, W), radii: (N,)

    return GaussianRasterResult(
        rendered_image=rendered_image,
        viewspace_points=screenspace_points,
        visibility_filter=(radii > 0),
        radii=radii,
    )


# ============================================================================
# GaussianRenderer
# ============================================================================

class GaussianRenderer(BaseRenderer):
    """3DGS BaseRenderer implementation for Trellis Gaussians."""

    def __init__(
        self,
        config: RenderConfig = None,
        device: str = "cuda",
        pipe_config: GaussianPipeConfig = None,
    ):
        """
        Args:
            config: Render configuration
            device: computing equipment
            pipe_config: Pipeline configuration
        """
        if config is None:
            config = RenderConfig(resolution=512, near=0.1, far=10.0, ssaa=1, bg_color='random')
        super().__init__(config, device)

        self.pipe_config = pipe_config or GaussianPipeConfig()
        self._colors_overwrite: Optional[torch.Tensor] = None

    # ========== Stage 1: Input Preparation ==========

    def prepare_inputs(
        self,
        geometry: Gaussian,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ):
        """Ready to type, reset color overlay"""
        self._colors_overwrite = None
        return super().prepare_inputs(geometry, extrinsics, intrinsics)

    def _is_empty_geometry(self, geometry: Gaussian) -> bool:
        """Check Gaussian Is it empty"""
        return geometry.get_xyz.shape[0] == 0

    # ========== Stage 2: Camera Transform ==========

    def compute_camera_data(
        self,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> GaussianCameraData:
        """
        Calculate 3DGS Private camera data

        Args:
            extrinsics: (4, 4) W2C Camera external parameters
            intrinsics: (3, 3) Camera internal parameters

        Returns:
            GaussianCameraData: Extended camera data
        """
        resolution = self.config.resolution
        ssaa = self.config.ssaa
        near, far = self.config.near, self.config.far

        # Basic camera data
        projection = intrinsics_to_projection(intrinsics, near, far)  # (4, 4)
        mvp = projection @ extrinsics  # (4, 4)
        camera_center = torch.inverse(extrinsics)[:3, 3]  # (3,)

        # FoV
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        fov_x = 2 * torch.atan(0.5 / fx).item()
        fov_y = 2 * torch.atan(0.5 / fy).item()

        # 3DGS Special: transposed matrix
        world_view_transform = extrinsics.T.contiguous()  # (4, 4)
        full_proj_transform = mvp.T.contiguous()  # (4, 4)

        return GaussianCameraData(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            projection=projection,
            mvp=mvp,
            camera_center=camera_center,
            fov=(fov_x, fov_y),
            image_height=resolution * ssaa,
            image_width=resolution * ssaa,
            world_view_transform=world_view_transform,
            full_proj_transform=full_proj_transform,
        )

    # ========== Stage 3: Geometry Processing ==========

    def _process_geometry(self, geometry: Gaussian, camera_data: CameraData) -> Gaussian:
        """Gaussian No preprocessing required"""
        return geometry

    # ========== Stage 4: Rasterization Core ==========

    def _rasterize_core(
        self,
        processed_geometry: Gaussian,
        camera_data: GaussianCameraData,
    ) -> RasterOutput:
        """
        call 3DGS rasterization

        Args:
            processed_geometry: Gaussian express
            camera_data: camera data

        Returns:
            RasterOutput: Rasterized results
        """
        # Perform rasterization
        result = gaussian_rasterize(
            camera_data=camera_data,
            gaussian=processed_geometry,
            pipe_config=self.pipe_config,
            bg_color=self._bg_color,
            override_color=self._colors_overwrite,
        )

        # Packed as RasterOutput
        return RasterOutput(
            rast=result,  # Store complete results
            depth_buffer=torch.zeros(camera_data.image_height, camera_data.image_width, device=self.device),  # 3DGS No depth output
            primitive_id=torch.zeros(camera_data.image_height, camera_data.image_width, device=self.device, dtype=torch.long),
        )

    # ========== Stage 5: Attribute Interpolation ==========

    def _interpolate_attributes(
        self,
        raster_output: RasterOutput,
        geometry: Gaussian,
        camera_data: CameraData,
        return_types: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Extract rendering results

        3DGS The rasterization has already completed the attribute interpolation. Here we only need to extract
        """
        result: GaussianRasterResult = raster_output.rast

        # color image: (3, H, W) -> (H, W, 3)
        color = result.rendered_image.permute(1, 2, 0)  # (H, W, 3)

        # 3DGS Not output directly mask, use all 1
        H, W = color.shape[:2]
        mask = torch.ones(H, W, device=self.device)  # (H, W)
        alpha = mask  # (H, W)

        # Depth of occupancy
        depth = torch.zeros(H, W, device=self.device)  # (H, W)

        return {
            'color': color,
            'mask': mask,
            'alpha': alpha,
            'depth': depth,
        }

    # ========== Extended interface ==========

    def render_with_color_override(
        self,
        geometry: Gaussian,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        colors_overwrite: torch.Tensor,
        return_types: List[str] = None,
    ) -> RenderOutput:
        """
        Render with overlay color

        Args:
            geometry: Gaussian express
            extrinsics: (4, 4) Camera external parameters
            intrinsics: (3, 3) Camera internal parameters
            colors_overwrite: (N, 3) Overlay color
            return_types: Return type list

        Returns:
            RenderOutput: Rendering results
        """
        self._colors_overwrite = colors_overwrite
        result = self.render(geometry, extrinsics, intrinsics, return_types)
        self._colors_overwrite = None
        return result

    # ========== Empty output ==========

    def _get_empty_output(self) -> RenderOutput:
        """Return empty Gaussian The output of"""
        res = self.config.resolution
        device = self.device
        return RenderOutput(
            depth=torch.zeros(res, res, device=device),
            alpha=torch.zeros(res, res, device=device),
            mask=torch.zeros(res, res, device=device),
            color=torch.zeros(res, res, 3, device=device),
        )
