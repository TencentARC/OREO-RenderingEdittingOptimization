"""
Renderer base class - Define a unified rendering pipeline interface

Rendering pipeline 7 stage:
    Stage 1: Input Preparation (Input preparation)
    Stage 2: Camera Transform (camera transformation)
    Stage 3: Geometry Processing (Geometry processing)
    Stage 4: Rasterization Core (rasterization/rendering core)
    Stage 5: Attribute Interpolation (Property interpolation/sampling)
    Stage 6: Post-processing (Post-processing)
    Stage 7: Output Assembly (Output assembly)

Usage example:
    class MyRenderer(BaseRenderer):
        def _rasterize_core(self, processed_geometry, camera_data):
            # Implement rasterization core logic
            ...

    renderer = MyRenderer(RenderConfig(resolution=512))
    output = renderer.render(geometry, extrinsics, intrinsics)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict


# ============================================================================
# Data structure definition
# ============================================================================

@dataclass
class RenderConfig:
    """
    Renderer configuration

    Attributes:
        resolution: Output image resolution (Pixel)
        near: Near clipping plane distance
        far: far clipping plane distance
        ssaa: Oversampling anti-aliasing multiple (1=No oversampling)
        bg_color: Background color, support:
            - float: single gray value
            - Tuple[float, float, float]: RGB value
            - 'random': Random black and white background
            - 'envmap': Sample from environment map
    """
    resolution: int = 512
    near: float = 0.1
    far: float = 10.0
    ssaa: int = 1
    bg_color: Union[float, Tuple[float, float, float], str] = 0.0


@dataclass
class CameraData:
    """
    Camera data container

    Attributes:
        extrinsics: (4, 4) W2C Camera external parameter matrix (OpenCV Agreement)
        intrinsics: (3, 3) Camera internal parameter matrix (normalized to [0, 1])
        projection: (4, 4) OpenGL projection matrix
        mvp: (4, 4) Model-View-Projection matrix
        camera_center: (3,) Camera world coordinates
        fov: Tuple[float, float] (fov_x, fov_y) radians
        rays_o: Optional[(H, W, 3)] ray origin
        rays_d: Optional[(H, W, 3)] Ray direction
    """
    extrinsics: torch.Tensor
    intrinsics: torch.Tensor
    projection: torch.Tensor
    mvp: torch.Tensor
    camera_center: torch.Tensor
    fov: Tuple[float, float]
    rays_o: Optional[torch.Tensor] = None
    rays_d: Optional[torch.Tensor] = None


@dataclass
class RasterOutput:
    """
    Rasterized output container

    Attributes:
        rast: Rasterized results (Depends on the specific implementation)
        depth_buffer: (H, W) depth buffer
        primitive_id: (H, W) primitive ID
        barycentric: Optional[(H, W, 3)] Center of gravity coordinates
    """
    rast: Any
    depth_buffer: torch.Tensor
    primitive_id: torch.Tensor
    barycentric: Optional[torch.Tensor] = None


@dataclass
class RenderOutput:
    """
    Render output container

    Dimension specification:
        - 2D Properties (scalar): (H, W) - depth, alpha, mask
        - 3D Properties (vector): (H, W, 3) - normal, color, shaded

    Attributes:
        depth: (H, W) Depth map
        alpha: (H, W) Transparency/Coverage
        mask: (H, W) binary mask
        normal: Optional[(H, W, 3)] normal map (Visual format [0, 1])
        color: Optional[(H, W, 3)] color map
        shaded: Optional[(H, W, 3)] PBR Coloring results
        extras: Dict[str, Tensor] Other custom properties
    """
    depth: torch.Tensor
    alpha: torch.Tensor
    mask: torch.Tensor
    normal: Optional[torch.Tensor] = None
    color: Optional[torch.Tensor] = None
    shaded: Optional[torch.Tensor] = None
    extras: Dict[str, torch.Tensor] = field(default_factory=dict)

    def to_edict(self) -> edict:
        """Convert to edict Format (Compatible with old interfaces)"""
        result = edict(
            depth=self.depth,
            alpha=self.alpha,
            mask=self.mask,
        )
        if self.normal is not None:
            result.normal = self.normal
        if self.color is not None:
            result.color = self.color
        if self.shaded is not None:
            result.shaded = self.shaded
        result.update(self.extras)
        return result


# ============================================================================
# Utility function
# ============================================================================

def intrinsics_to_projection(
    intrinsics: torch.Tensor,  # (3, 3)
    near: float,
    far: float,
) -> torch.Tensor:
    """
    OpenCV Internal parameter matrix → OpenGL projection matrix

    Args:
        intrinsics: (3, 3) Normalized camera internal parameters (fx, fy, cx, cy in [0, 1])
        near: near clipping plane
        far: far clipping plane

    Returns:
        projection: (4, 4) OpenGL projection matrix
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]  # scalar
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]  # scalar

    proj = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)  # (4, 4)
    proj[0, 0] = 2 * fx
    proj[1, 1] = 2 * fy
    proj[0, 2] = 2 * cx - 1
    proj[1, 2] = -2 * cy + 1
    proj[2, 2] = far / (far - near)
    proj[2, 3] = near * far / (near - far)
    proj[3, 2] = 1.0

    return proj  # (4, 4)


def depth_to_normal(
    depth: torch.Tensor,       # (H, W)
    intrinsics: torch.Tensor,  # (3, 3)
    mask: Optional[torch.Tensor] = None,  # (H, W)
) -> torch.Tensor:
    """
    Estimating camera space normals from depth map

    Args:
        depth: (H, W) Depth map
        intrinsics: (3, 3) Normalized camera internal parameters
        mask: (H, W) optional mask

    Returns:
        normal: (H, W, 3) Camera space normal, positive toward the camera
    """
    H, W = depth.shape[-2:]
    depth = depth.reshape(H, W)  # (H, W)
    device = depth.device

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # pixel grid
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )  # (H, W), (H, W)

    # Backproject to camera space
    z = depth  # (H, W)
    x = (x_grid / W - cx) * z / fx  # (H, W)
    y = (y_grid / H - cy) * z / fy  # (H, W)

    # Central difference calculation gradient
    dx = torch.zeros(H, W, 3, device=device)  # (H, W, 3)
    dy = torch.zeros(H, W, 3, device=device)  # (H, W, 3)

    dx[:, 1:-1, 0] = (x[:, 2:] - x[:, :-2]) / 2
    dx[:, 1:-1, 1] = (y[:, 2:] - y[:, :-2]) / 2
    dx[:, 1:-1, 2] = (z[:, 2:] - z[:, :-2]) / 2

    dy[1:-1, :, 0] = (x[2:, :] - x[:-2, :]) / 2
    dy[1:-1, :, 1] = (y[2:, :] - y[:-2, :]) / 2
    dy[1:-1, :, 2] = (z[2:, :] - z[:-2, :]) / 2

    # Normal = dy × dx
    normal = torch.linalg.cross(dy, dx)  # (H, W, 3)
    normal = F.normalize(normal, dim=-1).reshape(H, W, 3)  # (H, W, 3)

    # Make sure the normal is facing the camera (z < 0)
    normal = torch.where(normal[..., 2:3] > 0, -normal, normal)  # (H, W, 3)

    if mask is not None:
        mask = mask.reshape(H, W)  # (H, W)
        normal = normal * mask[..., None]  # (H, W, 3)

    return normal  # (H, W, 3)


def camera_normal_to_vis(normal: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Convert camera space normals to visualization format

    Args:
        normal: (H, W, 3) camera space normal (towards camera z < 0)
        mask: (H, W) optional mask

    Returns:
        normal_vis: (H, W, 3) Visual format [0, 1]
    """
    normal_vis = -normal * 0.5 + 0.5  # (H, W, 3)
    if mask is not None:
        bg = torch.tensor([0.5, 0.5, 1.0], device=normal.device)  # neutral blue background
        normal_vis = normal_vis * mask[..., None] + bg * (1 - mask[..., None])  # (H, W, 3)
    return normal_vis  # (H, W, 3)


def ssaa_downsample(
    img: torch.Tensor,  # (C, H*ssaa, W*ssaa) or (H*ssaa, W*ssaa, C)
    target_size: int,
    channel_first: bool = True,
) -> torch.Tensor:
    """
    SSAA Downsampling

    Args:
        img: high resolution images
        target_size: target resolution
        channel_first: Whether channel-first Format

    Returns:
        downsampled image
    """
    if not channel_first:
        img = img.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

    img = F.interpolate(
        img[None],  # (1, C, H, W)
        size=(target_size, target_size),
        mode='bilinear',
        align_corners=False,
        antialias=True
    ).squeeze(0)  # (C, H, W)

    if not channel_first:
        img = img.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)

    return img


# ============================================================================
# Renderer base class
# ============================================================================

class BaseRenderer(ABC):
    """
    Renderer abstract base class

    defines a unified 7 staged rendering pipeline:
        1. prepare_inputs: Input preparation
        2. compute_camera_data: camera transformation
        3. process_geometry: Geometry processing
        4. rasterize_core: Rasterization core (abstract method)
        5. interpolate_attributes: Property interpolation
        6. post_process: Post-processing
        7. assemble_output: Output assembly

    Subclasses must implement:
        - _rasterize_core: Core rasterization logic
        - _interpolate_attributes: Property interpolation logic

    Optional override for subclasses:
        - _process_geometry: Custom geometry processing
        - _post_process: Custom post-processing
    """

    def __init__(self, config: RenderConfig, device: str = 'cuda'):
        """
        Args:
            config: Render configuration
            device: computing equipment
        """
        self.config = config
        self.device = device
        self._bg_color: Optional[torch.Tensor] = None

    # ========== Stage 1: Input Preparation ==========

    def prepare_inputs(
        self,
        geometry: Any,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> Tuple[Any, torch.Tensor, torch.Tensor]:
        """
        Stage 1: Input preparation

        - Validate input
        - Initialize background color
        - Handle empty input

        Args:
            geometry: Geometric representation (Mesh/Gaussian/Voxel)
            extrinsics: (4, 4) W2C Camera external parameters
            intrinsics: (3, 3) Camera internal parameters

        Returns:
            processed (geometry, extrinsics, intrinsics)
        """
        # Initialize background color
        self._init_bg_color(extrinsics.device)

        return geometry, extrinsics, intrinsics

    def _init_bg_color(self, device: torch.device) -> None:
        """Initialize background color"""
        bg = self.config.bg_color

        if isinstance(bg, str):
            if bg == 'random':
                import numpy as np
                val = 1.0 if np.random.rand() < 0.5 else 0.0
                self._bg_color = torch.full((3,), val, device=device)  # (3,)
            else:
                self._bg_color = torch.zeros(3, device=device)  # (3,)
        elif isinstance(bg, (int, float)):
            self._bg_color = torch.full((3,), float(bg), device=device)  # (3,)
        else:
            self._bg_color = torch.tensor(bg, device=device, dtype=torch.float32)  # (3,)

    def _is_empty_geometry(self, geometry: Any) -> bool:
        """Check if geometry is empty (Subclasses can override)"""
        return False

    def _get_empty_output(self) -> RenderOutput:
        """Return empty output"""
        res = self.config.resolution
        device = self.device
        return RenderOutput(
            depth=torch.zeros(res, res, device=device),  # (H, W)
            alpha=torch.zeros(res, res, device=device),  # (H, W)
            mask=torch.zeros(res, res, device=device),   # (H, W)
        )

    # ========== Stage 2: Camera Transform ==========

    def compute_camera_data(
        self,
        extrinsics: torch.Tensor,  # (4, 4)
        intrinsics: torch.Tensor,  # (3, 3)
    ) -> CameraData:
        """
        Stage 2: camera transformation

        Calculate the projection matrix,MVP, camera center, etc.

        Args:
            extrinsics: (4, 4) W2C Camera external parameters
            intrinsics: (3, 3) Camera internal parameters

        Returns:
            CameraData: Camera data container
        """
        near, far = self.config.near, self.config.far

        # projection matrix
        projection = intrinsics_to_projection(intrinsics, near, far)  # (4, 4)

        # MVP
        mvp = projection @ extrinsics  # (4, 4)

        # camera center (world coordinates)
        camera_center = torch.inverse(extrinsics)[:3, 3]  # (3,)

        # FoV
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        fov_x = 2 * torch.atan(0.5 / fx).item()
        fov_y = 2 * torch.atan(0.5 / fy).item()

        return CameraData(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            projection=projection,
            mvp=mvp,
            camera_center=camera_center,
            fov=(fov_x, fov_y),
        )

    # ========== Stage 3: Geometry Processing ==========

    def process_geometry(
        self,
        geometry: Any,
        camera_data: CameraData,
    ) -> Any:
        """
        Stage 3: Geometry processing

        The default implementation returns the original geometry directly, and subclasses can override it.

        Args:
            geometry: original geometric representation
            camera_data: camera data

        Returns:
            Processed geometric data
        """
        return self._process_geometry(geometry, camera_data)

    def _process_geometry(self, geometry: Any, camera_data: CameraData) -> Any:
        """Geometry processing implementation (Subclasses can override)"""
        return geometry

    # ========== Stage 4: Rasterization Core ==========

    def rasterize(
        self,
        processed_geometry: Any,
        camera_data: CameraData,
    ) -> RasterOutput:
        """
        Stage 4: rasterization

        Args:
            processed_geometry: Processed geometric data
            camera_data: camera data

        Returns:
            RasterOutput: Rasterized results
        """
        return self._rasterize_core(processed_geometry, camera_data)

    @abstractmethod
    def _rasterize_core(
        self,
        processed_geometry: Any,
        camera_data: CameraData,
    ) -> RasterOutput:
        """
        Rasterization core implementation (Subclasses must implement)

        Different renderers use different rasterization techniques:
        - Mesh: nvdiffrast
        - Gaussian: diff_gaussian_rasterization
        - Voxel: o_voxel or soft z-buffer
        """
        pass

    # ========== Stage 5: Attribute Interpolation ==========

    def interpolate_attributes(
        self,
        raster_output: RasterOutput,
        geometry: Any,
        camera_data: CameraData,
        return_types: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Stage 5: Property interpolation

        Args:
            raster_output: Rasterized results
            geometry: original geometry
            camera_data: camera data
            return_types: List of properties to be returned

        Returns:
            Dict[str, Tensor]: Interpolated properties
        """
        return self._interpolate_attributes(raster_output, geometry, camera_data, return_types)

    @abstractmethod
    def _interpolate_attributes(
        self,
        raster_output: RasterOutput,
        geometry: Any,
        camera_data: CameraData,
        return_types: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Property interpolation implementation (Subclasses must implement)"""
        pass

    # ========== Stage 6: Post-processing ==========

    def post_process(
        self,
        raw_attrs: Dict[str, torch.Tensor],
        camera_data: CameraData,
    ) -> Dict[str, torch.Tensor]:
        """
        Stage 6: Post-processing

        - SSAA Downsampling
        - normal space transformation
        - Tone mapping
        - Gamma Correction

        Args:
            raw_attrs: original attribute
            camera_data: camera data

        Returns:
            Processed attributes
        """
        result = {}
        res = self.config.resolution
        ssaa = self.config.ssaa

        for key, value in raw_attrs.items():
            # SSAA Downsampling
            if ssaa > 1 and value.numel() > 0:
                if value.dim() == 2:  # (H, W)
                    value = ssaa_downsample(value[None], res, channel_first=True).squeeze(0)  # (H, W)
                elif value.dim() == 3:  # (H, W, C)
                    value = ssaa_downsample(value, res, channel_first=False)  # (H, W, C)

            result[key] = value

        # Subclasses extendable post-processing
        result = self._post_process(result, camera_data)

        return result

    def _post_process(
        self,
        attrs: Dict[str, torch.Tensor],
        camera_data: CameraData,
    ) -> Dict[str, torch.Tensor]:
        """Post-processing extension points (Subclasses can override)"""
        return attrs

    # ========== Stage 7: Output Assembly ==========

    def assemble_output(
        self,
        attrs: Dict[str, torch.Tensor],
    ) -> RenderOutput:
        """
        Stage 7: Output assembly

        Args:
            attrs: All properties

        Returns:
            RenderOutput: Unified output format
        """
        # Extract standard attributes
        depth = attrs.pop('depth', torch.zeros(self.config.resolution, self.config.resolution, device=self.device))
        alpha = attrs.pop('alpha', torch.zeros(self.config.resolution, self.config.resolution, device=self.device))
        mask = attrs.pop('mask', (alpha > 0.5).float())
        normal = attrs.pop('normal', None)
        color = attrs.pop('color', None)
        shaded = attrs.pop('shaded', None)

        return RenderOutput(
            depth=depth,
            alpha=alpha,
            mask=mask,
            normal=normal,
            color=color,
            shaded=shaded,
            extras=attrs,  # Put the remaining attributes into extras
        )

    # ========== Main rendering entrance ==========

    def render(
        self,
        geometry: Any,
        extrinsics: torch.Tensor,  # (4, 4)
        intrinsics: torch.Tensor,  # (3, 3)
        return_types: List[str] = None,
    ) -> RenderOutput:
        """
        Main rendering entrance

        execute complete 7 staged rendering pipeline

        Args:
            geometry: Geometric representation (Mesh/Gaussian/Voxel)
            extrinsics: (4, 4) W2C Camera external parameters
            intrinsics: (3, 3) Camera internal parameters
            return_types: List of properties to be returned

        Returns:
            RenderOutput: Rendering results
        """
        if return_types is None:
            return_types = ['depth', 'alpha', 'mask', 'normal']

        # Stage 1: Input Preparation
        geometry, extrinsics, intrinsics = self.prepare_inputs(geometry, extrinsics, intrinsics)

        # empty input check
        if self._is_empty_geometry(geometry):
            return self._get_empty_output()

        # Stage 2: Camera Transform
        camera_data = self.compute_camera_data(extrinsics, intrinsics)

        # Stage 3: Geometry Processing
        processed_geom = self.process_geometry(geometry, camera_data)

        # Stage 4: Rasterization Core
        raster_output = self.rasterize(processed_geom, camera_data)

        # Stage 5: Attribute Interpolation
        raw_attrs = self.interpolate_attributes(raster_output, processed_geom, camera_data, return_types)

        # Stage 6: Post-processing
        processed_attrs = self.post_process(raw_attrs, camera_data)

        # Stage 7: Output Assembly
        return self.assemble_output(processed_attrs)

    # ========== Compatible interface ==========

    def render_edict(
        self,
        geometry: Any,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        return_types: List[str] = None,
    ) -> edict:
        """
        Compatible with old interfaces, return edict
        """
        output = self.render(geometry, extrinsics, intrinsics, return_types)
        return output.to_edict()


# ============================================================================
# PBR Post-processing Mixin
# ============================================================================

class PBRPostProcessMixin:
    """
    PBR Post-processing mix-ins

    provide tone mapping and gamma Correction function
    """

    @staticmethod
    def aces_tonemapping(x: torch.Tensor) -> torch.Tensor:
        """ACES tone mapping"""
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        return torch.clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)

    @staticmethod
    def gamma_correction(x: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
        """Gamma correction"""
        return torch.clamp(x ** (1.0 / gamma), 0.0, 1.0)

    def apply_pbr_post_process(
        self,
        shaded: torch.Tensor,  # (H, W, 3)
        mask: torch.Tensor = None,  # (H, W)
        bg_color: torch.Tensor = None,  # (3,)
    ) -> torch.Tensor:
        """
        Application PBR Post-processing

        Args:
            shaded: (H, W, 3) Linear space shading results
            mask: (H, W) optional mask
            bg_color: (3,) Optional background color

        Returns:
            (H, W, 3) Post-processing results
        """
        # Tone mapping
        shaded = self.aces_tonemapping(shaded)

        # Gamma correction
        shaded = self.gamma_correction(shaded)

        # Background composition
        if mask is not None and bg_color is not None:
            shaded = shaded * mask[..., None] + bg_color * (1 - mask[..., None])

        return shaded


# ============================================================================
# Export
# ============================================================================

__all__ = [
    # data structure
    'RenderConfig',
    'CameraData',
    'RasterOutput',
    'RenderOutput',
    # Utility function
    'intrinsics_to_projection',
    'depth_to_normal',
    'camera_normal_to_vis',
    'ssaa_downsample',
    # base class
    'BaseRenderer',
    'PBRPostProcessMixin',
]
