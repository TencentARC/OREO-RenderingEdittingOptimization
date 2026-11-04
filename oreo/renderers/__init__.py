"""
OREO renderer package. All renderers inherit BaseRenderer's 7-stage pipeline.

Stages: Input -> Camera -> Rasterize -> Attribute -> Shade -> Post -> Output.
"""

# Base classes and tools
from oreo.renderers.base_renderer import (
    # data structure
    RenderConfig,
    CameraData,
    RasterOutput,
    RenderOutput,
    # Utility function
    intrinsics_to_projection,
    depth_to_normal,
    camera_normal_to_vis,
    ssaa_downsample,
    # base class
    BaseRenderer,
    PBRPostProcessMixin,
)

# Mesh Renderer
from oreo.renderers.sparseflex_trellis import (
    TrellisMeshRasterizer,
    MeshRasterData,
)

# Gaussian Renderer - Conditional import (requires trellis module)
try:
    from oreo.renderers.gaussian_splatting_trellis import (
        GaussianRenderer,
        GaussianPipeConfig,
        GaussianCameraData,
        GaussianRasterResult,
        gaussian_rasterize,
    )
    _HAS_GAUSSIAN = True
except ImportError:
    _HAS_GAUSSIAN = False
    GaussianRenderer = None
    GaussianPipeConfig = None
    GaussianCameraData = None
    GaussianRasterResult = None
    gaussian_rasterize = None

__all__ = [
    # Base classes and tools
    'RenderConfig',
    'CameraData',
    'RasterOutput',
    'RenderOutput',
    'intrinsics_to_projection',
    'depth_to_normal',
    'camera_normal_to_vis',
    'ssaa_downsample',
    'BaseRenderer',
    'PBRPostProcessMixin',
    # Mesh
    'TrellisMeshRasterizer',
    'MeshRasterData',
    # Gaussian (Optional)
    'GaussianRenderer',
    'GaussianPipeConfig',
    'GaussianCameraData',
    'GaussianRasterResult',
    'gaussian_rasterize',
    # Voxel / Soft Voxel / Quad12 The renderer is not local yet trellis used in the process
]
