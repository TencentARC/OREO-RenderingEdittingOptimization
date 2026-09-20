"""
oreo renderer module

All renderers inherit from BaseRenderer, follow the unified 7 staged rendering pipeline:
    Stage 1: Input Preparation (Input preparation)
    Stage 2: Camera Transform (camera transformation)
    Stage 3: Geometry Processing (Geometry processing)
    Stage 4: Rasterization Core (rasterization/rendering core)
    Stage 5: Attribute Interpolation (Property interpolation/sampling)
    Stage 6: Post-processing (Post-processing)
    Stage 7: Output Assembly (Output assembly)

Renderer list:
    - BaseRenderer: abstract base class
    - TrellisMeshRasterizer: Mesh rasterizer (nvdiffrast)
    - GaussianRenderer: 3D Gaussian Splatting Renderer (need trellis module)
    - VoxelRenderer: Basic voxel renderer (o_voxel)
    - PbrVoxelRenderer: PBR Shaded voxel renderer
    - DiffVoxelRenderer: Differentiable voxel renderer (STE)
    - SoftVoxelRenderer: pure PyTorch Differentiable voxel renderer
    - Quad12NormalRenderer: 12-Quad normal renderer
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
