"""Trellis mesh renderer via nvdiffrast (BaseRenderer 7-stage pipeline)."""

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import nvdiffrast.torch as dr

from oreo.renderers.base_renderer import (
    BaseRenderer,
    RenderConfig,
    CameraData,
    RasterOutput,
    RenderOutput,
)


@dataclass
class MeshRasterData:
    """Clip-space vertices and mesh raster intermediates."""
    vertices_clip: torch.Tensor  # (1, V, 4)
    vertices_cam: torch.Tensor   # (1, V, 4)
    faces: torch.Tensor          # (F, 3)
    mesh: Any


class TrellisMeshRasterizer(BaseRenderer):
    """nvdiffrast mesh rasterizer for Trellis decoded meshes."""

    def __init__(self, config: RenderConfig = None, device: str = "cuda"):
        """
        Args:
            config: rendering configuration,None Use default value when
            device: computing equipment
        """
        if config is None:
            config = RenderConfig(resolution=512, near=0.01, far=100.0, ssaa=1)
        super().__init__(config, device)

        # initialization nvdiffrast context
        self.glctx = dr.RasterizeCudaContext(device=device)

    # ========== Stage 1: Input Preparation ==========

    def _is_empty_geometry(self, geometry: Any) -> bool:
        """Check mesh Is it empty"""
        return geometry.vertices.shape[0] == 0 or geometry.faces.shape[0] == 0

    # ========== Stage 3: Geometry Processing ==========

    def _process_geometry(
        self,
        geometry: Any,  # MeshExtractResult
        camera_data: CameraData,
    ) -> MeshRasterData:
        """
        vertices transformed to clip space and camera space

        Args:
            geometry: MeshExtractResult, including vertices (V, 3), faces (F, 3)
            camera_data: camera data

        Returns:
            MeshRasterData: Transformed vertex data
        """
        # Vertex homogeneous coordinates
        vertices = geometry.vertices.unsqueeze(0)  # (1, V, 3)
        vertices_homo = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)  # (1, V, 4)

        # Clip Space: v @ MVP.T
        vertices_clip = torch.bmm(
            vertices_homo,
            camera_data.mvp.unsqueeze(0).transpose(-1, -2)
        )  # (1, V, 4)

        # Camera Space (for depth): v @ Extrinsics.T
        vertices_cam = torch.bmm(
            vertices_homo,
            camera_data.extrinsics.unsqueeze(0).transpose(-1, -2)
        )  # (1, V, 4)

        return MeshRasterData(
            vertices_clip=vertices_clip,
            vertices_cam=vertices_cam,
            faces=geometry.faces.int(),
            mesh=geometry,
        )

    # ========== Stage 4: Rasterization Core ==========

    def _rasterize_core(
        self,
        processed_geometry: MeshRasterData,
        camera_data: CameraData,
    ) -> RasterOutput:
        """
        nvdiffrast rasterization

        Args:
            processed_geometry: Transformed vertex data
            camera_data: camera data

        Returns:
            RasterOutput: Rasterized results
        """
        resolution = self.config.resolution
        ssaa = self.config.ssaa
        render_res = resolution * ssaa

        # rasterization
        rast, _ = dr.rasterize(
            self.glctx,
            processed_geometry.vertices_clip,
            processed_geometry.faces,
            (render_res, render_res)
        )  # (1, H, W, 4) - [u, v, z, triangle_id]

        return RasterOutput(
            rast=rast,
            depth_buffer=rast[..., 2],       # (1, H, W)
            primitive_id=rast[..., 3].long(),  # (1, H, W)
        )

    # ========== Stage 5: Attribute Interpolation ==========

    def _interpolate_attributes(
        self,
        raster_output: RasterOutput,
        geometry: Any,  # MeshRasterData
        camera_data: CameraData,
        return_types: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Property interpolation

        Args:
            raster_output: Rasterized results
            geometry: original mesh
            camera_data: camera data
            return_types: List of properties to be returned

        Returns:
            Dict[str, Tensor]: Interpolated properties
        """
        rast = raster_output.rast  # (1, H, W, 4)

        processed = geometry
        mesh = geometry.mesh
        v_clip = geometry.vertices_clip
        v_cam = geometry.vertices_cam
        faces = geometry.faces
        cam_rot = camera_data.extrinsics[:3, :3].contiguous()  # (3, 3)

        result = {}

        for attr_type in return_types:
            img = self._interpolate_single_attribute(
                attr_type, rast, v_clip, v_cam, mesh, faces, cam_rot
            )  # (1, H, W, C)

            # remove batch dimensions, converted to (H, W, C) or (H, W)
            img = img.squeeze(0)  # (H, W, C)
            if img.shape[-1] == 1:
                img = img.squeeze(-1)  # (H, W)

            result[attr_type] = img

        # Make sure there is alpha
        if 'alpha' not in result and 'mask' in result:
            result['alpha'] = result['mask']
        elif 'alpha' not in result:
            result['alpha'] = (rast[..., -1] > 0).float().squeeze(0)  # (H, W)

        return result

    def _interpolate_single_attribute(
        self,
        attr_type: str,
        rast: torch.Tensor,      # (1, H, W, 4)
        v_clip: torch.Tensor,    # (1, V, 4)
        v_cam: torch.Tensor,     # (1, V, 4)
        mesh: Any,
        faces: torch.Tensor,     # (F, 3)
        cam_rot: torch.Tensor,  # (3, 3)
    ) -> torch.Tensor:
        """
        Interpolate a single attribute

        Args:
            attr_type: Property type ('mask', 'depth', 'normal', 'color', 'normal_map')
            rast: Rasterized results
            v_clip: clip space vertex
            v_cam: camera space vertex
            mesh: original mesh
            faces: face index

        Returns:
            img: (1, H, W, C) Interpolation results
        """
        if attr_type == "mask":
            # Mask:triangle_id > 0
            img = dr.antialias(
                (rast[..., -1:] > 0).float(),
                rast, v_clip, faces
            )  # (1, H, W, 1)

        elif attr_type == "depth":
            # Depth:camera space z
            depth = v_cam[..., 2:3].contiguous()  # (1, V, 1)
            img = dr.interpolate(depth, rast, faces)[0]  # (1, H, W, 1)
            img = dr.antialias(img, rast, v_clip, faces)  # (1, H, W, 1)

        elif attr_type == "normal":
            # Surface normal interpolation
            # build per-face-vertex normal index
            face_normal_indices = torch.arange(
                mesh.faces.shape[0] * 3,
                device=self.device,
                dtype=torch.int
            ).reshape(-1, 3)  # (F, 3)

            normals_world = mesh.face_normal  # (F, 3) or (F, 3, 3)
            normals_cam = -(normals_world @ cam_rot.T)  # (F, 3) or (F, 3, 3)
            normals_cam = normals_cam.reshape(1, -1, 3)  # (1, F*3, 3)

            normals = dr.interpolate(
                normals_cam,
                rast,
                face_normal_indices,
            )[0]  # (1, H, W, 3)
            img = dr.antialias(normals, rast, v_clip, faces)  # (1, H, W, 3)
            # Convert to visualization scope [0, 1]
            img = (img + 1) / 2  # (1, H, W, 3)

            # Background blending (using mask)
            mask = dr.antialias(
                (rast[..., -1:] > 0).float(),
                rast,
                v_clip,
                faces
            )  # (1, H, W, 1)
            bg = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 1, 1, 3)  # (1, 1, 1, 3)
            img = img * mask + bg * (1 - mask)  # (1, H, W, 3)

        elif attr_type == "normal_map":
            # Vertex normal map
            if mesh.vertex_attrs is not None and mesh.vertex_attrs.shape[-1] >= 6:
                nm = mesh.vertex_attrs[:, 3:6].unsqueeze(0).contiguous()  # (1, V, 3)
                img = dr.interpolate(nm, rast, faces)[0]  # (1, H, W, 3)
                img = dr.antialias(img, rast, v_clip, faces)  # (1, H, W, 3)
            else:
                img = torch.zeros_like(rast[..., :3])  # (1, H, W, 3)

        elif attr_type == "color":
            # vertex color
            if mesh.vertex_attrs is not None and mesh.vertex_attrs.shape[-1] >= 3:
                color = mesh.vertex_attrs[:, :3].unsqueeze(0).contiguous()  # (1, V, 3)
                img = dr.interpolate(color, rast, faces)[0]  # (1, H, W, 3)
                img = dr.antialias(img, rast, v_clip, faces)  # (1, H, W, 3)
            else:
                img = torch.zeros_like(rast[..., :3])  # (1, H, W, 3)

        else:
            # Unknown property type
            img = torch.zeros_like(rast[..., :1])  # (1, H, W, 1)

        return img

    # ========== Empty output ==========

    def _get_empty_output(self) -> RenderOutput:
        """Return empty mesh The output of"""
        res = self.config.resolution
        device = self.device
        return RenderOutput(
            depth=torch.zeros(res, res, device=device),        # (H, W)
            alpha=torch.zeros(res, res, device=device),        # (H, W)
            mask=torch.zeros(res, res, device=device),         # (H, W)
            normal=torch.zeros(res, res, 3, device=device),    # (H, W, 3)
            color=torch.zeros(res, res, 3, device=device),     # (H, W, 3)
        )
