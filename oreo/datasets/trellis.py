from utils3d.torch.transforms import intrinsics_from_fov_xy

import glob
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import utils3d
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from .utils import build_mvp_from_w2c


def compute_adaptive_distance(
    fov_deg: float,
    fill_ratio: float = 0.9,
    object_half_size: float = 0.5
) -> float:
    """
    According to FOV Calculate the camera distance so that the object occupies a specified proportion of the field of view.

    TRELLIS in the training data mesh be normalized to [-0.5, 0.5]^3 within the range.

    Args:
        fov_deg: Field of view (degrees)
        fill_ratio: The proportion of the object in the field of view (0.0~1.0), such as 0.9 Indicates that the object occupies 90% vision
        object_half_size: Object half size (TRELLIS After normalization, it is 0.5)

    Returns:
        distance from camera to origin r

    Formula derivation:
        half height visible = r × tan(FOV/2)
        object half size / half height visible = fill_ratio
        => r = object_half_size / (fill_ratio × tan(FOV/2))
    """
    fov_rad = math.radians(fov_deg)
    tan_half_fov = math.tan(fov_rad / 2)
    r = object_half_size / (fill_ratio * tan_half_fov)
    return r


def build_trellis_intrinsics(fovy_deg: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    fov_tensor = torch.tensor([math.radians(fovy_deg)], device=device, dtype=dtype)  # [1]
    intrinsics = intrinsics_from_fov_xy(fov_tensor, fov_tensor)[0]  # [3,3]
    return intrinsics  # [3,3]


def intrinsics_to_projection(
    intrinsics: torch.Tensor,
    near: float = 1.0,
    far: float = 100.0,
) -> torch.Tensor:
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]  # [], []
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]  # [], []
    proj = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)  # [4,4]
    proj[0, 0] = 2 * fx  # []
    proj[1, 1] = 2 * fy  # []
    proj[0, 2] = 2 * cx - 1  # []
    proj[1, 2] = -2 * cy + 1  # []
    proj[2, 2] = far / (far - near)  # []
    proj[2, 3] = near * far / (near - far)  # []
    proj[3, 2] = 1.0  # []
    return proj  # [4,4]


class BaseImageDatasetTrellis(Dataset):
    """Trellis variant dataset (remove threestudio / default dependency)"""

    def __init__(self, cfg: "TrellisDataConfig", split: str):
        self.cfg = cfg
        self.split = split
        self.image_paths = self._load_image_paths()  # [N]

    def _load_image_paths(self) -> List[str]:
        """from eval_image_path or image_dataset_dir Read image list"""
        image_paths: List[str] = []

        # eval mode first eval_image_path
        if self.split == "test" and self.cfg.eval_image_path:
            if os.path.exists(self.cfg.eval_image_path):
                if os.path.isfile(self.cfg.eval_image_path):
                    image_paths = [self.cfg.eval_image_path]
                else:
                    extensions = list(set(["png", "jpg", "jpeg", self.cfg.image_file_extension]))
                    for ext in extensions:
                        image_paths.extend(glob.glob(os.path.join(self.cfg.eval_image_path, f"*.{ext}")))

        # only in train/val or not provided eval_image_path used when image_dataset_dir(Don't do it twice fallback)
        if not image_paths and self.cfg.image_dataset_dir:
            image_dir = self.cfg.image_dataset_dir
            if os.path.exists(image_dir):
                extensions = list(set(["png", "jpg", "jpeg", self.cfg.image_file_extension]))
                for ext in extensions:
                    image_paths.extend(glob.glob(os.path.join(image_dir, f"*.{ext}")))

        if not image_paths:
            raise FileNotFoundError(f"No images found under eval_image_path={self.cfg.eval_image_path} or image_dataset_dir={self.cfg.image_dataset_dir}")

        return sorted(list(set(image_paths)))  # [N]

    def compute_views_train(self, num_views: int) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        Training samples:yaw/pitch/r/fov from each range Random sampling.

        Returns:
            yaws_deg: [V] yaw angle (degree)
            pitch_deg: [V] pitch angle (degree)
            r: float camera distance
            fov: float field of view (degree)
        """
        train_cfg = self.cfg.train
        if train_cfg.yaw_range is None or train_cfg.pitch_range is None or train_cfg.r_range is None or train_cfg.fov_range is None:
            raise ValueError("train camera ranges must be provided: yaw_range, pitch_range, r_range, fov_range")

        yaw_min, yaw_max = train_cfg.yaw_range
        pitch_min, pitch_max = train_cfg.pitch_range
        yaws_deg = torch.rand(num_views) * (yaw_max - yaw_min) + yaw_min  # [V]
        pitch_deg = torch.rand(num_views) * (pitch_max - pitch_min) + pitch_min  # [V]
        fov = float(np.random.uniform(*train_cfg.fov_range))  # float

        # When adaptive distance is turned on,r by fov Determine; otherwise sample randomly within the range.
        if train_cfg.adaptive_distance.enabled:
            r = compute_adaptive_distance(fov, train_cfg.adaptive_distance.fill_ratio)
        else:
            r = float(np.random.uniform(*train_cfg.r_range))  # float

        return yaws_deg, pitch_deg, r, fov

    def compute_views_eval(self, num_views: int) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        Evaluation sampling:yaw The interval is sampled in uniform steps, and the remaining parameters are taken from the midpoint of the range (deterministic).

        Returns:
            yaws_deg: [V] yaw angle (degree)
            pitch_deg: [V] pitch angle (degree)
            r: float camera distance
            fov: float field of view (degree)
        """
        eval_cfg = self.cfg.eval
        if eval_cfg.yaw_range is None or eval_cfg.pitch_range is None or eval_cfg.r_range is None or eval_cfg.fov_range is None:
            raise ValueError("eval camera ranges must be provided: yaw_range, pitch_range, r_range, fov_range")

        yaw_min, yaw_max = eval_cfg.yaw_range
        yaws_deg = torch.linspace(float(yaw_min), float(yaw_max), num_views)  # [V]

        pitch_min, pitch_max = eval_cfg.pitch_range
        pitch_val = 0.5 * float(pitch_min + pitch_max)  # float
        pitch_deg = torch.full((num_views,), pitch_val, dtype=torch.float32)  # [V]

        fov_min, fov_max = eval_cfg.fov_range
        fov = 0.5 * float(fov_min + fov_max)  # float

        # When adaptive distance is turned on,r by fov decide; otherwise take r_range Midpoint to ensure assessment certainty.
        if eval_cfg.adaptive_distance.enabled:
            r = compute_adaptive_distance(fov, eval_cfg.adaptive_distance.fill_ratio)
        else:
            r_min, r_max = eval_cfg.r_range
            r = 0.5 * float(r_min + r_max)  # float

        return yaws_deg, pitch_deg, r, fov

    def _build_camera(
        self,
        fovy: float,
        yaw_deg: float,
        pitch_deg: float,
        distance: float,
        device: torch.device,
        width: int,
        height: int,
    ) -> Dict[str, torch.Tensor]:

        # with TRELLIS render_utils.py in yaw_pitch_r_fov_to_extrinsics_intrinsics Alignment:
        # yaw, pitch in radians
        # orig = [sin(yaw)cos(pitch), cos(yaw)cos(pitch), sin(pitch)] * r
        # extr = look_at(orig, 0, up=[0,0,1])  # Z-up coordinate system
        #
        # yaw=0 When the camera is +Y axis direction
        yaw = torch.deg2rad(torch.tensor(yaw_deg, dtype=torch.float32, device=device))  # []
        pitch = torch.deg2rad(torch.tensor(pitch_deg, dtype=torch.float32, device=device))  # []
        fovy_rad = torch.deg2rad(torch.tensor(fovy, dtype=torch.float32, device=device))  # []

        # Calculate camera position (origin)
        # with TRELLIS render_utils.py consistent: [sin(yaw)*cos(pitch), cos(yaw)*cos(pitch), sin(pitch)]
        orig = torch.stack([
            torch.sin(yaw) * torch.cos(pitch),  # []
            torch.cos(yaw) * torch.cos(pitch),  # []
            torch.sin(pitch),  # []
        ]).to(device) * distance  # [3]

        # Extrinsics (World-to-Camera)
        # with TRELLIS render_utils.py consistent: up = [0, 0, 1] (Z-up)
        w2c = utils3d.torch.extrinsics_look_at(
            orig,  # [3]
            torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32),  # [3]
            torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32),  # [3]
        )  # [4,4]

        c2w = torch.inverse(w2c)  # [4,4]
        camera_position = c2w[:3, 3]  # [3]

        # Intrinsics (Normalized)
        # Trellis uses square fov for intrinsics calculation in render_utils
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(fovy_rad, fovy_rad)  # [3,3]

        # Projection Matrix
        # Use local intrinsics_to_projection; near/far relax with blender_script consistent
        proj = intrinsics_to_projection(
            intrinsics,  # [3,3]
            near=1.0,
            far=100.0,
        )  # [4,4]

        mvp = build_mvp_from_w2c(w2c.unsqueeze(0), proj.unsqueeze(0)).squeeze(0)  # [4,4]

        return {
            "c2w_matrix": c2w,  # [4,4]
            "w2c_matrix": w2c,  # [4,4]
            "intrinsics": intrinsics,  # [3,3]
            "camera_positions": camera_position,  # [3]
            "mvp_matrix": mvp,  # [4,4]
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_path = self.image_paths[index]  # str
        pil_image = Image.open(image_path).convert("RGBA")  # PIL

        is_eval = self.split in ("test", "val")
        num_views = self.cfg.eval.n_view if is_eval else self.cfg.train.n_view  # int
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # device

        # press split Choose a sampling strategy:train=random,eval=uniform step size+Deterministic parameters.
        if is_eval:
            yaws_deg, pitch_deg, r, fov = self.compute_views_eval(num_views)  # [V], [V], float, float
        else:
            yaws_deg, pitch_deg, r, fov = self.compute_views_train(num_views)  # [V], [V], float, float

        cameras = [
            self._build_camera(
                fovy=fov,
                yaw_deg=float(yaws_deg[i].item()),
                pitch_deg=float(pitch_deg[i].item()),
                distance=r,
                device=device,
                width=self.cfg.width,
                height=self.cfg.height,
            )
            for i in range(num_views)
        ]  # list[len=V]

        c2w = torch.stack([cam["c2w_matrix"] for cam in cameras])  # [V,4,4]
        w2c = torch.stack([cam["w2c_matrix"] for cam in cameras])  # [V,4,4]
        intrinsics = torch.stack([cam["intrinsics"] for cam in cameras])  # [V,3,3]
        mvp = torch.stack([cam["mvp_matrix"] for cam in cameras])  # [V,4,4]
        positions = torch.stack([cam["camera_positions"] for cam in cameras]).unsqueeze(1)  # [V,1,3]

        return {
            "index": index,
            "name": os.path.basename(image_path),
            "paths": image_path,
            "image_pils": pil_image,
            "num_views": num_views,
            "c2w": c2w,
            "w2c": w2c,
            "intrinsics": intrinsics,
            "mvp": mvp,
            "positions": positions,
        }

    def __len__(self) -> int:
        return len(self.image_paths)

    @staticmethod
    def collate(batch) -> Dict[str, Any]:
        if not batch:
            return {}
        image_pils = [item["image_pils"] for item in batch]  # list[len=B]
        batch_no_img = [{k: v for k, v in item.items() if k != "image_pils"} for item in batch]  # list[len=B]
        collated = torch.utils.data.default_collate(batch_no_img)  # dict
        collated["image_pils"] = image_pils  # list[len=B]
        return collated


# === Lightweight configuration and DataModule(only train / eval) ===
# Use TRELLIS native naming: yaw, pitch, r, fov


@dataclass
class AdaptiveDistanceConfig:
    """Adaptive camera distance configuration"""
    enabled: bool = False                 # Whether to enable adaptive distance (based on FOV Calculate r)
    fill_ratio: float = 0.9               # The proportion of the object in the field of view (0.0~1.0)


@dataclass
class TrellisCameraTrainConfig:
    """Camera parameter configuration during training"""
    n_view: int                           # Number of views during training
    yaw_range: List[float]                # yaw Sampling range (degree)
    pitch_range: List[float]              # pitch Sampling range (degree)
    r_range: List[float]                  # Camera distance range (adaptive_distance.enabled=False when used)
    fov_range: List[float]                # Field of view range (degree)
    adaptive_distance: AdaptiveDistanceConfig  # Adaptive distance configuration


@dataclass
class TrellisCameraEvalConfig:
    """Camera parameter configuration during evaluation (used uniformly range)"""
    n_view: int                           # Number of viewing angles during evaluation
    yaw_range: List[float]                # yaw Sampling range (degree)
    pitch_range: List[float]              # pitch Sampling range (degree)
    r_range: List[float]                  # Camera distance range (adaptive_distance.enabled=False when used)
    fov_range: List[float]                # Field of view range (degree)
    adaptive_distance: AdaptiveDistanceConfig  # Adaptive distance configuration


@dataclass
class TrellisDataConfig:
    batch_size: int = 1
    eval_batch_size: int = 1
    width: int = 512
    height: int = 512
    image_dataset_dir: str = "test_images"
    image_file_extension: str = "png"
    eval_image_path: Optional[str] = None
    # Separate camera configuration
    train: Optional[TrellisCameraTrainConfig] = None
    eval: Optional[TrellisCameraEvalConfig] = None

    def __post_init__(self):
        if self.train is None:
            raise ValueError("TrellisDataConfig.train must be provided")
        if self.eval is None:
            raise ValueError("TrellisDataConfig.eval must be provided")


class TrellisDataModule:
    def __init__(self, cfg: TrellisDataConfig, num_replicas: int = 1, rank: int = 0):
        self.cfg = cfg
        self.num_replicas = num_replicas
        self.rank = rank
        self.train_dataset: Optional[BaseImageDatasetTrellis] = None
        self.eval_dataset: Optional[BaseImageDatasetTrellis] = None
        self.train_sampler: Optional[DistributedSampler] = None
        self.eval_sampler: Optional[DistributedSampler] = None

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_dataset = BaseImageDatasetTrellis(self.cfg, "train")
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.num_replicas,
                rank=self.rank,
                shuffle=True,
                drop_last=True,
            )
        if stage in (None, "eval", "test", "predict"):
            # Use uniformly test split as an assessment
            self.eval_dataset = BaseImageDatasetTrellis(self.cfg, "test")
            self.eval_sampler = DistributedSampler(
                self.eval_dataset,
                num_replicas=self.num_replicas,
                rank=self.rank,
                shuffle=False,
                drop_last=False,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            sampler=self.train_sampler,
            num_workers=0,
            persistent_workers=False,
            collate_fn=self.train_dataset.collate if self.train_dataset else None,
        )

    def eval_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.cfg.eval_batch_size,
            sampler=self.eval_sampler,
            num_workers=0,
            persistent_workers=False,
            collate_fn=self.eval_dataset.collate if self.eval_dataset else None,
        )
