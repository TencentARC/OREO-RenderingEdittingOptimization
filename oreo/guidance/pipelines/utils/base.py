"""
基础类定义。

包含:
- BaseStateTracker: 状态追踪器基类
"""

from dataclasses import dataclass
from typing import Any, List, Optional
import torch
from PIL import Image


@dataclass
class BaseStateTracker:
    """
    状态追踪器基类。
    
    所有 Guidance 方法的 Tracker（FlowEdit、CSD、SDS）都继承此类。
    
    共有属性：
        - height, width: 图像尺寸（用于 decode）
    
    子类应通过 Mixin 或直接实现提供：
        - target: 目标 latent（用于 loss 计算）
        - loss(): 主要 loss 计算方法（由 LossMixin 提供）
    """
    
    height: int = 0
    width: int = 0
    
    # =========================================================================
    # 共有方法
    # =========================================================================
    
    def _decode_latent(self, pipe: Any, latent: torch.Tensor) -> Optional[Image.Image]:
        """
        Decode 单个 latent 为图像。
        
        Args:
            pipe: Pipeline（需要 _decode_latent_to_image 方法）
            latent: [1, seq, C]
        
        Returns:
            PIL Image
        """
        with torch.no_grad():
            images = pipe._decode_latent_to_image(
                latent,
                self.height,
                self.width,
                "pil"
            )
            return images[0] if images else None
    
    def _concat_images_horizontal(self, *images: Image.Image) -> Optional[Image.Image]:
        """
        水平拼接多张图像。
        
        Args:
            *images: PIL Image 列表
        
        Returns:
            拼接后的 PIL Image
        """
        images = [img for img in images if img is not None]
        if not images:
            return None
        
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)
        
        combined = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in images:
            combined.paste(img, (x_offset, 0))
            x_offset += img.width
        
        return combined
