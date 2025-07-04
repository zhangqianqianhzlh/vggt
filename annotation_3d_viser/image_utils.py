"""
图像处理工具模块
"""

import numpy as np
from typing import Optional


class ImageUtils:
    """图像处理工具类"""
    
    @staticmethod
    def draw_bbox_on_image(img: np.ndarray, bbox: Optional[tuple]) -> np.ndarray:
        """
        在图片上绘制边框
        
        Args:
            img: 输入图像 (H, W, 3)
            bbox: 边框坐标 (x1, y1, x2, y2) 或 None
            
        Returns:
            绘制边框后的图像
        """
        if bbox is None:
            return img
        
        img_with_bbox = img.copy()
        x1, y1, x2, y2 = bbox
        
        # 确保坐标在图片范围内
        h, w = img.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        # 绘制红色边框
        color = (255, 0, 0)
        thickness = 3
        
        # 画矩形框的四条边
        img_with_bbox[y1:y1+thickness, x1:x2+1] = color  # 上边
        img_with_bbox[y2-thickness+1:y2+1, x1:x2+1] = color  # 下边
        img_with_bbox[y1:y2+1, x1:x1+thickness] = color  # 左边
        img_with_bbox[y1:y2+1, x2-thickness+1:x2+1] = color  # 右边
        
        return img_with_bbox 