"""
3D目标检测和可视化工具包

主要功能：
1. 使用VGGT模型从多张图片生成3D重建
2. 使用Qwen模型检测图片中的目标物体
3. 将2D检测框映射到3D空间，生成水平对齐的3D包围框
4. 使用viser进行交互式3D可视化
5. 保存COLMAP格式的重建结果

作者：Facebook Research & 修改者
"""

from .object_detector import ObjectDetector
from .box_mapper import Box3DMapper
from .image_utils import ImageUtils
from .visualizer import ViserVisualizer
from .scene_aligner import SceneAligner
from .scene_renderer import Scene3DRenderer
from .utils import save_colmap_reconstruction, apply_sky_segmentation, save_multi_detection_results_json
from .main import main_pipeline

__version__ = "1.0.0"
__all__ = [
    "ObjectDetector",
    "Box3DMapper", 
    "ImageUtils",
    "ViserVisualizer",
    "SceneAligner",
    "Scene3DRenderer",
    "save_colmap_reconstruction",
    "apply_sky_segmentation",
    "save_multi_detection_results_json",
    "main_pipeline"
] 