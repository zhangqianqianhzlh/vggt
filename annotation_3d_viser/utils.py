"""
工具函数模块
"""

import os
import glob
import json
import warnings
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import cv2
import trimesh
from tqdm.auto import tqdm

from visual_util import segment_sky, download_file_from_url
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track

from .config import DEFAULT_CONFIG

# 可选依赖
try:
    import onnxruntime
except ImportError:
    warnings.warn("onnxruntime not found. Sky segmentation may not work.")


def save_colmap_reconstruction(pred_dict: dict, image_folder: str, colmap_path: str, 
                             conf_thres_value: float = None, max_points_for_colmap: int = None):
    """
    保存COLMAP格式的重建结果
    
    Args:
        pred_dict: VGGT预测结果字典
        image_folder: 图片文件夹路径
        colmap_path: COLMAP输出路径
        conf_thres_value: 置信度阈值
        max_points_for_colmap: 最大点数
    """
    if conf_thres_value is None:
        conf_thres_value = DEFAULT_CONFIG["conf_thres_value"]
    if max_points_for_colmap is None:
        max_points_for_colmap = DEFAULT_CONFIG["max_points_for_colmap"]
    
    print(f"\n=== 保存COLMAP格式重建结果 - 尺寸调试 ===")
    
    # 解包预测结果
    images = pred_dict["images"]  # (S, 3, H, W)
    depth_map = pred_dict["depth"]  # (S, H, W, 1)
    depth_conf = pred_dict["depth_conf"]  # (S, H, W)
    extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
    intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)
    
    # 添加调试信息
    print(f"images 形状: {images.shape}")
    print(f"depth_map 形状: {depth_map.shape}")
    print(f"depth_conf 形状: {depth_conf.shape}")
    print(f"extrinsics_cam 形状: {extrinsics_cam.shape}")
    print(f"intrinsics_cam 形状: {intrinsics_cam.shape}")
    
    # 获取图片文件名
    image_files = sorted(glob.glob(os.path.join(image_folder, "*")))
    base_image_path_list = [os.path.basename(path) for path in image_files]
    
    # 生成3D点云
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
    
    # 从points_3d获取尺寸信息
    num_frames, height, width, _ = points_3d.shape
    print(f"从points_3d获取: num_frames={num_frames}, height={height}, width={width}")

    # COLMAP转换参数
    shared_camera = False  # 在前向推理模式下，不支持共享相机
    camera_type = "PINHOLE"  # 在前向推理模式下，只支持PINHOLE相机

    # 使用实际的图像尺寸，而不是固定的518x518
    actual_height, actual_width = depth_conf.shape[1], depth_conf.shape[2]
    image_size = np.array([actual_width, actual_height])  # 注意：COLMAP使用[width, height]格式

    print(f"使用的image_size: {image_size} (width={actual_width}, height={actual_height})")
    
    # 准备RGB颜色 - 保持与depth_conf相同的尺寸
    points_rgb = images.transpose(0, 2, 3, 1)  # (S, H, W, 3) - 保持原始尺寸350x518
    points_rgb = (points_rgb * 255).astype(np.uint8)

    print(f"points_rgb 形状: {points_rgb.shape}")
    print(f"depth_conf 形状: {depth_conf.shape}")
    
    # 创建像素坐标网格 (S, H, W, 3), 包含x, y坐标和帧索引
    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)
    
    # 应用置信度掩码
    conf_mask = depth_conf >= conf_thres_value
    # 最多写入max_points_for_colmap个3D点到colmap重建对象
    conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)
    
    points_3d = points_3d[conf_mask]
    points_xyf = points_xyf[conf_mask]
    points_rgb = points_rgb[conf_mask]
    
    print(f"有效3D点数量: {len(points_3d)}")
    print("转换为COLMAP格式...")
    
    # 转换为COLMAP格式
    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points_3d,
        points_xyf,
        points_rgb,
        extrinsics_cam,
        intrinsics_cam,
        image_size,
        shared_camera=shared_camera,
        camera_type=camera_type,
    )
    
    # 重命名和缩放相机参数（这里简化处理，不进行复杂的坐标变换）
    reconstruction = rename_colmap_recons_and_rescale_camera_simple(
        reconstruction,
        base_image_path_list,
        img_size=518,
        shared_camera=shared_camera,
    )
    
    # 保存重建结果
    print(f"保存重建结果到 {colmap_path}")
    os.makedirs(colmap_path, exist_ok=True)
    reconstruction.write(colmap_path)
    
    # 保存点云用于快速可视化
    point_cloud_path = os.path.join(colmap_path, "points.ply")
    trimesh.PointCloud(points_3d, colors=points_rgb).export(point_cloud_path)
    print(f"保存点云到 {point_cloud_path}")
    
    print("COLMAP格式保存完成!")


def rename_colmap_recons_and_rescale_camera_simple(
    reconstruction, image_paths, img_size, shared_camera=False
):
    """
    简化版的重命名和缩放相机参数函数
    """
    rescale_camera = True
    
    for pyimageid in reconstruction.images:
        # 重命名图像到原始名称
        pyimage = reconstruction.images[pyimageid]
        pyimage.name = image_paths[pyimageid - 1]
        
        if shared_camera:
            # 如果使用共享相机，所有图像共享同一个相机
            # 不需要再次缩放
            rescale_camera = False
    
    return reconstruction


def apply_sky_segmentation(conf: np.ndarray, image_folder: str) -> np.ndarray:
    """应用天空分割掩码"""
    S, H, W = conf.shape
    sky_masks_dir = image_folder.rstrip("/") + "_sky_masks"
    os.makedirs(sky_masks_dir, exist_ok=True)

    # 下载天空分割模型
    if not os.path.exists("skyseg.onnx"):
        print("Downloading skyseg.onnx...")
        download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx")

    skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
    image_files = sorted(glob.glob(os.path.join(image_folder, "*")))
    sky_mask_list = []

    print("Generating sky masks...")
    for i, image_path in enumerate(tqdm(image_files[:S])):
        image_name = os.path.basename(image_path)
        mask_filepath = os.path.join(sky_masks_dir, image_name)

        if os.path.exists(mask_filepath):
            sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
        else:
            sky_mask = segment_sky(image_path, skyseg_session, mask_filepath)

        if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
            sky_mask = cv2.resize(sky_mask, (W, H))

        sky_mask_list.append(sky_mask)

    sky_mask_array = np.array(sky_mask_list)
    sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)
    conf = conf * sky_mask_binary

    print("Sky segmentation applied successfully")
    return conf


def save_multi_detection_results_json(detection_results: Dict[str, Dict[str, Any]], 
                                    multi_bboxes: Dict[str, List[Optional[tuple]]], 
                                    image_files: List[str], 
                                    scene_center: np.ndarray, 
                                    output_path: str):
    """
    保存多个3D框的关键坐标和2D图片目标信息为JSON格式
    """
    # 创建输出数据结构
    detection_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_objects": len(detection_results),
            "object_names": list(detection_results.keys()),
            "total_images": len(image_files),
            "detection_method": "VGGT + Qwen2.5-VL",
            "coordinate_system": "world_coordinates"
        },
        
        # 多个3D包围框信息
        "3d_bounding_boxes": {},
        
        # 2D检测结果（按图片组织）
        "2d_detections": []
    }
    
    # 处理每个目标的3D包围框
    for object_name, box_info in detection_results.items():
        if box_info is not None:
            detection_data["3d_bounding_boxes"][object_name] = {
                "center": (box_info['center'] + scene_center).tolist(),     # [x, y, z] - 3个数字
                "size": box_info['size'].tolist(),                          # [w, l, h] - 3个数字
                "rotation_angle": float(box_info['rotation_angle'])         # 角度 - 1个数字
                # 总共只有7个数字！
            }
    
    # 按图片组织2D检测结果
    for i, image_file in enumerate(image_files):
        image_name = os.path.basename(image_file)
        
        image_detection = {
            "image_id": i,
            "image_name": image_name,
            "image_path": image_file,
            "objects": {}
        }
        
        # 处理该图片中的每个目标
        for object_name in detection_results.keys():
            if i < len(multi_bboxes[object_name]):
                bbox = multi_bboxes[object_name][i]
                
                if bbox is not None:
                    image_detection["objects"][object_name] = {
                        "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],  # [x1, y1, x2, y2]
                        "score": 1.0  # 默认置信度
                    }
                else:
                    image_detection["objects"][object_name] = None
        
        detection_data["2d_detections"].append(image_detection)
    
    # 保存JSON文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(detection_data, f, ensure_ascii=False, indent=2)
    
    print(f"多目标检测结果已保存到: {output_path}")
    print(f"- 检测到的目标数量: {len(detection_results)}")
    for object_name, box_info in detection_results.items():
        if box_info is not None:
            center = (box_info['center'] + scene_center).tolist()
            size = box_info['size'].tolist()
            angle = float(box_info['rotation_angle'])
            print(f"- {object_name}: 中心{center}, 尺寸{size}, 角度{angle:.3f}")
        else:
            print(f"- {object_name}: 3D框计算失败") 