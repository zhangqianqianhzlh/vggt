"""
主函数模块
"""

import time
import threading
import glob
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import os

from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map

from .object_detector import ObjectDetector
from .box_mapper import Box3DMapper
from .visualizer import ViserVisualizer
from .scene_aligner import SceneAligner
from .scene_renderer import Scene3DRenderer
from .utils import save_colmap_reconstruction, apply_sky_segmentation, save_multi_detection_results_json
from .config import load_llm_config, DEFAULT_CONFIG


def main_pipeline(
    pred_dict: dict,
    port: int = None,
    init_conf_threshold: float = None,
    use_point_map: bool = False,
    background_mode: bool = False,
    mask_sky: bool = False,
    image_folder: str = None,
    object_names: List[str] = None,
    up_down_direction: str = None,  # 新增参数
    save_colmap: bool = False,
    colmap_path: str = "colmap_recons",
    align_to_gravity: bool = True,
    save_3d_views: bool = True,
):
    """主要的处理和可视化流程"""
    
    # 使用默认配置
    if port is None:
        port = DEFAULT_CONFIG["default_port"]
    if init_conf_threshold is None:
        init_conf_threshold = DEFAULT_CONFIG["default_conf_threshold"]
    if object_names is None:
        object_names = ["黄色小车"]
    
    # 初始化可视化器
    visualizer = ViserVisualizer(port)
    visualizer.start_server()
    
    # 解包预测结果
    images = pred_dict["images"]  # (S, 3, H, W)
    world_points_map = pred_dict["world_points"]  # (S, H, W, 3)
    conf_map = pred_dict["world_points_conf"]  # (S, H, W)
    depth_map = pred_dict["depth"]  # (S, H, W, 1)
    depth_conf = pred_dict["depth_conf"]  # (S, H, W)
    extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
    intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)
    
    # 初始化场景对齐信息
    scene_alignment_info = None
    
    # === 场景对齐：自动检测朝向并对齐到重力方向 ===
    if align_to_gravity:
        print("\n=== 场景重力对齐 ===")
        
        # 检查用户指定的上下方向是否有效
        if up_down_direction and up_down_direction.upper() in ['X', 'Y', 'Z', '-X', '-Y', '-Z']:
            print(f"使用用户指定的上下方向: {up_down_direction}")
        elif up_down_direction is not None:
            print(f"用户指定的上下方向 '{up_down_direction}' 无效，将使用自动检测")
        else:
            print("使用自动检测的朝向")
            
        if not use_point_map:
            # 如果使用深度图生成点云，先生成点云再对齐
            world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
            world_points, extrinsics_cam, transform_matrix, scene_alignment_info = SceneAligner.align_scene_to_gravity(
                world_points, extrinsics_cam, user_specified_up_axis=up_down_direction)
            # 更新预测字典
            pred_dict["world_points"] = world_points
            pred_dict["extrinsic"] = extrinsics_cam
        else:
            # 直接对齐现有的点云
            world_points_map, extrinsics_cam, transform_matrix, scene_alignment_info = SceneAligner.align_scene_to_gravity(
                world_points_map, extrinsics_cam, user_specified_up_axis=up_down_direction)
            pred_dict["world_points"] = world_points_map
            pred_dict["extrinsic"] = extrinsics_cam
            
        if scene_alignment_info['user_specified']:
            print("场景已对齐到用户指定的方向")
        else:
            print("场景已对齐到自动检测的最佳方向")
    
    # 选择点云数据源
    if not use_point_map:
        if align_to_gravity:
            world_points = pred_dict["world_points"]  # 使用对齐后的点云
        else:
            world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        conf = depth_conf
    else:
        world_points = world_points_map
        conf = conf_map

    # 应用天空分割
    if mask_sky and image_folder is not None:
        conf = apply_sky_segmentation(conf, image_folder)

    # 准备点云数据
    colors = images.transpose(0, 2, 3, 1)  # (S, H, W, 3)
    S, H, W, _ = world_points.shape
    
    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)

    # 相机变换
    cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)
    cam_to_world = cam_to_world_mat[:, :3, :]

    # === 修复：提前计算场景中心，确保3D框和点云使用相同的中心化 ===
    scene_center = np.mean(points, axis=0)
    print(f"\n=== 场景中心化 ===")
    print(f"原始场景中心: [{scene_center[0]:.3f}, {scene_center[1]:.3f}, {scene_center[2]:.3f}]")
    
    # 对点云和相机都应用中心化
    points_centered = points - scene_center
    cam_to_world[..., -1] -= scene_center

    # 构建GUI
    frame_indices = np.repeat(np.arange(S), H * W)
    
    # 创建GUI控制界面
    gui_controls = visualizer.create_gui_controls(
        points_centered, colors_flat, conf_flat, frame_indices, init_conf_threshold, S
    )
    gui_show_frames, gui_points_conf, gui_frame_selector, point_cloud = gui_controls

    # === 主要逻辑：目标检测和3D框生成 ===
    if image_folder is not None:
        image_files = sorted(glob.glob(os.path.join(image_folder, "*")))
        if len(image_files) > 0:
            print(f"找到 {len(image_files)} 张图片")
            
            # 步骤1：目标检测
            llm_config = load_llm_config()
            detector = ObjectDetector(
                base_url=llm_config["llm_server"]["base_url"], 
                api_key=llm_config["llm_server"]["api_key"]
            )
            bboxes = detector.detect_multi_images_multi_objects(image_files, object_names, max_images=40)
            print(f"检测到的边框: {bboxes}")
            
            # 步骤2：显示带边框的相机视锥体（传递场景对齐信息）
            visualizer.visualize_frames_with_bbox(cam_to_world, images, bboxes[object_names[0]], scene_alignment_info)
            
            # === 修复：创建中心化的world_points用于3D框计算 ===
            world_points_centered = world_points.copy()
            for s in range(world_points.shape[0]):
                for h in range(world_points.shape[1]):
                    for w in range(world_points.shape[2]):
                        if np.all(np.isfinite(world_points[s, h, w])):
                            world_points_centered[s, h, w] = world_points[s, h, w] - scene_center
            
            print(f"应用场景中心化到world_points用于3D框计算")
            
            # 步骤3：3D框计算（使用中心化的点云和场景对齐信息）
            if use_point_map:
                world_points_map_centered = world_points_map.copy()
                for s in range(world_points_map.shape[0]):
                    for h in range(world_points_map.shape[1]):
                        for w in range(world_points_map.shape[2]):
                            if np.all(np.isfinite(world_points_map[s, h, w])):
                                world_points_map_centered[s, h, w] = world_points_map[s, h, w] - scene_center
                
                box_info = Box3DMapper.map_to_3d_box(
                    bboxes[object_names[0]], world_points_map_centered, depth_map, 
                    extrinsics_cam, intrinsics_cam, scene_alignment_info
                )
            else:
                box_info = Box3DMapper.map_to_3d_box(
                    bboxes[object_names[0]], world_points_centered, depth_map, 
                    extrinsics_cam, intrinsics_cam, scene_alignment_info
                )
            
            # 步骤4：3D框可视化（不需要再次中心化，因为已经在中心化坐标系中计算了）
            if box_info is not None:
                print(f"\n=== 3D包围框可视化 ===")
                print(f"处理方法: {box_info.get('method', 'unknown')}")
                print(f"旋转角度: {box_info['rotation_angle']:.3f} 弧度 ({np.degrees(box_info['rotation_angle']):.1f} 度)")
                print(f"3D框中心（中心化坐标）: [{box_info['center'][0]:.3f}, {box_info['center'][1]:.3f}, {box_info['center'][2]:.3f}]")
                
                # 添加目标名称到box_info中
                box_info['object_name'] = object_names[0]  # 使用检测的目标名称
                
                # === 修复：直接使用中心化的box_info，不再减去scene_center ===
                visualizer.add_3d_box_no_centering(box_info)  # 新的方法，不进行中心化
                
                # === 新增：保存3D视图（只生成俯瞰图） ===
                if save_3d_views:
                    print(f"\n=== 保存俯瞰图 ===")
                    views_output_dir = os.path.join(colmap_path, "3d_views")
                    Scene3DRenderer.render_top_down_view(
                        world_points, box_info, scene_center, images, views_output_dir
                    )
                
                # === 保存检测结果为JSON ===
                json_output_path = os.path.join(colmap_path, "detection_results.json")
                print(f"保存检测结果到: {json_output_path}")
                save_multi_detection_results_json(
                    {object_names[0]: box_info}, bboxes, image_files, scene_center, json_output_path
                )
            else:
                print("3D框计算失败")
        else:
            print("图片文件夹为空")
    else:
        print("未提供图片文件夹路径")

    # === 保存COLMAP格式结果 ===
    if save_colmap and image_folder is not None:
        save_colmap_reconstruction(
            pred_dict, 
            image_folder, 
            colmap_path
        )

    # 启动服务器循环
    print("Starting viser server...")
    if background_mode:
        def server_loop():
            while True:
                time.sleep(0.001)
        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
    else:
        while True:
            time.sleep(0.01)

    return visualizer.server 