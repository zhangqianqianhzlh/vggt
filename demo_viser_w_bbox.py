#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VGGT 3D目标检测和可视化演示脚本

主要功能：
1. 使用VGGT模型从多张图片生成3D重建
2. 使用Qwen模型检测图片中的目标物体
3. 将2D检测框映射到3D空间，生成水平对齐的3D包围框
4. 使用viser进行交互式3D可视化

作者：Facebook Research & 修改者
"""

import os
import glob
import time
import threading
import argparse
import re
from typing import List, Optional, Dict, Tuple, Any
import warnings

import numpy as np
import torch
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf
import cv2
import base64
from openai import OpenAI
from PIL import Image
import io

# 可选依赖
try:
    import onnxruntime
except ImportError:
    warnings.warn("onnxruntime not found. Sky segmentation may not work.")

# VGGT相关导入
from visual_util import segment_sky, download_file_from_url
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images, preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

import yaml
llm_config = yaml.safe_load(open("env/llm.yaml"))


class ObjectDetector:
    """Qwen视觉模型目标检测器"""
    
    def __init__(self, base_url: str, 
                 api_key: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = "Qwen/Qwen2.5-VL-72B-Instruct"
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """将PIL图像转换为base64编码"""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def detect_single_image(self, image_path: str, object_name: str) -> Optional[Tuple[int, int, int, int]]:
        """
        检测单张图片中的目标物体
        
        Args:
            image_path: 图片路径
            object_name: 目标物体名称
            
        Returns:
            边框坐标 (x1, y1, x2, y2) 或 None
        """
        try:
            images = preprocess_images([image_path])
            image = images[0]
            image_base64 = self._image_to_base64(image)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": f"识别图中{object_name}的坐标, 输出边框坐标, 用 [x1, y1, x2, y2] 格式输出"
                    }, {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    }]
                }],
                temperature=0.0,
                max_tokens=512
            )
            
            content = response.choices[0].message.content
            print(f"Qwen识别结果: {content}")
            
            # 提取坐标数字
            numbers = re.findall(r'\d+', content)
            if len(numbers) >= 4:
                x1, y1, x2, y2 = map(int, numbers[:4])
                print(f"提取的边框坐标: ({x1}, {y1}, {x2}, {y2})")
                return (x1, y1, x2, y2)
            else:
                print("未能从响应中提取有效的边框坐标")
                return None
                
        except Exception as e:
            print(f"检测失败: {e}")
            return None
    
    def detect_multi_images(self, image_paths: List[str], object_name: str, 
                          max_images: int = 4) -> List[Optional[Tuple[int, int, int, int]]]:
        """
        检测多张图片中的目标物体
        
        Args:
            image_paths: 图片路径列表
            object_name: 目标物体名称
            max_images: 最多处理的图片数量
            
        Returns:
            每张图片的边框坐标列表
        """
        selected_paths = image_paths[:max_images]
        results = []
        
        print(f"开始识别 {len(selected_paths)} 张图片中的目标...")
        
        for i, image_path in enumerate(selected_paths):
            print(f"正在识别第 {i+1}/{len(selected_paths)} 张图片: {image_path}")
            bbox = self.detect_single_image(image_path, object_name)
            results.append(bbox)
        
        return results


class Box3DMapper:
    """2D边框到3D包围框的映射器"""
    
    @staticmethod
    def analyze_camera_setup(extrinsics: np.ndarray, intrinsics: np.ndarray, 
                           bboxes: List[Optional[tuple]], world_points: np.ndarray, 
                           depth_map: np.ndarray) -> Tuple[str, float]:
        """分析相机设置和拍摄类型"""
        print("\n=== 相机设置分析 ===")
        
        # 计算相机位置
        cam_positions = []
        for i in range(len(extrinsics)):
            cam_to_world_4x4 = closed_form_inverse_se3(extrinsics[i:i+1])[0]
            cam_pos = cam_to_world_4x4[:3, 3]
            cam_positions.append(cam_pos)
            print(f"相机{i+1}位置: [{cam_pos[0]:.6f}, {cam_pos[1]:.6f}, {cam_pos[2]:.6f}]")
        
        cam_positions = np.array(cam_positions)
        
        # 分析相机间距离
        if len(cam_positions) > 1:
            distances = []
            for i in range(len(cam_positions)):
                for j in range(i+1, len(cam_positions)):
                    dist = np.linalg.norm(cam_positions[i] - cam_positions[j])
                    distances.append(dist)
            
            avg_cam_distance = np.mean(distances)
            print(f"相机平均距离: {avg_cam_distance:.6f}")
        else:
            avg_cam_distance = 0.0
        
        # 分析目标深度
        valid_bboxes = [bbox for bbox in bboxes if bbox is not None]
        if valid_bboxes:
            H, W = depth_map.shape[1], depth_map.shape[2]
            target_depths = []
            
            for i, bbox in enumerate(valid_bboxes):
                if i >= len(valid_bboxes):
                    continue
                x1, y1, x2, y2 = bbox
                
                # 采样边框中心区域的深度
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                center_x = max(0, min(center_x, W-1))
                center_y = max(0, min(center_y, H-1))
                
                depth_val = depth_map[i, center_y, center_x, 0]
                if depth_val > 0.01:
                    target_depths.append(depth_val)
            
            if target_depths:
                avg_target_depth = np.mean(target_depths)
                print(f"目标平均深度: {avg_target_depth:.6f}")
                
                # 根据深度判断拍摄类型
                if avg_target_depth < 0.5:
                    setup_type = "close_up"
                    scale_param = avg_target_depth
                elif avg_target_depth < 1.5:
                    setup_type = "medium"
                    scale_param = avg_target_depth
                else:
                    setup_type = "far"
                    scale_param = avg_target_depth
            else:
                setup_type = "unknown"
                scale_param = avg_cam_distance
        else:
            setup_type = "unknown"
            scale_param = avg_cam_distance
        
        print(f"判断结果: {setup_type}, 尺度参数: {scale_param:.6f}")
        return setup_type, scale_param
    
    @staticmethod
    def map_to_3d_box(bboxes: List[Optional[tuple]], world_points: np.ndarray, 
                     depth_map: np.ndarray, extrinsics: np.ndarray, 
                     intrinsics: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        将2D边框映射到3D包围框（强制X轴向上，水平对齐）
        
        Args:
            bboxes: 2D边框列表
            world_points: 世界坐标点云 (S, H, W, 3)
            depth_map: 深度图 (S, H, W, 1)
            extrinsics: 外参矩阵 (S, 3, 4)
            intrinsics: 内参矩阵 (S, 3, 3)
            
        Returns:
            3D包围框信息字典
        """
        # 相机设置分析
        camera_setup, scale_param = Box3DMapper.analyze_camera_setup(
            extrinsics, intrinsics, bboxes, world_points, depth_map)
        
        valid_3d_points = []
        H, W = world_points.shape[1], world_points.shape[2]
        
        print(f"\n=== 3D点采样 ===")
        
        # 从每个边框中采样3D点
        for img_idx, bbox in enumerate(bboxes):
            if bbox is None:
                continue
                
            x1, y1, x2, y2 = bbox
            # 确保坐标在图像范围内
            x1 = max(0, min(x1, W-1))
            y1 = max(0, min(y1, H-1))
            x2 = max(0, min(x2, W-1))
            y2 = max(0, min(y2, H-1))
            
            print(f"图片{img_idx+1} 边框: ({x1}, {y1}, {x2}, {y2})")
            
            # 内部采样（避免边缘噪声）
            margin_ratio = 0.15
            margin_x = max(1, int((x2 - x1) * margin_ratio))
            margin_y = max(1, int((y2 - y1) * margin_ratio))
            
            inner_x1 = x1 + margin_x
            inner_y1 = y1 + margin_y
            inner_x2 = x2 - margin_x
            inner_y2 = y2 - margin_y
            
            if inner_x2 <= inner_x1 or inner_y2 <= inner_y1:
                continue
            
            # 采样步长
            sample_step = max(1, min(inner_x2 - inner_x1, inner_y2 - inner_y1) // 6)
            
            bbox_points = []
            bbox_depths = []
            
            # 在边框内部采样点
            for px in range(inner_x1, inner_x2+1, sample_step):
                for py in range(inner_y1, inner_y2+1, sample_step):
                    depth_val = depth_map[img_idx, py, px, 0]
                    point_3d = world_points[img_idx, py, px]
                    
                    # 过滤有效点
                    if (0.05 < depth_val < 5.0 and
                        np.all(np.isfinite(point_3d)) and
                        not np.allclose(point_3d, 0)):
                        
                        bbox_points.append(point_3d)
                        bbox_depths.append(depth_val)
            
            # 深度过滤
            if len(bbox_points) > 3:
                depths_array = np.array(bbox_depths)
                depth_median = np.median(depths_array)
                depth_mad = np.median(np.abs(depths_array - depth_median))
                
                if depth_mad > 0:
                    depth_threshold = max(0.02, 2.0 * depth_mad)
                    valid_mask = np.abs(depths_array - depth_median) <= depth_threshold
                    
                    filtered_points = np.array(bbox_points)[valid_mask]
                    print(f"  深度过滤后: {len(filtered_points)} 个点")
                    
                    if len(filtered_points) > 2:
                        valid_3d_points.extend(filtered_points)
        
        if len(valid_3d_points) < 4:
            print(f"有效3D点不足: {len(valid_3d_points)}")
            return None
        
        points_3d = np.array(valid_3d_points)
        print(f"总共有效3D点: {len(points_3d)}")
        
        # 计算中心点
        center_3d = np.mean(points_3d, axis=0)
        print(f"3D点云中心: [{center_3d[0]:.3f}, {center_3d[1]:.3f}, {center_3d[2]:.3f}]")
        
        # === 强制X轴向上，在YZ平面内进行PCA ===
        up_direction = np.array([1.0, 0.0, 0.0])  # X轴向上
        
        # 只取YZ坐标进行PCA
        horizontal_points = points_3d[:, [1, 2]]  # Y, Z坐标
        horizontal_center = np.mean(horizontal_points, axis=0)
        centered_horizontal = horizontal_points - horizontal_center
        
        if len(centered_horizontal) > 1:
            # 在YZ平面上进行PCA
            cov_matrix_2d = np.cov(centered_horizontal.T)
            eigenvalues_2d, eigenvectors_2d = np.linalg.eigh(cov_matrix_2d)
            
            # 按特征值排序
            idx = np.argsort(eigenvalues_2d)[::-1]
            eigenvalues_2d = eigenvalues_2d[idx]
            eigenvectors_2d = eigenvectors_2d[:, idx]
            
            # 转换为3D向量
            main_direction = np.array([0.0, eigenvectors_2d[0, 0], eigenvectors_2d[1, 0]])
            secondary_direction = np.array([0.0, eigenvectors_2d[0, 1], eigenvectors_2d[1, 1]])
            
            # 归一化
            main_direction = main_direction / np.linalg.norm(main_direction)
            secondary_direction = secondary_direction / np.linalg.norm(secondary_direction)
            
            # 计算旋转角度
            rotation_angle = np.arctan2(main_direction[2], main_direction[1])
            
        else:
            # 默认方向
            main_direction = np.array([0.0, 1.0, 0.0])      # Y轴
            secondary_direction = np.array([0.0, 0.0, 1.0])  # Z轴
            rotation_angle = 0.0
        
        # 计算3D框尺寸
        centered_points = points_3d - center_3d
        projections_main = np.dot(centered_points, main_direction)
        projections_secondary = np.dot(centered_points, secondary_direction)
        projections_up = np.dot(centered_points, up_direction)
        
        # 使用95%分位数计算尺寸
        width = np.percentile(projections_main, 95) - np.percentile(projections_main, 5)
        length = np.percentile(projections_secondary, 95) - np.percentile(projections_secondary, 5)
        height = np.percentile(projections_up, 95) - np.percentile(projections_up, 5)
        
        # 最小尺寸限制
        min_size = 0.02
        width = max(width, min_size)
        length = max(length, min_size)
        height = max(height, min_size)
        
        print(f"3D框尺寸: 宽={width:.4f}, 长={length:.4f}, 高={height:.4f}")
        print(f"旋转角度: {rotation_angle:.3f} 弧度 ({np.degrees(rotation_angle):.1f} 度)")
        
        return {
            'center': center_3d,
            'size': np.array([width, length, height]),
            'main_direction': main_direction,
            'secondary_direction': secondary_direction,
            'up_direction': up_direction,
            'rotation_angle': rotation_angle,
            'points_3d': points_3d,
            'camera_setup': camera_setup,
            'scale_param': scale_param,
            'method': 'horizontal_aligned'
        }
    
    @staticmethod
    def create_3d_box_vertices(box_info: Dict[str, Any]) -> np.ndarray:
        """创建3D包围框的8个顶点"""
        center = box_info['center']
        size = box_info['size']
        main_dir = box_info['main_direction']
        secondary_dir = box_info['secondary_direction']
        up_dir = box_info['up_direction']
        
        # 半尺寸
        half_size = size / 2
        
        # 8个顶点的局部坐标
        local_vertices = np.array([
            [-half_size[0], -half_size[1], -half_size[2]],  # 底面4个点
            [+half_size[0], -half_size[1], -half_size[2]],
            [+half_size[0], +half_size[1], -half_size[2]],
            [-half_size[0], +half_size[1], -half_size[2]],
            [-half_size[0], -half_size[1], +half_size[2]],  # 顶面4个点
            [+half_size[0], -half_size[1], +half_size[2]],
            [+half_size[0], +half_size[1], +half_size[2]],
            [-half_size[0], +half_size[1], +half_size[2]],
        ])
        
        # 构建旋转矩阵
        rotation_matrix = np.column_stack([main_dir, secondary_dir, up_dir])
        
        # 转换到世界坐标
        world_vertices = np.dot(local_vertices, rotation_matrix.T) + center
        
        return world_vertices


class ImageUtils:
    """图像处理工具类"""
    
    @staticmethod
    def draw_bbox_on_image(img: np.ndarray, bbox: Optional[tuple]) -> np.ndarray:
        """在图片上绘制边框"""
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


class ViserVisualizer:
    """Viser 3D可视化器"""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.server = None
        self.frames = []
        self.frustums = []
    
    def start_server(self):
        """启动viser服务器"""
        print(f"Starting viser server on port {self.port}")
        self.server = viser.ViserServer(host="0.0.0.0", port=self.port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    
    def visualize_frames_with_bbox(self, extrinsics: np.ndarray, images: np.ndarray, 
                                 bboxes: List[Optional[tuple]]) -> None:
        """可视化带边框的相机视锥体"""
        # 清除现有的frames和frustums
        for f in self.frames:
            f.remove()
        self.frames.clear()
        for fr in self.frustums:
            fr.remove()
        self.frustums.clear()

        def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in self.server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        for img_id in range(len(images)):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            # 添加相机坐标轴
            frame_axis = self.server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            self.frames.append(frame_axis)

            # 转换图片格式
            img = images[img_id]  # (3, H, W)
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)  # (H, W, 3)
            h, w = img.shape[:2]
            
            # 在图片上绘制边框
            if img_id < len(bboxes) and bboxes[img_id] is not None:
                bbox = bboxes[img_id]
                img_with_bbox = ImageUtils.draw_bbox_on_image(img, bbox)
                print(f"在相机{img_id+1}的图片上绘制边框: {bbox}")
            else:
                img_with_bbox = img
                print(f"相机{img_id+1}没有边框信息")

            # 计算FOV
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            # 添加视锥体
            frustum_cam = self.server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum", 
                fov=fov, 
                aspect=w / h, 
                scale=0.05, 
                image=img_with_bbox,
                line_width=1.0
            )
            self.frustums.append(frustum_cam)
            attach_callback(frustum_cam, frame_axis)
    
    def add_3d_box(self, box_info: Dict[str, Any], scene_center: np.ndarray) -> None:
        """添加3D包围框到场景"""
        # 应用场景中心化
        box_info['center'] = box_info['center'] - scene_center
        box_info['points_3d'] = box_info['points_3d'] - scene_center
        
        # 创建3D框顶点
        box_points = Box3DMapper.create_3d_box_vertices(box_info)
        
        # 添加红色3D框线段
        box_edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
            [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
            [0, 4], [1, 5], [2, 6], [3, 7],  # 连接线
        ])
        
        line_segments = np.array([
            [box_points[edge[0]], box_points[edge[1]]] for edge in box_edges
        ])
        
        self.server.scene.add_line_segments(
            name="3d_bounding_box",
            points=line_segments,
            colors=(255, 0, 0),
            line_width=4.0
        )
        
        # 添加中心点（紫色）
        self.server.scene.add_point_cloud(
            name="box_center",
            points=np.array([box_info['center']]),
            colors=np.array([[255, 0, 255]]),
            point_size=0.02
        )
        
        # ===== 新增：添加3D文本标签 =====
        # 添加目标物体的标签
        object_name = box_info.get('object_name', '检测目标')
        
        # 在3D框上方添加标签
        label_position = box_info['center'].copy()
        label_position[2] += box_info['size'][2] * 0.6  # 在框上方
        
        self.server.scene.add_label(
            name="object_label",
            text=object_name,
            position=label_position,
            visible=True
        )
        
        # 添加尺寸信息标签
        # size_text = f"尺寸: {box_info['size'][0]:.2f}×{box_info['size'][1]:.2f}×{box_info['size'][2]:.2f}"
        # size_label_position = box_info['center'].copy()
        # size_label_position[2] += box_info['size'][2] * 0.4  # 在主标签下方
        
        # self.server.scene.add_label(
        #     name="size_label",
        #     text=size_text,
        #     position=size_label_position,
        #     visible=True
        # )
        
        # 添加角度信息标签
        # angle_deg = np.degrees(box_info['rotation_angle'])
        # angle_text = f"旋转角度: {angle_deg:.1f}°"
        # angle_label_position = box_info['center'].copy()
        # angle_label_position[2] += box_info['size'][2] * 0.2  # 在尺寸标签下方
        
        # self.server.scene.add_label(
        #     name="angle_label", 
        #     text=angle_text,
        #     position=angle_label_position,
        #     visible=True
        # )
        
        print("3D包围框和标签可视化完成")
        print(f"- 目标标签: {object_name}")

    def add_3d_text_labels(self, labels_info: List[Dict[str, Any]]) -> None:
        """添加3D文本标签到场景
        
        Args:
            labels_info: 标签信息列表，每个元素包含:
                - text: 标签文本
                - position: 3D位置 (x, y, z)
                - color: 颜色 (可选，默认白色)
        """
        for i, label_info in enumerate(labels_info):
            text = label_info.get('text', f'Label_{i}')
            position = label_info.get('position', (0, 0, 0))
            
            # 添加3D文本标签
            self.server.scene.add_label(
                name=f"label_{i}",
                text=text,
                position=position,
                visible=True
            )
            
            print(f"添加3D标签: '{text}' 在位置 {position}")


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


def main_pipeline(
    pred_dict: dict,
    port: int = 8080,
    init_conf_threshold: float = 50.0,
    use_point_map: bool = False,
    background_mode: bool = False,
    mask_sky: bool = False,
    image_folder: str = None,
    object_name: str = "黄色小车",
):
    """主要的处理和可视化流程"""
    
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

    # 选择点云数据源
    if not use_point_map:
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

    # 场景中心化
    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center
    cam_to_world[..., -1] -= scene_center

    # 构建GUI
    frame_indices = np.repeat(np.arange(S), H * W)
    
    gui_show_frames = visualizer.server.gui.add_checkbox("Show Cameras", initial_value=True)
    gui_points_conf = visualizer.server.gui.add_slider(
        "Confidence Percent", min=0, max=100, step=0.1, initial_value=init_conf_threshold
    )
    gui_frame_selector = visualizer.server.gui.add_dropdown(
        "Show Points from Frames", options=["All"] + [str(i) for i in range(S)], initial_value="All"
    )

    # 创建点云
    init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
    init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)
    point_cloud = visualizer.server.scene.add_point_cloud(
        name="viser_pcd",
        points=points_centered[init_conf_mask],
        colors=colors_flat[init_conf_mask],
        point_size=0.001,
        point_shape="circle",
    )

    def update_point_cloud() -> None:
        """更新点云显示"""
        current_percentage = gui_points_conf.value
        threshold_val = np.percentile(conf_flat, current_percentage)
        conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)

        if gui_frame_selector.value == "All":
            frame_mask = np.ones_like(conf_mask, dtype=bool)
        else:
            selected_idx = int(gui_frame_selector.value)
            frame_mask = frame_indices == selected_idx

        combined_mask = conf_mask & frame_mask
        point_cloud.points = points_centered[combined_mask]
        point_cloud.colors = colors_flat[combined_mask]

    @gui_points_conf.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_frame_selector.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_show_frames.on_update
    def _(_) -> None:
        for f in visualizer.frames:
            f.visible = gui_show_frames.value
        for fr in visualizer.frustums:
            fr.visible = gui_show_frames.value

    # === 主要逻辑：目标检测和3D框生成 ===
    if image_folder is not None:
        image_files = sorted(glob.glob(os.path.join(image_folder, "*")))
        if len(image_files) > 0:
            print(f"找到 {len(image_files)} 张图片")
            
            # 步骤1：目标检测
            detector = ObjectDetector(base_url=llm_config["llm_server"]["base_url"], api_key=llm_config["llm_server"]["api_key"])
            bboxes = detector.detect_multi_images(image_files, object_name, max_images=40)
            print(f"检测到的边框: {bboxes}")
            
            # 步骤2：显示带边框的相机视锥体
            visualizer.visualize_frames_with_bbox(cam_to_world, images, bboxes)
            
            # 步骤3：3D框计算
            if use_point_map:
                box_info = Box3DMapper.map_to_3d_box(
                    bboxes, world_points_map, depth_map, extrinsics_cam, intrinsics_cam
                )
            else:
                box_info = Box3DMapper.map_to_3d_box(
                    bboxes, world_points, depth_map, extrinsics_cam, intrinsics_cam
                )
            
            # 步骤4：3D框可视化
            if box_info is not None:
                print(f"\n=== 3D包围框可视化 ===")
                print(f"处理方法: {box_info.get('method', 'unknown')}")
                print(f"旋转角度: {box_info['rotation_angle']:.3f} 弧度 ({np.degrees(box_info['rotation_angle']):.1f} 度)")
                
                # 添加目标名称到box_info中
                box_info['object_name'] = object_name  # 使用检测的目标名称
                visualizer.add_3d_box(box_info, scene_center)
                
                # # 可选：添加额外的自定义标签
                # custom_labels = [
                #     {
                #         'text': '检测置信度: 85%',
                #         'position': (box_info['center'][0], box_info['center'][1], box_info['center'][2] - box_info['size'][2] * 0.3)
                #     },
                #     {
                #         'text': '检测时间: 2024-01-15',
                #         'position': (box_info['center'][0], box_info['center'][1], box_info['center'][2] - box_info['size'][2] * 0.5)
                #     }
                # ]
                # visualizer.add_3d_text_labels(custom_labels)
            else:
                print("3D框计算失败")
        else:
            print("图片文件夹为空")
    else:
        print("未提供图片文件夹路径")

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


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="VGGT 3D目标检测和可视化演示")
    parser.add_argument("--image_folder", type=str, default="examples/kitchen_simple/images/", help="包含图片的文件夹路径")
    parser.add_argument("--use_point_map", action="store_true", help="使用点图而不是基于深度的点")
    parser.add_argument("--background_mode", action="store_true", help="在后台模式运行viser服务器")
    parser.add_argument("--port", type=int, default=8080, help="viser服务器端口号")
    parser.add_argument("--conf_threshold", type=float, default=25.0, help="初始置信度阈值百分比")
    parser.add_argument("--mask_sky", action="store_true", help="应用天空分割过滤天空点")
    parser.add_argument("--object_name", type=str, default="黄色小车", help="要检测的目标物体名称")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 加载VGGT模型
    print("初始化和加载VGGT模型...")
    model = VGGT.from_pretrained("/data3/qq/models/VGGT-1B").to(device)

    # 加载图片
    print(f"从 {args.image_folder} 加载图片...")
    image_names = glob.glob(os.path.join(args.image_folder, "*"))
    print(f"找到 {len(image_names)} 张图片")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"预处理后的图片形状: {images.shape}")

    # 运行推理
    print("运行推理...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    # 转换姿态编码
    print("转换姿态编码为外参和内参矩阵...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # 处理模型输出
    print("处理模型输出...")
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)

    # 启动可视化
    print("启动可视化...")
    viser_server = main_pipeline(
        predictions,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        background_mode=args.background_mode,
        mask_sky=args.mask_sky,
        image_folder=args.image_folder,
        object_name=args.object_name,
    )
    print("可视化完成")


if __name__ == "__main__":
    main()