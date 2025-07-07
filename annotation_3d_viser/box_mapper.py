"""
3D包围框映射器模块
"""

import numpy as np
from typing import List, Optional, Dict, Tuple, Any
from vggt.utils.geometry import closed_form_inverse_se3
from .config import DEFAULT_CONFIG


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
                     intrinsics: np.ndarray, scene_alignment_info: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        将2D边框映射到3D包围框（与地面平行）
        
        Args:
            bboxes: 2D边框列表
            world_points: 世界坐标点云 (S, H, W, 3)
            depth_map: 深度图 (S, H, W, 1)
            extrinsics: 外参矩阵 (S, 3, 4)
            intrinsics: 内参矩阵 (S, 3, 3)
            scene_alignment_info: 场景对齐信息
            
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
            margin_ratio = DEFAULT_CONFIG["sampling_margin_ratio"]
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
                    depth_threshold = max(0.02, DEFAULT_CONFIG["depth_filter_threshold"] * depth_mad)
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
        
        # === 使用场景对齐信息构建坐标系 ===
        if scene_alignment_info is not None:
            up_direction = scene_alignment_info['up_direction']
            axis_name = scene_alignment_info['axis_name']
            print(f"使用场景对齐的向上方向: {up_direction} ({axis_name}轴)")
        else:
            up_direction = np.array([0.0, 0.0, 1.0])  # 默认Z轴向上
            axis_name = "Z"
            print("使用默认的向上方向: Z轴")
        
        # 确定水平面的两个方向
        # 根据向上方向确定水平面
        if np.allclose(up_direction, [1.0, 0.0, 0.0]) or np.allclose(up_direction, [-1.0, 0.0, 0.0]):
            print("X轴向上，水平面是YZ平面")
            horizontal_indices = [1, 2]  # Y, Z
            horizontal_labels = ['Y', 'Z']
        elif np.allclose(up_direction, [0.0, 1.0, 0.0]) or np.allclose(up_direction, [0.0, -1.0, 0.0]):
            # Y轴向上，水平面是XZ平面
            horizontal_indices = [0, 2]  # X, Z
            horizontal_labels = ['X', 'Z']
        else:
            # Z轴向上（默认），水平面是XY平面
            horizontal_indices = [0, 1]  # X, Y
            horizontal_labels = ['X', 'Y']
        
        print(f"水平面: {horizontal_labels[0]}{horizontal_labels[1]}平面")
        
        # 在水平面内进行PCA
        horizontal_points = points_3d[:, horizontal_indices]
        horizontal_center = np.mean(horizontal_points, axis=0)
        centered_horizontal = horizontal_points - horizontal_center
        
        if len(centered_horizontal) > 1:
            # 在水平面上进行PCA
            cov_matrix_2d = np.cov(centered_horizontal.T)
            eigenvalues_2d, eigenvectors_2d = np.linalg.eigh(cov_matrix_2d)
            
            # 按特征值排序
            idx = np.argsort(eigenvalues_2d)[::-1]
            eigenvalues_2d = eigenvalues_2d[idx]
            eigenvectors_2d = eigenvectors_2d[:, idx]
            
            # 构建水平面内的主方向
            main_direction_2d = eigenvectors_2d[:, 0]  # 主方向
            secondary_direction_2d = eigenvectors_2d[:, 1]  # 次方向
            
            # 转换为3D向量
            main_direction = np.zeros(3)
            secondary_direction = np.zeros(3)
            
            main_direction[horizontal_indices] = main_direction_2d
            secondary_direction[horizontal_indices] = secondary_direction_2d
            
            # 归一化
            main_direction = main_direction / np.linalg.norm(main_direction)
            secondary_direction = secondary_direction / np.linalg.norm(secondary_direction)
            
            # 计算旋转角度（在水平面内）
            if horizontal_indices == [0, 1]:  # XY平面
                rotation_angle = np.arctan2(main_direction[1], main_direction[0])
            elif horizontal_indices == [0, 2]:  # XZ平面
                rotation_angle = np.arctan2(main_direction[2], main_direction[0])
            else:  # YZ平面
                rotation_angle = np.arctan2(main_direction[2], main_direction[1])
            
        else:
            # 默认方向（水平对齐）
            main_direction = np.zeros(3)
            secondary_direction = np.zeros(3)
            
            main_direction[horizontal_indices[0]] = 1.0  # 第一个水平轴
            secondary_direction[horizontal_indices[1]] = 1.0  # 第二个水平轴
            rotation_angle = 0.0
        
        # === 确保正交性 ===
        # 重新计算secondary_direction以确保正交
        secondary_direction = np.cross(up_direction, main_direction)
        if np.linalg.norm(secondary_direction) > 1e-6:
            secondary_direction = secondary_direction / np.linalg.norm(secondary_direction)
        
        # 重新计算main_direction以确保完全正交
        main_direction = np.cross(secondary_direction, up_direction)
        if np.linalg.norm(main_direction) > 1e-6:
            main_direction = main_direction / np.linalg.norm(main_direction)
        
        # 验证正交性
        dot_main_sec = np.dot(main_direction, secondary_direction)
        dot_main_up = np.dot(main_direction, up_direction)
        dot_sec_up = np.dot(secondary_direction, up_direction)
        
        print(f"正交性验证:")
        print(f"  main·secondary = {dot_main_sec:.6f} (应接近0)")
        print(f"  main·up = {dot_main_up:.6f} (应接近0)")
        print(f"  secondary·up = {dot_sec_up:.6f} (应接近0)")
        
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
        min_size = DEFAULT_CONFIG["min_box_size"]
        width = max(width, min_size)
        length = max(length, min_size)
        height = max(height, min_size)
        
        print(f"3D框尺寸: 宽={width:.4f}, 长={length:.4f}, 高={height:.4f}")
        print(f"旋转角度: {rotation_angle:.3f} 弧度 ({np.degrees(rotation_angle):.1f} 度)")
        print(f"坐标系: {axis_name}轴向上, 在{horizontal_labels[0]}{horizontal_labels[1]}平面内旋转, 保证正交")
        
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
            'method': 'ground_aligned_orthogonal',
            'aligned_axis': axis_name
        }
    
    @staticmethod
    def create_3d_box_vertices(box_info: Dict[str, Any]) -> np.ndarray:
        """创建3D包围框的8个顶点（确保是标准矩形框）"""
        center = box_info['center']
        size = box_info['size']
        main_dir = box_info['main_direction']
        secondary_dir = box_info['secondary_direction']
        up_dir = box_info['up_direction']
        
        # 验证输入的正交性
        dot_main_sec = np.dot(main_dir, secondary_dir)
        dot_main_up = np.dot(main_dir, up_dir)
        dot_sec_up = np.dot(secondary_dir, up_dir)
        
        if abs(dot_main_sec) > 1e-6 or abs(dot_main_up) > 1e-6 or abs(dot_sec_up) > 1e-6:
            print(f"警告: 输入的方向向量不正交!")
            print(f"  main·secondary = {dot_main_sec:.6f}")
            print(f"  main·up = {dot_main_up:.6f}")
            print(f"  secondary·up = {dot_sec_up:.6f}")
        
        # 半尺寸
        half_size = size / 2
        
        # 8个顶点的局部坐标（标准矩形框）
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
        
        # 构建正交旋转矩阵
        rotation_matrix = np.column_stack([main_dir, secondary_dir, up_dir])
        
        # 验证旋转矩阵的正交性
        should_be_identity = np.dot(rotation_matrix.T, rotation_matrix)
        identity_error = np.max(np.abs(should_be_identity - np.eye(3)))
        
        if identity_error > 1e-6:
            print(f"警告: 旋转矩阵不正交! 最大误差: {identity_error:.6f}")
        
        # 转换到世界坐标
        world_vertices = np.dot(local_vertices, rotation_matrix.T) + center
        
        return world_vertices 