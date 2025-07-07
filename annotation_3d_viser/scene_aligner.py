"""
场景对齐器模块
"""

import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from .config import DEFAULT_CONFIG


class SceneAligner:
    """场景对齐器 - 让地面与坐标轴对齐"""
    
    @staticmethod
    def estimate_ground_plane(points_3d: np.ndarray, sample_ratio: float = 0.1) -> np.ndarray:
        """
        估计地面平面法向量
        
        Args:
            points_3d: 3D点云
            sample_ratio: 采样比例
            
        Returns:
            地面法向量 (3,)
        """
        # 随机采样点云
        n_points = len(points_3d)
        n_sample = max(1000, int(n_points * sample_ratio))
        indices = np.random.choice(n_points, min(n_sample, n_points), replace=False)
        sampled_points = points_3d[indices]
        
        # 使用PCA找到最大的平面
        centered_points = sampled_points - np.mean(sampled_points, axis=0)
        cov_matrix = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 最小特征值对应的特征向量是法向量
        normal = eigenvectors[:, 0]
        
        return normal
    
    @staticmethod
    def find_best_axis_alignment(ground_normal: np.ndarray, extrinsics: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        找到最小旋转的轴对齐方案
        
        Args:
            ground_normal: 地面法向量
            extrinsics: 相机外参
            
        Returns:
            最佳目标法向量和轴名称
        """
        from vggt.utils.geometry import closed_form_inverse_se3
        
        # 三个坐标轴方向
        axes = {
            'X': np.array([1.0, 0.0, 0.0]),
            'Y': np.array([0.0, 1.0, 0.0]),
            'Z': np.array([0.0, 0.0, 1.0]),
            '-X': np.array([-1.0, 0.0, 0.0]),
            '-Y': np.array([0.0, -1.0, 0.0]),
            '-Z': np.array([0.0, 0.0, -1.0])
        }
        
        # 计算每个轴的旋转角度
        angles = {}
        for axis_name, axis_vec in axes.items():
            dot_product = np.clip(np.dot(ground_normal, axis_vec), -1.0, 1.0)
            angle = np.arccos(abs(dot_product))  # 使用绝对值，考虑两个方向
            angles[axis_name] = angle
        
        # 找到最小旋转角度的轴
        best_axis_name = min(angles, key=angles.get)
        best_angle = angles[best_axis_name]
        best_target = axes[best_axis_name]
        
        # 如果地面法向量与选定轴方向相反，调整目标向量
        if np.dot(ground_normal, best_target) < 0:
            best_target = -best_target
            best_axis_name = '-' + best_axis_name if not best_axis_name.startswith('-') else best_axis_name[1:]
        
        print(f"地面法向量: [{ground_normal[0]:.3f}, {ground_normal[1]:.3f}, {ground_normal[2]:.3f}]")
        print(f"最佳对齐轴: {best_axis_name}")
        print(f"最小旋转角度: {np.degrees(best_angle):.1f} 度")
        
        # 考虑相机位置的影响
        if len(extrinsics) > 0:
            # 计算相机位置
            cam_to_world_mat = closed_form_inverse_se3(extrinsics)
            cam_positions = cam_to_world_mat[:, :3, 3]
            camera_center = np.mean(cam_positions, axis=0)
            
            print(f"相机中心: [{camera_center[0]:.3f}, {camera_center[1]:.3f}, {camera_center[2]:.3f}]")
            
            # 如果旋转角度很小（小于15度），直接使用
            if np.degrees(best_angle) < 15:
                print("旋转角度很小，直接使用最小旋转方案")
                return best_target, best_axis_name
            
            # 如果旋转角度较大，考虑相机的主要观察方向
            # 选择与相机位置分布最一致的轴
            camera_spread = np.std(cam_positions, axis=0)
            dominant_camera_axis = np.argmax(camera_spread)
            
            print(f"相机位置分布标准差: [{camera_spread[0]:.3f}, {camera_spread[1]:.3f}, {camera_spread[2]:.3f}]")
            print(f"相机主要分布轴: {['X', 'Y', 'Z'][dominant_camera_axis]}")
            
            # 如果相机主要沿某个轴分布，优先考虑垂直于该轴的方向作为"向上"
            if dominant_camera_axis == 0:  # 相机主要沿X轴分布
                preferred_axes = ['Y', 'Z', '-Y', '-Z']
            elif dominant_camera_axis == 1:  # 相机主要沿Y轴分布
                preferred_axes = ['X', 'Z', '-X', '-Z']
            else:  # 相机主要沿Z轴分布
                preferred_axes = ['X', 'Y', '-X', '-Y']
            
            # 在首选轴中找到旋转角度最小的
            best_preferred_angle = float('inf')
            best_preferred_axis = best_axis_name
            best_preferred_target = best_target
            
            for axis_name in preferred_axes:
                if axis_name in angles:
                    if angles[axis_name] < best_preferred_angle:
                        best_preferred_angle = angles[axis_name]
                        best_preferred_axis = axis_name
                        best_preferred_target = axes[axis_name]
                        if np.dot(ground_normal, best_preferred_target) < 0:
                            best_preferred_target = -best_preferred_target
            
            # 如果首选方案的旋转角度不比最小方案大太多（差异小于30度），使用首选方案
            if np.degrees(best_preferred_angle - best_angle) < 30:
                print(f"使用相机友好的对齐方案: {best_preferred_axis}, 角度: {np.degrees(best_preferred_angle):.1f} 度")
                return best_preferred_target, best_preferred_axis
        
        return best_target, best_axis_name
    
    @staticmethod
    def parse_user_axis(user_axis: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        解析用户指定的轴方向
        
        Args:
            user_axis: 用户指定的轴方向，如 "X", "Y", "Z", "-X", "-Y", "-Z"
            
        Returns:
            对应的单位向量和轴名称，如果无效则返回 (None, None)
        """
        axis_mapping = {
            'X': (np.array([1.0, 0.0, 0.0]), 'X'),
            'Y': (np.array([0.0, 1.0, 0.0]), 'Y'),
            'Z': (np.array([0.0, 0.0, 1.0]), 'Z'),
            '-X': (np.array([-1.0, 0.0, 0.0]), '-X'),
            '-Y': (np.array([0.0, -1.0, 0.0]), '-Y'),
            '-Z': (np.array([0.0, 0.0, -1.0]), '-Z')
        }
        
        if user_axis is None:
            return None, None
        
        user_axis_upper = user_axis.upper()
        if user_axis_upper in axis_mapping:
            return axis_mapping[user_axis_upper]
        else:
            print(f"警告: 不支持的轴方向 '{user_axis}'，将使用自动检测")
            return None, None
    
    @staticmethod
    def align_scene_to_gravity(world_points: np.ndarray, 
                             extrinsics: np.ndarray,
                             user_specified_up_axis: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        将场景对齐到重力方向（支持用户指定或自动检测）
        
        Args:
            world_points: 世界坐标点云 (S, H, W, 3)
            extrinsics: 相机外参 (S, 3, 4)
            user_specified_up_axis: 用户指定的地面法向量方向，如 "X", "Y", "Z", "-X", "-Y", "-Z"
            
        Returns:
            对齐后的点云、外参、变换矩阵和对齐信息
        """
        # 展平点云进行分析
        points_flat = world_points.reshape(-1, 3)
        valid_mask = np.all(np.isfinite(points_flat), axis=1)
        valid_points = points_flat[valid_mask]
        
        if len(valid_points) < 100:
            print("警告: 有效点太少，跳过场景对齐")
            default_alignment_info = {
                'up_direction': np.array([0.0, 0.0, 1.0]),
                'axis_name': 'Z',
                'rotation_matrix': np.eye(3),
                'user_specified': False
            }
            return world_points, extrinsics, np.eye(4), default_alignment_info
        
        print("\n=== 场景对齐 ===")
        
        # 尝试解析用户指定的方向
        user_ground_normal, axis_name = SceneAligner.parse_user_axis(user_specified_up_axis)
        
        # 如果用户指定了有效的地面法向量方向，直接使用
        if user_ground_normal is not None and axis_name is not None:
            print(f"使用用户指定的地面法向量方向: {axis_name}轴")
            print(f"用户指定的地面法向量: [{user_ground_normal[0]:.3f}, {user_ground_normal[1]:.3f}, {user_ground_normal[2]:.3f}]")
            
            # 直接使用用户指定的地面法向量
            ground_normal = user_ground_normal
            target_normal = user_ground_normal  # 目标就是用户指定的方向
            
            user_specified = True
        else:
            # 自动检测最佳对齐方向
            print("使用自动检测的最佳对齐方向")
            
            # 首先估计当前的地面法向量
            ground_normal = SceneAligner.estimate_ground_plane(valid_points)
            print(f"自动检测的地面法向量: [{ground_normal[0]:.3f}, {ground_normal[1]:.3f}, {ground_normal[2]:.3f}]")
            
            # 然后找到最佳对齐轴
            target_normal, axis_name = SceneAligner.find_best_axis_alignment(ground_normal, extrinsics)
            print(f"选定目标法向量: [{target_normal[0]:.3f}, {target_normal[1]:.3f}, {target_normal[2]:.3f}] ({axis_name}轴)")
            
            user_specified = False
        
        # 计算旋转矩阵
        if np.allclose(ground_normal, target_normal):
            rotation_matrix = np.eye(3)
            print("地面法向量已经对齐，无需旋转")
        else:
            # 使用Rodrigues公式计算旋转矩阵
            axis = np.cross(ground_normal, target_normal)
            if np.linalg.norm(axis) < 1e-6:
                # 如果轴长度太小，使用单位矩阵
                rotation_matrix = np.eye(3)
                print("轴长度太小，使用单位矩阵")
            else:
                axis = axis / np.linalg.norm(axis)
                angle = np.arccos(np.clip(np.dot(ground_normal, target_normal), -1.0, 1.0))
                
                print(f"旋转轴: [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]")
                print(f"实际旋转角度: {np.degrees(angle):.1f} 度")
                
                # Rodrigues旋转公式
                K = np.array([[0, -axis[2], axis[1]],
                             [axis[2], 0, -axis[0]],
                             [-axis[1], axis[0], 0]])
                
                rotation_matrix = (np.eye(3) + 
                                 np.sin(angle) * K + 
                                 (1 - np.cos(angle)) * np.dot(K, K))
        
        # 构建4x4变换矩阵
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        
        # 创建对齐信息字典
        alignment_info = {
            'up_direction': target_normal,  # 这是地面的法向量，也就是"向上"方向
            'axis_name': axis_name,
            'rotation_matrix': rotation_matrix,
            'user_specified': user_specified
        }
        
        print(f"对齐信息:")
        print(f"  地面法向量(向上方向): {alignment_info['up_direction']}")
        print(f"  对齐轴: {alignment_info['axis_name']}")
        print(f"  用户指定: {alignment_info['user_specified']}")
        
        # 应用变换到点云
        aligned_world_points = world_points.copy()
        for s in range(world_points.shape[0]):
            for h in range(world_points.shape[1]):
                for w in range(world_points.shape[2]):
                    point = world_points[s, h, w]
                    if np.all(np.isfinite(point)):
                        aligned_point = np.dot(rotation_matrix, point)
                        aligned_world_points[s, h, w] = aligned_point
        
        # 应用变换到相机外参
        aligned_extrinsics = extrinsics.copy()
        for s in range(extrinsics.shape[0]):
            # 外参格式是世界到相机的变换 [R|t]
            R_world_to_cam = extrinsics[s, :3, :3]
            t_world_to_cam = extrinsics[s, :3, 3]
            
            # 当世界坐标系旋转时，世界到相机的变换需要相应调整
            # 新的世界到相机变换 = R_world_to_cam * rotation_matrix^T
            # 因为：新世界坐标 = rotation_matrix * 旧世界坐标
            # 所以：相机坐标 = R_world_to_cam * rotation_matrix^T * 新世界坐标 + t_world_to_cam
            R_aligned = np.dot(R_world_to_cam, rotation_matrix.T)
            t_aligned = t_world_to_cam  # 平移部分保持不变，因为只是坐标系旋转
            
            aligned_extrinsics[s, :3, :3] = R_aligned
            aligned_extrinsics[s, :3, 3] = t_aligned
        
        # 验证结果
        aligned_points_flat = aligned_world_points.reshape(-1, 3)
        aligned_valid_mask = np.all(np.isfinite(aligned_points_flat), axis=1)
        aligned_valid_points = aligned_points_flat[aligned_valid_mask]
        
        if len(aligned_valid_points) > 0:
            aligned_center = np.mean(aligned_valid_points, axis=0)
            print(f"对齐后场景中心: [{aligned_center[0]:.3f}, {aligned_center[1]:.3f}, {aligned_center[2]:.3f}]")
        
        return aligned_world_points, aligned_extrinsics, transform_matrix, alignment_info 