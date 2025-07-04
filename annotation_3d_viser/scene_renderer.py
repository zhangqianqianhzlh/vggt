"""
3D场景渲染器模块
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Any
from .box_mapper import Box3DMapper
from .config import DEFAULT_CONFIG


class Scene3DRenderer:
    """3D场景渲染器 - 生成俯瞰图"""
    
    @staticmethod
    def render_top_down_view(world_points: np.ndarray, 
                           box_info: Dict[str, Any], 
                           scene_center: np.ndarray,
                           images: np.ndarray,
                           output_dir: str = "3d_views",
                           point_size: float = 0.5) -> None:
        """
        生成俯瞰图（从地面向上的方向）
        """
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Noto Serif CJK JP']
        plt.rcParams['axes.unicode_minus'] = False
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n=== 生成俯瞰图 ===")
        print(f"场景中心: [{scene_center[0]:.3f}, {scene_center[1]:.3f}, {scene_center[2]:.3f}]")
        print(f"3D框中心: [{box_info['center'][0]:.3f}, {box_info['center'][1]:.3f}, {box_info['center'][2]:.3f}]")
        
        # box_info已经在中心化坐标系中，直接使用
        centered_box_info = box_info.copy()
        
        # 展平点云并应用中心化
        points_flat = world_points.reshape(-1, 3)
        valid_mask = np.all(np.isfinite(points_flat), axis=1)
        valid_points = points_flat[valid_mask] - scene_center  # 应用中心化
        
        # 获取对应的颜色信息
        colors = images.transpose(0, 2, 3, 1)  # (S, H, W, 3)
        colors_flat = colors.reshape(-1, 3)
        valid_colors = colors_flat[valid_mask]
        
        # 随机采样点云以提高渲染速度
        max_points = DEFAULT_CONFIG["max_render_points"]
        if len(valid_points) > max_points:
            indices = np.random.choice(len(valid_points), max_points, replace=False)
            valid_points = valid_points[indices]
            valid_colors = valid_colors[indices]
        
        # 创建3D框顶点
        box_vertices = Box3DMapper.create_3d_box_vertices(centered_box_info)
        
        # 获取向上方向
        up_dir = box_info['up_direction']
        main_dir = box_info['main_direction']
        
        # 计算相机位置（从上方俯瞰）
        box_size = np.max(box_info['size'])
        camera_distance = max(3.0, box_size * DEFAULT_CONFIG["camera_distance_multiplier"])
        
        # 俯瞰视角：从向上方向俯瞰
        view = {
            'name': 'top_down_view',
            'title': '俯瞰图',
            'camera_pos': centered_box_info['center'] + up_dir * camera_distance,
            'up_vector': main_dir  # 使用主方向作为"上"向量，确保正确的朝向
        }
        
        print(f"点云数量: {len(valid_points)}")
        print(f"向上方向: [{up_dir[0]:.3f}, {up_dir[1]:.3f}, {up_dir[2]:.3f}]")
        print(f"相机位置: [{view['camera_pos'][0]:.3f}, {view['camera_pos'][1]:.3f}, {view['camera_pos'][2]:.3f}]")
        
        # 渲染俯瞰图
        Scene3DRenderer._render_single_view(
            valid_points, valid_colors, box_vertices, centered_box_info, 
            view, output_dir, point_size
        )
        
        print(f"俯瞰图已保存到: {output_dir}")
    
    @staticmethod
    def _render_single_view(points: np.ndarray, 
                          colors: np.ndarray,
                          box_vertices: np.ndarray,
                          box_info: Dict[str, Any],
                          view: Dict[str, Any],
                          output_dir: str,
                          point_size: float) -> None:
        """渲染单个视角"""
        
        # 创建图形
        fig_size = DEFAULT_CONFIG["render_figure_size"]
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111, projection='3d')
        
        # 设置相机视角
        camera_pos = view['camera_pos']
        target_pos = box_info['center']
        up_vector = view['up_vector']
        
        # 绘制点云（使用真实颜色）
        if len(points) > 0:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c=colors, s=point_size, alpha=0.6, label='点云')
        
        # 绘制3D包围框
        Scene3DRenderer._draw_3d_box(ax, box_vertices, box_info)
        
        # 绘制坐标轴
        Scene3DRenderer._draw_coordinate_axes(ax, box_info['center'])
        
        # 设置视角
        # 计算视角参数
        view_vector = camera_pos - target_pos
        view_vector = view_vector / np.linalg.norm(view_vector)
        
        # 计算方位角和仰角
        azim = np.degrees(np.arctan2(view_vector[1], view_vector[0]))
        elev = np.degrees(np.arcsin(view_vector[2]))
        
        ax.view_init(elev=elev, azim=azim)
        
        # 设置坐标轴范围
        center = box_info['center']
        max_range = np.max(box_info['size']) * 2
        
        ax.set_xlim([center[0] - max_range, center[0] + max_range])
        ax.set_ylim([center[1] - max_range, center[1] + max_range])
        ax.set_zlim([center[2] - max_range, center[2] + max_range])
        
        # 设置标签和标题（中文）
        aligned_axis = box_info.get('aligned_axis', 'Z')
        ax.set_xlabel('X轴', fontsize=12)
        ax.set_ylabel('Y轴', fontsize=12)
        ax.set_zlabel('Z轴', fontsize=12)
        ax.set_title(f'{view["title"]} (地面对齐到{aligned_axis}轴)\n3D目标检测结果', fontsize=14, fontweight='bold')
        
        # 添加图例
        ax.legend(loc='upper right')
        
        # 添加信息文本（中文）
        info_text = f"""
3D框信息:
中心: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]
尺寸: [{box_info['size'][0]:.3f}, {box_info['size'][1]:.3f}, {box_info['size'][2]:.3f}]
旋转: {np.degrees(box_info['rotation_angle']):.1f}°
对齐轴: {aligned_axis}轴向上
方法: {box_info.get('method', 'unknown')}
        """
        
        ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, 
                 verticalalignment='top', fontsize=10, 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 保存图片
        output_path = os.path.join(output_dir, f"{view['name']}.png")
        dpi = DEFAULT_CONFIG["render_dpi"]
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"  已保存: {output_path}")
    
    @staticmethod
    def _draw_3d_box(ax, vertices: np.ndarray, box_info: Dict[str, Any]) -> None:
        """绘制3D包围框"""
        
        # 定义12条边的连接关系
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
            [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
            [0, 4], [1, 5], [2, 6], [3, 7],  # 连接线
        ]
        
        # 绘制边框
        for edge in edges:
            start, end = edge
            ax.plot3D([vertices[start, 0], vertices[end, 0]],
                     [vertices[start, 1], vertices[end, 1]],
                     [vertices[start, 2], vertices[end, 2]], 
                     'r-', linewidth=3, label='3D包围框' if edge == edges[0] else "")
        
        # 标记中心点
        center = box_info['center']
        ax.scatter([center[0]], [center[1]], [center[2]], 
                  c='red', s=100, marker='o', label='框中心')
        
        # 绘制方向向量
        scale = np.max(box_info['size']) * 0.6
        
        # 主方向（红色）
        main_end = center + box_info['main_direction'] * scale
        ax.plot3D([center[0], main_end[0]], [center[1], main_end[1]], [center[2], main_end[2]], 
                 'r-', linewidth=4, alpha=0.8, label='主方向')
        
        # 次方向（绿色）
        sec_end = center + box_info['secondary_direction'] * scale
        ax.plot3D([center[0], sec_end[0]], [center[1], sec_end[1]], [center[2], sec_end[2]], 
                 'g-', linewidth=4, alpha=0.8, label='次方向')
        
        # 上方向（蓝色）
        up_end = center + box_info['up_direction'] * scale
        ax.plot3D([center[0], up_end[0]], [center[1], up_end[1]], [center[2], up_end[2]], 
                 'b-', linewidth=4, alpha=0.8, label='上方向')
    
    @staticmethod
    def _draw_coordinate_axes(ax, origin: np.ndarray, length: float = 0.5) -> None:
        """绘制坐标轴"""
        
        # X轴（红色）
        ax.plot3D([origin[0], origin[0] + length], [origin[1], origin[1]], [origin[2], origin[2]], 
                 'r-', linewidth=2, alpha=0.7, label='X轴')
        
        # Y轴（绿色）
        ax.plot3D([origin[0], origin[0]], [origin[1], origin[1] + length], [origin[2], origin[2]], 
                 'g-', linewidth=2, alpha=0.7, label='Y轴')
        
        # Z轴（蓝色）
        ax.plot3D([origin[0], origin[0]], [origin[1], origin[1]], [origin[2], origin[2] + length], 
                 'b-', linewidth=2, alpha=0.7, label='Z轴') 