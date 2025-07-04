"""
Viser 3D可视化器模块
"""

import time
import numpy as np
import viser
import viser.transforms as viser_tf
from typing import List, Optional, Dict, Any
from .box_mapper import Box3DMapper
from .image_utils import ImageUtils
from .config import DEFAULT_CONFIG


class ViserVisualizer:
    """Viser 3D可视化器"""
    
    def __init__(self, port: int = None):
        self.port = port if port is not None else DEFAULT_CONFIG["default_port"]
        self.server = None
        self.frames = []
        self.frustums = []
    
    def start_server(self):
        """启动Viser服务器"""
        self.server = viser.ViserServer(port=self.port)
        print(f"Viser服务器已启动，端口: {self.port}")
        print(f"请在浏览器中打开: http://localhost:{self.port}")
        
        # 添加自定义坐标轴
        self.add_custom_coordinate_axes()
        
        # 等待一下让服务器完全启动
        time.sleep(0.5)
    
    def add_custom_coordinate_axes(self):
        """添加自定义的坐标轴和标签"""
        
        # 坐标轴长度
        axis_length = 1.0
        
        # X轴（红色）
        self.server.scene.add_line_segments(
            name="x_axis",
            points=np.array([[[0, 0, 0], [axis_length, 0, 0]]]),
            colors=(255, 0, 0),
            line_width=3.0
        )
        
        # Y轴（绿色）
        self.server.scene.add_line_segments(
            name="y_axis", 
            points=np.array([[[0, 0, 0], [0, axis_length, 0]]]),
            colors=(0, 255, 0),
            line_width=3.0
        )
        
        # Z轴（蓝色）
        self.server.scene.add_line_segments(
            name="z_axis",
            points=np.array([[[0, 0, 0], [0, 0, axis_length]]]),
            colors=(0, 0, 255),
            line_width=3.0
        )
        
        # 原点标记
        self.server.scene.add_point_cloud(
            name="origin",
            points=np.array([[0, 0, 0]]),
            colors=np.array([[255, 255, 255]]),
            point_size=0.02
        )
        
        # 坐标轴标签
        self.server.scene.add_label(
            name="x_label",
            text="X轴",
            position=(axis_length * 1.1, 0, 0),
            visible=True
        )
        
        self.server.scene.add_label(
            name="y_label", 
            text="Y轴",
            position=(0, axis_length * 1.1, 0),
            visible=True
        )
        
        self.server.scene.add_label(
            name="z_label",
            text="Z轴", 
            position=(0, 0, axis_length * 1.1),
            visible=True
        )
        
        self.server.scene.add_label(
            name="origin_label",
            text="原点(0,0,0)",
            position=(0.1, 0.1, 0.1),
            visible=True
        )
        
        print("自定义坐标轴和标签已添加")
        print("- X轴: 红色，标记为'X轴'")
        print("- Y轴: 绿色，标记为'Y轴'") 
        print("- Z轴: 蓝色，标记为'Z轴'")
        print("- 原点: 白色点，标记为'原点(0,0,0)'")
    
    def visualize_frames_with_bbox(self, extrinsics: np.ndarray, images: np.ndarray, 
                                 bboxes: List[Optional[tuple]], 
                                 scene_alignment_info: Dict[str, Any] = None) -> None:
        """
        可视化相机帧和边界框
        
        Args:
            extrinsics: 相机到世界的变换矩阵 (S, 3, 4) - cam_to_world格式
            images: 图片数组 (S, 3, H, W)
            bboxes: 每个图片的边界框列表
            scene_alignment_info: 场景对齐信息
        """
        print(f"可视化 {len(images)} 个相机帧")
        
        def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in self.server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        # 获取场景对齐信息
        if scene_alignment_info is not None:
            up_direction = scene_alignment_info['up_direction']
            axis_name = scene_alignment_info['axis_name']
            rotation_matrix = scene_alignment_info['rotation_matrix']
            print(f"使用场景对齐信息: {axis_name}轴向上")
        else:
            up_direction = np.array([0.0, 0.0, 1.0])
            axis_name = "Z"
            rotation_matrix = np.eye(3)
            print("使用默认坐标系: Z轴向上")

        for img_id in range(len(images)):
            cam2world_3x4 = extrinsics[img_id]  # cam_to_world格式 (3, 4)
            
            # 构建4x4变换矩阵
            cam2world_4x4 = np.eye(4)
            cam2world_4x4[:3, :] = cam2world_3x4
            
            # 创建SE3变换
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_4x4)

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
        
        print(f"相机位姿已调整以匹配{axis_name}轴向上的坐标系")
    
    def add_3d_box_no_centering(self, box_info: Dict[str, Any]) -> None:
        """添加3D包围框到场景（不进行中心化，因为box_info已经在中心化坐标系中）"""
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
        
        # 添加3D文本标签
        object_name = box_info.get('object_name', '检测目标')
        
        # 在3D框上方添加标签（使用向上方向）
        up_direction = box_info.get('up_direction', np.array([0.0, 0.0, 1.0]))
        label_position = box_info['center'].copy()
        label_position += up_direction * box_info['size'][2] * 0.6  # 沿向上方向偏移
        
        self.server.scene.add_label(
            name="object_label",
            text=object_name,
            position=label_position,
            visible=True
        )
        
        # 添加方向向量可视化
        center = box_info['center']
        scale = np.max(box_info['size']) * 0.4
        
        # 主方向（红色）
        main_end = center + box_info['main_direction'] * scale
        self.server.scene.add_line_segments(
            name="main_direction",
            points=np.array([[center, main_end]]),
            colors=(255, 100, 100),
            line_width=3.0
        )
        
        # 次方向（绿色）
        sec_end = center + box_info['secondary_direction'] * scale
        self.server.scene.add_line_segments(
            name="secondary_direction",
            points=np.array([[center, sec_end]]),
            colors=(100, 255, 100),
            line_width=3.0
        )
        
        # 上方向（蓝色）
        up_end = center + up_direction * scale
        self.server.scene.add_line_segments(
            name="up_direction",
            points=np.array([[center, up_end]]),
            colors=(100, 100, 255),
            line_width=3.0
        )
        
        aligned_axis = box_info.get('aligned_axis', 'Z')
        print("3D包围框和标签可视化完成（中心化坐标系）")
        print(f"- 目标标签: {object_name}")
        print(f"- 3D框中心: [{box_info['center'][0]:.3f}, {box_info['center'][1]:.3f}, {box_info['center'][2]:.3f}]")
        print(f"- 地面对齐轴: {aligned_axis}")
        print(f"- 向上方向: [{up_direction[0]:.3f}, {up_direction[1]:.3f}, {up_direction[2]:.3f}]")

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
    
    def create_gui_controls(self, points_centered: np.ndarray, colors_flat: np.ndarray, 
                          conf_flat: np.ndarray, frame_indices: np.ndarray, 
                          init_conf_threshold: float, S: int):
        """创建GUI控制界面"""
        # 构建GUI
        gui_show_frames = self.server.gui.add_checkbox("Show Cameras", initial_value=True)
        gui_points_conf = self.server.gui.add_slider(
            "Confidence Percent", min=0, max=100, step=0.1, initial_value=init_conf_threshold
        )
        gui_frame_selector = self.server.gui.add_dropdown(
            "Show Points from Frames", options=["All"] + [str(i) for i in range(S)], initial_value="All"
        )

        # 创建点云
        init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
        init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)
        point_cloud = self.server.scene.add_point_cloud(
            name="viser_pcd",
            points=points_centered[init_conf_mask],
            colors=colors_flat[init_conf_mask],
            point_size=DEFAULT_CONFIG["point_size"],
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
            for f in self.frames:
                f.visible = gui_show_frames.value
            for fr in self.frustums:
                fr.visible = gui_show_frames.value

        return gui_show_frames, gui_points_conf, gui_frame_selector, point_cloud 