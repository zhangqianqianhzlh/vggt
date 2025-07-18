#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COLMAP可视化器 - 基于viser官方示例
使用viser可视化COLMAP稀疏重建输出，并支持显示检测结果中的3D包围框

用法:
python colmap_visualizer.py --colmap_path /path/to/colmap/sparse/0 --images_path /path/to/images --detection_json /path/to/detection_results.json
"""

import random
import time
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
import argparse

import imageio.v3 as iio
import numpy as np
from tqdm.auto import tqdm
import cv2  # 添加OpenCV用于绘制2D框

import viser
import viser.transforms as vtf
from viser.extras.colmap import (
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
)


def reconstruct_3d_box_vertices(center: List[float], size: List[float], rotation_angle: float) -> np.ndarray:
    """
    从简化信息重建3D包围框的8个顶点
    
    Args:
        center: 3D框中心坐标 [x, y, z]
        size: 3D框尺寸 [width, length, height]
        rotation_angle: 绕X轴的旋转角度（弧度）
        
    Returns:
        8个顶点的坐标 (8, 3)
    """
    center = np.array(center)
    size = np.array(size)
    
    # 固定的方向向量（与原始函数保持一致）
    up_direction = np.array([1.0, 0.0, 0.0])  # X轴向上
    
    # 根据旋转角度计算主方向和次方向
    main_direction = np.array([0.0, np.cos(rotation_angle), np.sin(rotation_angle)])
    secondary_direction = np.array([0.0, -np.sin(rotation_angle), np.cos(rotation_angle)])
    
    # 归一化
    main_direction = main_direction / np.linalg.norm(main_direction)
    secondary_direction = secondary_direction / np.linalg.norm(secondary_direction)
    up_direction = up_direction / np.linalg.norm(up_direction)
    
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
    
    # 构建旋转矩阵 - 修正：与原始函数保持一致的顺序
    rotation_matrix = np.column_stack([main_direction, secondary_direction, up_direction])
    
    # 转换到世界坐标
    world_vertices = np.dot(local_vertices, rotation_matrix.T) + center
    
    return world_vertices


def draw_2d_boxes_on_image(image: np.ndarray, detections: Dict[str, Any], 
                          downsample_factor: int = 1) -> np.ndarray:
    """
    在图像上绘制2D目标框
    
    Args:
        image: 输入图像 (H, W, 3)
        detections: 该图像的检测结果，格式为 {"objects": {"目标名": {"bbox": [x1,y1,x2,y2], "score": 1.0}}}
        downsample_factor: 下采样因子
        
    Returns:
        绘制了目标框的图像
    """
    if not detections or not detections.get('objects'):
        return image
    
    # 复制图像避免修改原图
    image_with_boxes = image.copy()
    
    # 定义颜色列表 (BGR格式，用于OpenCV) - 全部改为红色
    colors = [
        (0, 0, 255),    # 红色
        (0, 0, 255),    # 红色  
        (0, 0, 255),    # 红色
        (0, 0, 255),    # 红色
        (0, 0, 255),    # 红色
        (0, 0, 255),    # 红色
    ]
    
    color_idx = 0
    for obj_name, obj_info in detections['objects'].items():
        if obj_info is None:  # 该图像中没有检测到此目标
            continue
            
        bbox = obj_info['bbox']  # [x1, y1, x2, y2] 格式
        confidence = obj_info.get('score', 1.0)  # 使用'score'而不是'confidence'
        
        # 打印原始2D坐标
        print(f"目标 '{obj_name}' 的2D坐标:")
        print(f"  原始坐标: x1={bbox[0]}, y1={bbox[1]}, x2={bbox[2]}, y2={bbox[3]}")
        print(f"  框尺寸: 宽度={bbox[2]-bbox[0]}, 高度={bbox[3]-bbox[1]}")
        
        # 根据下采样因子调整坐标
        x1 = int(bbox[0] / downsample_factor)
        y1 = int(bbox[1] / downsample_factor)
        x2 = int(bbox[2] / downsample_factor)
        y2 = int(bbox[3] / downsample_factor)
        
        # 打印调整后的坐标
        print(f"  下采样后坐标 (因子={downsample_factor}): x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        
        # 确保坐标在图像范围内
        h, w = image_with_boxes.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        
        print(f"  最终坐标 (图像范围内): x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        print(f"  图像尺寸: 高度={h}, 宽度={w}")
        
        # 使用红色
        color = (0, 0, 255)  # 红色 (BGR格式)
        
        # 绘制矩形框
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 2)
        
        # 准备标签文本
        if confidence < 1.0:
            label_text = f"{obj_name} ({confidence:.2f})"
        else:
            label_text = obj_name
        
        # 计算文本尺寸
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, font, font_scale, thickness
        )
        
        # 绘制标签背景
        label_y = max(y1 - 10, text_height + 10)
        cv2.rectangle(
            image_with_boxes,
            (x1, label_y - text_height - baseline - 5),
            (x1 + text_width + 10, label_y + baseline + 5),
            color,
            -1  # 填充
        )
        
        # 绘制标签文本
        cv2.putText(
            image_with_boxes,
            label_text,
            (x1 + 5, label_y - 5),
            font,
            font_scale,
            (255, 255, 255),  # 白色文字
            thickness,
            cv2.LINE_AA
        )
    
    return image_with_boxes


def load_detection_results(json_path: Path) -> Optional[Dict[str, Any]]:
    """
    加载检测结果JSON文件
    
    Args:
        json_path: JSON文件路径
        
    Returns:
        检测结果字典或None
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功加载检测结果: {json_path}")
        print(f"- 检测到的目标: {data['metadata']['object_names']}")
        print(f"- 3D框数量: {len(data['3d_bounding_boxes'])}")
        return data
    except Exception as e:
        print(f"加载检测结果失败: {e}")
        return None


def visualize_3d_boxes(server: viser.ViserServer, detection_data: Dict[str, Any]) -> List[viser.SceneNodeHandle]:
    """
    可视化3D包围框
    
    Args:
        server: viser服务器
        detection_data: 检测结果数据
        
    Returns:
        创建的场景节点列表
    """
    scene_nodes = []
    
    # 定义颜色列表
    colors = [
        (255, 0, 0),    # 红色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 紫色
        (0, 255, 255),  # 青色
    ]
    
    for i, (object_name, box_info) in enumerate(detection_data['3d_bounding_boxes'].items()):
        color = colors[i % len(colors)]
        
        print(f"可视化3D框: {object_name}")
        print(f"- 中心: {box_info['center']}")
        print(f"- 尺寸: {box_info['size']}")
        print(f"- 旋转角度: {box_info['rotation_angle']:.3f} 弧度")
        
        # 重建3D框顶点
        box_vertices = reconstruct_3d_box_vertices(
            box_info['center'], 
            box_info['size'], 
            box_info['rotation_angle']
        )
        
        # 定义3D框的边
        box_edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
            [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
            [0, 4], [1, 5], [2, 6], [3, 7],  # 连接线
        ])
        
        # 创建线段
        line_segments = np.array([
            [box_vertices[edge[0]], box_vertices[edge[1]]] for edge in box_edges
        ])
        
        # 添加3D框线段
        box_lines = server.scene.add_line_segments(
            name=f"/detection/3d_box_{object_name}",
            points=line_segments,
            colors=color,
            line_width=4.0
        )
        scene_nodes.append(box_lines)
        
        # 添加中心点
        center_point = server.scene.add_point_cloud(
            name=f"/detection/center_{object_name}",
            points=np.array([box_info['center']]),
            colors=np.array([color]),
            point_size=0.02
        )
        scene_nodes.append(center_point)
        
        # 添加标签
        label_position = np.array(box_info['center'])
        label_position[2] += box_info['size'][2] * 0.6  # 在框上方
        
        label = server.scene.add_label(
            name=f"/detection/label_{object_name}",
            text=object_name,
            position=label_position,
            visible=True
        )
        scene_nodes.append(label)
    
    return scene_nodes


def main(
    colmap_path: Path,
    images_path: Path,
    detection_json: Optional[Path] = None,
    downsample_factor: int = 2,
    reorient_scene: bool = True,
    port: int = 8080,
) -> None:
    """可视化COLMAP稀疏重建输出和3D检测框
    
    Args:
        colmap_path: COLMAP重建目录路径 (包含cameras.bin, images.bin, points3D.bin)
        images_path: COLMAP图像目录路径
        detection_json: 检测结果JSON文件路径
        downsample_factor: 图像下采样因子
        reorient_scene: 是否重新定向场景
        port: viser服务器端口
    """
    print(f"启动viser服务器，端口: {port}")
    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    print("加载COLMAP数据...")
    # 加载COLMAP信息
    cameras = read_cameras_binary(colmap_path / "cameras.bin")
    images = read_images_binary(colmap_path / "images.bin")
    points3d = read_points3d_binary(colmap_path / "points3D.bin")

    print(f"加载了 {len(cameras)} 个相机, {len(images)} 张图片, {len(points3d)} 个3D点")

    # 提取3D点和颜色
    points = np.array([points3d[p_id].xyz for p_id in points3d])
    colors = np.array([points3d[p_id].rgb for p_id in points3d])

    # 加载检测结果
    detection_data = None
    if detection_json and detection_json.exists():
        detection_data = load_detection_results(detection_json)

    # GUI控件
    gui_reset_up = server.gui.add_button(
        "重置上方向",
        hint="将相机控制的'上'方向设置为当前相机的'上'方向",
    )

    # 3D框显示控制
    gui_show_boxes = server.gui.add_checkbox(
        "显示3D检测框", 
        initial_value=True,
        disabled=detection_data is None
    )

    # 2D框显示控制
    gui_show_2d_boxes = server.gui.add_checkbox(
        "显示2D检测框", 
        initial_value=True,
        disabled=detection_data is None
    )

    # 重新定向场景，使平均相机方向指向上方
    if reorient_scene:
        print("重新定向场景...")
        average_up = (
            vtf.SO3(np.array([img.qvec for img in images.values()]))
            @ np.array([0.0, -1.0, 0.0])  # 局部坐标系中-y是上方向
        ).mean(axis=0)
        average_up /= np.linalg.norm(average_up)
        server.scene.set_up_direction((average_up[0], average_up[1], average_up[2]))

    @gui_reset_up.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None
        client.camera.up_direction = vtf.SO3(client.camera.wxyz) @ np.array(
            [0.0, -1.0, 0.0]
        )

    # 控制显示的点和帧数量
    gui_points = server.gui.add_slider(
        "最大点数",
        min=1,
        max=len(points3d),
        step=1,
        initial_value=min(len(points3d), 50_000),
    )
    gui_frames = server.gui.add_slider(
        "最大帧数",
        min=1,
        max=len(images),
        step=1,
        initial_value=min(len(images), 100),
    )
    gui_point_size = server.gui.add_slider(
        "点大小", min=0.001, max=0.1, step=0.001, initial_value=0.002
    )

    # 初始化点云
    point_mask = np.random.choice(points.shape[0], gui_points.value, replace=False)
    point_cloud = server.scene.add_point_cloud(
        name="/colmap/pcd",
        points=points[point_mask],
        colors=colors[point_mask],
        point_size=gui_point_size.value,
    )
    frames: List[viser.FrameHandle] = []
    
    # 可视化3D框
    box_nodes = []
    if detection_data:
        box_nodes = visualize_3d_boxes(server, detection_data)

    @gui_show_boxes.on_update
    def _(_) -> None:
        """控制3D框显示/隐藏"""
        for node in box_nodes:
            node.visible = gui_show_boxes.value

    @gui_show_2d_boxes.on_update
    def _(_) -> None:
        """控制2D框显示/隐藏"""
        nonlocal need_update
        need_update = True  # 需要重新渲染图像

    def visualize_frames() -> None:
        """将所有COLMAP元素发送到viser进行可视化"""
        print("可视化相机帧...")
        
        # 移除现有的图像帧
        for frame in frames:
            frame.remove()
        frames.clear()

        # 解释图像和相机
        img_ids = [im.id for im in images.values()]
        random.shuffle(img_ids)
        img_ids = sorted(img_ids[: gui_frames.value])

        for img_id in tqdm(img_ids, desc="处理图像"):
            img = images[img_id]
            cam = cameras[img.camera_id]

            # 跳过不存在的图像
            image_filename = images_path / img.name
            if not image_filename.exists():
                print(f"警告: 图像文件不存在: {image_filename}")
                continue

            # 计算相机到世界的变换矩阵
            T_world_camera = vtf.SE3.from_rotation_and_translation(
                vtf.SO3(img.qvec), img.tvec
            ).inverse()
            
            # 添加相机坐标轴
            frame = server.scene.add_frame(
                f"/colmap/frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.1,
                axes_radius=0.005,
            )
            frames.append(frame)

            # 处理针孔相机模型
            if cam.model != "PINHOLE":
                print(f"预期针孔相机，但得到了 {cam.model}")

            H, W = cam.height, cam.width
            fy = cam.params[1]
            
            # 加载和下采样图像
            try:
                image = iio.imread(image_filename)
                image = image[::downsample_factor, ::downsample_factor]
                
                # 如果有检测结果，在图像上绘制2D框
                if detection_data and gui_show_2d_boxes.value:
                    # 查找对应的2D检测结果
                    image_detections = None
                    for detection in detection_data.get('2d_detections', []):
                        if detection['image_name'] == img.name:
                            image_detections = detection
                            break
                    
                    if image_detections:
                        print(f"为图像 {img.name} 绘制2D框: {list(image_detections.get('objects', {}).keys())}")
                        print(f"图像尺寸: {image.shape}")  # 添加图像尺寸信息
                        image = draw_2d_boxes_on_image(image, image_detections, downsample_factor)
                
                # 添加相机视锥体
                frustum = server.scene.add_camera_frustum(
                    f"/colmap/frame_{img_id}/frustum",
                    fov=2 * np.arctan2(H / 2, fy),
                    aspect=W / H,
                    scale=0.15,
                    image=image,
                )

                @frustum.on_click
                def _(_, frame=frame) -> None:
                    for client in server.get_clients().values():
                        client.camera.wxyz = frame.wxyz
                        client.camera.position = frame.position
                        
            except Exception as e:
                print(f"加载图像失败 {image_filename}: {e}")

    need_update = True

    @gui_points.on_update
    def _(_) -> None:
        point_mask = np.random.choice(points.shape[0], gui_points.value, replace=False)
        with server.atomic():
            point_cloud.points = points[point_mask]
            point_cloud.colors = colors[point_mask]

    @gui_frames.on_update
    def _(_) -> None:
        nonlocal need_update
        need_update = True

    @gui_point_size.on_update
    def _(_) -> None:
        point_cloud.point_size = gui_point_size.value

    print("开始可视化循环...")
    print(f"在浏览器中打开: http://localhost:{port}")
    
    while True:
        if need_update:
            need_update = False
            visualize_frames()

        time.sleep(1e-3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COLMAP可视化器（支持3D检测框）")
    parser.add_argument(
        "--colmap_path", 
        type=Path, 
        default="/data3/qq/proj2/3d_rec/vggt/examples/sample1/sparse",
        help="COLMAP重建目录路径 (包含cameras.bin, images.bin, points3D.bin)"
    )
    parser.add_argument(
        "--images_path", 
        type=Path, 
        default="/data3/qq/proj2/3d_rec/vggt/examples/sample1/images",
        help="COLMAP图像目录路径"
    )
    parser.add_argument(
        "--detection_json", 
        type=Path, 
        help="检测结果JSON文件路径"
    )
    parser.add_argument(
        "--downsample_factor", 
        type=int, 
        default=2,
        help="图像下采样因子"
    )
    parser.add_argument(
        "--no_reorient", 
        action="store_true",
        help="不重新定向场景"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8080,
        help="viser服务器端口"
    )

    args = parser.parse_args()
    
    # 检查路径是否存在
    if not args.colmap_path.exists():
        print(f"错误: COLMAP路径不存在: {args.colmap_path}")
        exit(1)
    
    if not args.images_path.exists():
        print(f"错误: 图像路径不存在: {args.images_path}")
        exit(1)
    
    # 检查必需的COLMAP文件
    required_files = ["cameras.bin", "images.bin", "points3D.bin"]
    missing_files = []
    for file in required_files:
        if not (args.colmap_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"错误: 缺少COLMAP文件: {missing_files}")
        exit(1)
    
    # 检查检测结果JSON文件
    if args.detection_json and not args.detection_json.exists():
        print(f"警告: 检测结果JSON文件不存在: {args.detection_json}")
        args.detection_json = None
    
    main(
        colmap_path=args.colmap_path,
        images_path=args.images_path,
        detection_json=args.detection_json,
        downsample_factor=args.downsample_factor,
        reorient_scene=not args.no_reorient,
        port=args.port,
    ) 