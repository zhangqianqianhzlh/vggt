#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VGGT 3D目标检测和可视化演示脚本（重构版）

主要功能：
1. 使用VGGT模型从多张图片生成3D重建
2. 使用Qwen模型检测图片中的目标物体
3. 将2D检测框映射到3D空间，生成水平对齐的3D包围框
4. 使用viser进行交互式3D可视化
5. 保存COLMAP格式的重建结果

作者：Facebook Research & 修改者
"""

import os
import glob
import argparse
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# 导入重构后的模块
from annotation_3d_viser import main_pipeline


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="VGGT 3D目标检测和可视化演示（重构版）")
    parser.add_argument("--image_folder", type=str, default="/data3/qq/proj2/3d_rec/vggt/examples/kitchen/images/", help="包含图片的文件夹路径")
    parser.add_argument("--use_point_map", action="store_true", help="使用点图而不是基于深度的点")
    parser.add_argument("--background_mode", action="store_true", help="在后台模式运行viser服务器")
    parser.add_argument("--port", type=int, default=8080, help="viser服务器端口号")
    parser.add_argument("--conf_threshold", type=float, default=25.0, help="初始置信度阈值百分比")
    parser.add_argument("--mask_sky", action="store_true", help="应用天空分割过滤天空点")
    parser.add_argument("--object_names", type=str, nargs='+', default=["黄色小车"], help="要检测的目标物体名称列表，用空格分隔")
    parser.add_argument("--save_colmap", action="store_true", help="保存colmap重建结果")
    parser.add_argument("--colmap_path", type=str, default="examples/kitchen_simple/sparse", help="colmap重建结果路径")
    parser.add_argument("--align_to_gravity", action="store_true", default=True, help="将场景对齐到重力方向")
    parser.add_argument("--save_3d_views", action="store_true", default=True, help="保存3D场景的三个方向视图")

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
    print(f"图片尺寸详细信息: {images.shape}")

    # 运行推理
    print("运行推理...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    # 在predictions获取后立即添加调试信息：
    print("=== VGGT模型输出尺寸调试 ===")
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")

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
        object_names=args.object_names,
        save_colmap=args.save_colmap,
        colmap_path=args.colmap_path,
        align_to_gravity=args.align_to_gravity,
        save_3d_views=args.save_3d_views,
    )
    print("可视化完成")


if __name__ == "__main__":  
    main() 