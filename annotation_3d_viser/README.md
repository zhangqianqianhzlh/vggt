# 3D目标检测和可视化工具包

本工具包是对原始`demo_viser_w_bbox.py`脚本的重构版本，将长脚本拆分成多个模块，使代码结构更加规范和易于维护。

## 文件结构

```
annotation_3d_viser/
├── __init__.py              # 包初始化文件，导出主要类和函数
├── config.py                # 配置文件，包含常量和配置加载
├── object_detector.py       # 目标检测器模块（Qwen视觉模型）
├── box_mapper.py           # 3D包围框映射器模块
├── image_utils.py          # 图像处理工具模块
├── visualizer.py           # Viser 3D可视化器模块
├── scene_aligner.py        # 场景对齐器模块
├── scene_renderer.py       # 3D场景渲染器模块
├── utils.py                # 工具函数模块
├── main.py                 # 主函数和流程控制模块
└── README.md               # 本文件
```

## 主要模块说明

### 1. `object_detector.py`
- **ObjectDetector类**: 使用Qwen视觉模型进行目标检测
- 支持单张图片和多张图片的多目标检测
- 包含预设的kitchen数据集检测结果

### 2. `box_mapper.py`
- **Box3DMapper类**: 将2D边框映射到3D包围框
- 分析相机设置和拍摄类型
- 构建标准正交坐标系的3D包围框

### 3. `visualizer.py`
- **ViserVisualizer类**: Viser 3D可视化器
- 支持交互式3D可视化
- 包含GUI控制界面和相机视锥体显示

### 4. `scene_aligner.py`
- **SceneAligner类**: 场景对齐器
- 估计地面平面并将场景对齐到重力方向

### 5. `scene_renderer.py`
- **Scene3DRenderer类**: 3D场景渲染器
- 从不同角度渲染并保存场景图片

### 6. `utils.py`
- 包含各种工具函数：
  - `save_colmap_reconstruction()`: 保存COLMAP格式重建结果
  - `apply_sky_segmentation()`: 应用天空分割
  - `save_multi_detection_results_json()`: 保存检测结果为JSON格式

### 7. `main.py`
- **main_pipeline()**: 主要的处理和可视化流程函数
- 整合所有模块，提供统一的接口

### 8. `config.py`
- 包含所有配置常量和默认值
- 加载LLM配置
- 预设的kitchen数据集检测结果

## 使用方法

### 方法1: 使用重构后的入口脚本
```bash
python demo_viser_w_bbox_refactored.py --image_folder /path/to/images
```

### 方法2: 直接导入模块使用
```python
from annotation_3d_viser import main_pipeline

# 使用main_pipeline函数
viser_server = main_pipeline(
    predictions,
    port=8080,
    image_folder="/path/to/images",
    object_names=["目标物体"],
    save_colmap=True
)
```

### 方法3: 使用单个模块
```python
from annotation_3d_viser import ObjectDetector, Box3DMapper, ViserVisualizer

# 创建检测器
detector = ObjectDetector(base_url="...", api_key="...")

# 检测目标
bboxes = detector.detect_multi_images_multi_objects(image_paths, ["目标"])

# 映射到3D框
box_info = Box3DMapper.map_to_3d_box(bboxes, world_points, depth_map, extrinsics, intrinsics)

# 可视化
visualizer = ViserVisualizer(port=8080)
visualizer.start_server()
visualizer.add_3d_box_no_centering(box_info)
```

## 优势

1. **模块化设计**: 每个文件负责一个主要功能，职责明确
2. **易于维护**: 代码结构清晰，便于修改和扩展
3. **可重用性**: 各个模块可以独立使用
4. **配置集中**: 所有配置参数集中在config.py中
5. **类型提示**: 完整的类型注解，提高代码质量
6. **文档完整**: 每个模块和函数都有详细的文档字符串

## 依赖关系

- `object_detector.py` → `config.py`
- `box_mapper.py` → `config.py`
- `visualizer.py` → `config.py`, `image_utils.py`, `box_mapper.py`
- `scene_aligner.py` → `config.py`
- `scene_renderer.py` → `config.py`, `box_mapper.py`
- `utils.py` → `config.py`
- `main.py` → 所有其他模块

## 注意事项

1. 确保所有依赖库已正确安装
2. 配置文件`env/llm.yaml`需要正确设置
3. VGGT模型路径需要正确配置
4. 如果使用天空分割，需要安装onnxruntime

