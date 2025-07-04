"""
配置文件
"""

import yaml
import os

# 加载LLM配置
def load_llm_config():
    """加载LLM配置"""
    config_path = "env/llm.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        print(f"警告: 配置文件 {config_path} 不存在")
        return {
            "llm_server": {
                "base_url": "http://localhost:8000/v1",
                "api_key": "dummy_key"
            }
        }

# 默认配置
DEFAULT_CONFIG = {
    "model_name": "Qwen/Qwen2.5-VL-72B-Instruct",
    "default_port": 8080,
    "default_conf_threshold": 25.0,
    "max_points_for_colmap": 100000,
    "conf_thres_value": 5.0,
    "point_size": 0.001,
    "camera_distance_multiplier": 4.0,
    "min_box_size": 0.02,
    "sampling_margin_ratio": 0.15,
    "depth_filter_threshold": 2.0,
    "ground_percentile": 20,
    "max_render_points": 50000,
    "render_dpi": 300,
    "render_figure_size": (12, 10)
}

# 预设的kitchen数据集检测结果
KITCHEN_PRESET_RESULTS = {
    "00.png": {"黄色小车": (182, 38, 455, 237)},
    "01.png": {"黄色小车": (164, 48, 420, 248)},
    "02.png": {"黄色小车": (152, 34, 335, 264)},
    "03.png": {"黄色小车": (177, 48, 372, 274)},
    "04.png": {"黄色小车": (104, 53, 332, 263)},
    "05.png": {"黄色小车": (182, 19, 345, 247)},
    "06.png": {"黄色小车": (148, 63, 375, 306)},
    "07.png": {"黄色小车": (202, 25, 353, 252)},
    "08.png": {"黄色小车": (187, 58, 434, 206)},
    "09.png": {"黄色小车": (128, 72, 356, 220)},
    "10.png": {"黄色小车": (203, 61, 399, 208)},
    "11.png": {"黄色小车": (189, 86, 394, 213)},
    "12.png": {"黄色小车": (209, 34, 350, 211)},
    "13.png": {"黄色小车": (203, 91, 293, 220)},
    "14.png": {"黄色小车": (184, 109, 337, 257)},
    "15.png": {"黄色小车": (190, 89, 385, 245)},
    "16.png": {"黄色小车": (127, 22, 382, 181)},
    "17.png": {"黄色小车": (131, 89, 345, 233)},
    "18.png": {"黄色小车": (155, 39, 392, 194)},
    "19.png": {"黄色小车": (114, 37, 430, 246)},
    "20.png": {"黄色小车": (142, 86, 322, 234)},
    "21.png": {"黄色小车": (237, 86, 362, 209)},
    "22.png": {"黄色小车": (145, 64, 425, 235)},
    "23.png": {"黄色小车": (99, 69, 374, 226)},
    "24.png": {"黄色小车": (65, 49, 304, 190)},
} 