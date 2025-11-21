import sys
import os

# 获取当前脚本的绝对路径，再向上跳转一级（回到项目根目录）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import time

from model.vit_config import VITConfig
from model.vit_model import VitModel
from model_weights.load_weight import load_dinov2_weights

# 测试用例
if __name__ == "__main__":
    print("===== Load weights_vit_base_224 =====")

    config = VITConfig()
    model = VitModel(config)
    start = time.time()
    weight_dir = os.path.join(project_root, "extract_weights/weights_vit_base_224")
    load_dinov2_weights(model, config, weight_dir)
    end = time.time()

    print("耗时: {:.4f} 秒".format(end - start))
    print(" ")
