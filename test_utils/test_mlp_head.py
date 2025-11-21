import sys
import os

# 获取当前脚本的绝对路径，再向上跳转一级（回到项目根目录）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import time
import numpy as np

from model.mlp_head import MLPHead
from model.vit_config import VITConfig

config = VITConfig()

# 测试用例
if __name__ == "__main__":
    print("===== MLPHead test =====")
    # 1. 初始化 MLPHead
    mlp_head = MLPHead(config)

    # 2. 模拟输入CLS向量（batch = 4）
    batch_size = 4
    cls_token = np.random.randn(batch_size, config.hidden_size).astype(np.float32)
    training = True

    # 3. 前向传播
    start = time.time()
    logits = mlp_head.forward(cls_token, training)  # [B, num_classes]
    end = time.time()

    # 4. 输出调试信息
    print("输入 CLS token 形状:", cls_token.shape)       # [4, 768]
    print("输出 logits 形状:", logits.shape)             # [4, 2]
    print("logits 值:\n", logits)
    print("耗时: {:.4f} 秒".format(end - start))

    # 5. 简单数值检查（softmax）
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exps / np.sum(exps, axis=1, keepdims=True)
    print("softmax 概率分布:\n", probs)
    print("每行是否为概率分布（和为1）:", np.allclose(np.sum(probs, axis=1), 1.0))
    print(" ")
