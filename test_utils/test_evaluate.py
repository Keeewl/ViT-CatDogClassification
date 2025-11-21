import sys
import os

# 获取当前脚本的绝对路径，再向上跳转一级（回到项目根目录）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import numpy as np
import time

from evaluate.eval import cross_entropy_loss, accuracy

# 损失函数与准确率测试用例
if __name__ == "__main__":
    print("===== 损失函数与准确率测试 =====")
    logits = np.array([[1.0, 2.0], [0.5, -1.0]])  # 模拟logits
    labels = np.array([1, 0])  # 真实标签
    start = time.time()
    loss = cross_entropy_loss(logits, labels)
    acc = accuracy(logits, labels)
    end = time.time()
    print("Logits:", logits)
    print("Labels:", labels)
    print("Loss:", loss)
    print("Accuracy:", acc)
    print("耗时: {:.4f} 秒".format(end - start))
    print(" ")