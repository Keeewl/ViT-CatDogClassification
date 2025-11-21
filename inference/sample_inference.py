import sys
import os

# 项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import argparse
import numpy as np
from PIL import Image
import time

from data.data_process import preprocess_image_pil
from model.vit_config import VITConfig
from model.vit_model import VitModel
from model.mlp_head import MLPHead
from model_weights.load_weight import load_dinov2_weights


def preprocess_to_npy(image_path: str) -> tuple[np.ndarray, str]:
    """读取单张图片，按 data_process 一致的流程预处理并保存为 .npy。"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"未找到图片: {image_path}")
    with Image.open(image_path) as img:
        processed = preprocess_image_pil(img).astype(np.float32)

    base = os.path.splitext(os.path.basename(image_path))[0]
    npy_path = os.path.join(os.path.dirname(image_path), f"{base}.npy")
    np.save(npy_path, processed)
    print(f"💾 预处理后的样本已保存为: {npy_path}")
    return processed, npy_path


def load_mlp_head_weights(head: MLPHead, weight_path: str) -> None:
    """将训练阶段保存的 MLP Head 权重加载到当前 head 实例。"""
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"未找到 MLP Head 权重文件: {weight_path}")
    weights = np.load(weight_path)
    head.W1 = weights["W1"]
    head.b1 = weights["b1"]
    head.W2 = weights["W2"]
    head.b2 = weights["b2"]
    print(f"✅ 成功加载 MLP Head 权重: {weight_path}")


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def run_inference(image_path: str) -> None:
    config = VITConfig()
    vit = VitModel(config)

    vit_weight_dir = os.path.join(project_root, "extract_weights", "weights_vit_base_224")
    print("🔄 加载 ViT 主干权重...")
    load_dinov2_weights(vit, config, vit_weight_dir)

    head = MLPHead(config)
    mlp_weight_path = os.path.join(project_root, "extract_weights", "mlp_head_trained_weights.npz")
    load_mlp_head_weights(head, mlp_weight_path)

    sample, npy_path = preprocess_to_npy(image_path)
    sample_batch = np.expand_dims(sample, axis=0)  # [1, 224, 224, 3]

    print("🚀 开始前向推理...")
    vit_outputs = vit.forward(sample_batch, training=False)
    cls_feature = vit_outputs[:, 0, :]  # [1, 768]
    logits = head.forward(cls_feature, training=False)
    probs = softmax(logits)[0]

    labels = ["Cat", "Dog"]
    print(f"\n📄 输入样本: {image_path}")
    print(f"📁 预处理 .npy: {npy_path}")
    for idx, label in enumerate(labels):
        print(f"{label} 概率: {probs[idx] * 100:.2f}%")

    predicted = labels[int(np.argmax(probs))]
    print(f"\n✅ 预测结果: {predicted}")


def parse_args():
    parser = argparse.ArgumentParser(description="单张图片推理 (Cat vs Dog)")
    default_image = os.path.join(project_root, "data", "sample", "mycat.jpg")
    parser.add_argument(
        "--image",
        type=str,
        default=default_image,
        help="待推理的图片路径，默认 data/sample 下的样本",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    time_start = time.time()
    run_inference(args.image)
    time_end = time.time()
    print(f"\n⏱️ 总推理时间: {time_end - time_start:.4f} 秒")