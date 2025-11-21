import sys
import os

# 获取当前脚本的绝对路径，再向上跳转一级（回到项目根目录）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import numpy as np
import time
import os

from model.vit_config import VITConfig
from model.vit_model import VitModel
from model_weights.load_weight import load_dinov2_weights


# === ViT FeatureExtractor ===
class FeatureExtractor:
    def __init__(self, vit_model, output_dir="features"):
        """
        ViT特征提取模块
        :param vit_model: ViTModel 实例（已加载权重的ViT主干）
        :param output_dir: 特征保存目录
        """
        self.vit = vit_model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def extract_features(self, images, labels, prefix="train", batch_size = 32):
        """
        批量提取图像特征并保存为.npy文件
        :param images: 图像数组 [N, H, W, C]
        :param labels: 标签数组 [N]
        :param prefix: 文件名前缀 (e.g., "train", "test")
        :return: 提取的特征数组 [N, 768]
        """
        features = []
        N = len(images)
        # batch_size = 32  # 根据内存调整

        print(f"⏳ 开始提取 {prefix} 集特征...")
        start_time = time.time()

        # 分批处理避免内存溢出
        for i in range(0, N, batch_size):
            batch = images[i:i + batch_size]

            # 前向传播提取特征
            outputs = self.vit.forward(batch, training=False)  # [B, 257, 768]
            cls_features = outputs[:, 0, :]  # [B, 768]
            features.append(cls_features)

            # 进度提示
            if (i // batch_size) % 10 == 0:
                print(f"  ✅ 已处理 {min(i + batch_size, N)}/{N} 张图像")

        # 合并所有批次特征
        features = np.concatenate(features, axis=0)
        elapsed = time.time() - start_time
        print(f"✨ {prefix}集特征提取完成! 形状: {features.shape}, 耗时: {elapsed:.2f}秒")

        # 保存特征和标签
        feature_path = os.path.join(self.output_dir, f"{prefix}_features.npy")
        label_path = os.path.join(self.output_dir, f"{prefix}_labels.npy")
        np.save(feature_path, features)
        np.save(label_path, labels)
        print(f"💾 特征已保存至: {feature_path}")
        print(f"💾 标签已保存至: {label_path}")

        return features


# === 测试模式：用50个样本验证特征提取流程 ===
if __name__ == "__main__":
    # 开启测试模式提示
    print("===== 【测试模式】用50个样本验证特征提取 =====")

    # 初始化配置和模型
    config = VITConfig()
    model = VitModel(config)

    # 加载预训练权重
    print("🔄 加载预训练权重...")
    weight_dir = os.path.join(project_root, "extract_weights/weights_vit_base_224")
    load_dinov2_weights(model, config, weight_dir)

    # 加载全量数据后，截取前50个样本（测试用）
    print("📥 加载数据并截取前50个样本...")
    X_train_full = np.load("data/data_to_npy/X_train.npy")
    y_train_full = np.load("data/data_to_npy/y_train.npy")
    X_test_full = np.load("data/data_to_npy/X_test.npy")
    y_test_full = np.load("data/data_to_npy/y_test.npy")

    # 截取前50个样本（如果数据本身不足50个，取全部）
    sample_num = 50
    X_train = X_train_full[:sample_num]
    y_train = y_train_full[:sample_num]
    X_test = X_test_full[:sample_num]
    y_test = y_test_full[:sample_num]

    # 打印测试数据形状，确认截取正确
    print(f"📊 测试数据形状：")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")

    # 初始化特征提取器（输出目录改为测试专用，避免覆盖正式数据）
    test_output_dir = "feature_test_50samples"
    feature_extractor = FeatureExtractor(model, test_output_dir)
    
    # 提取测试样本的特征
    train_feature = feature_extractor.extract_features(X_train, y_train, prefix="my_train")
    test_feature = feature_extractor.extract_features(X_test, y_test, prefix="my_test")
    
    # 验证输出结果
    print("\n🔍 测试特征验证:")
    print(f"raw   : {train_feature.shape}, {test_feature.shape}")

    # 后续训练程序测试提示
    print("\n✅ 50个样本的特征提取测试完成！")
    print("👉 后续训练程序可修改输入路径为上述测试目录，快速验证训练流程。")



# # === 全部提取特征 ===
# if __name__ == "__main__":
#     # 提示
#     print("===== 全部特征提取 =====")

#     # 初始化配置和模型
#     config = VITConfig()
#     model = VitModel(config)

#     # 加载预训练权重
#     print("🔄 加载预训练权重...")
#     weight_dir = os.path.join(project_root, "extract_weights/weights_vit_base_224")
#     load_dinov2_weights(model, config, weight_dir)

#     # 加载全量数据
#     X_train = np.load("data/data_to_npy/X_train.npy")
#     y_train = np.load("data/data_to_npy/y_train.npy")
#     X_test = np.load("data/data_to_npy/X_test.npy")
#     y_test = np.load("data/data_to_npy/y_test.npy")

#     # 打印测试数据形状，确认截取正确
#     print(f"📊 测试数据形状：")
#     print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
#     print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")

#     # 初始化特征提取器（输出目录改为测试专用，避免覆盖正式数据）
#     test_output_dir = "new_feature"
#     feature_extractor = FeatureExtractor(model, test_output_dir)
    
#     # 提取测试样本的特征
#     train_feature = feature_extractor.extract_features(X_train, y_train, prefix="train")
#     test_feature = feature_extractor.extract_features(X_test, y_test, prefix="test")
    
#     # 验证输出结果
#     print("\n🔍 测试特征验证:")
#     print(f"raw   : {train_feature.shape}, {test_feature.shape}")

#     # 后续训练程序测试提示
#     print("\n✅ 全部样本的特征提取测试完成！")
