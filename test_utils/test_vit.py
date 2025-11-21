import sys
import os

# 获取当前脚本的绝对路径，再向上跳转一级（回到项目根目录）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import numpy as np
import time

from model.vit_config import VITConfig
from model.vit_model import (Conv2dPatchEmbedding, AddCLSTokenAndMask, AddPositionEncoding,
                       TransformerEncoderBlock, TransformerEncoder, VitModel)

config = VITConfig()

if __name__ == "__main__":
    print("===== Conv2dPatchEmbedding test =====")
    # 模拟输入图像
    dummy_image = np.random.rand(1, 224, 224, 3).astype(np.float32)
    # 模块初始化
    patch_embed = Conv2dPatchEmbedding(config)
    # 前向传播
    start = time.time()
    output = patch_embed.forward(dummy_image)  # [1, 256, 768]
    end = time.time()
    print("输入图像 shape:", dummy_image.shape)
    print("输出 embedding shape:", output.shape)
    print("耗时: {:.4f} 秒".format(end - start))
    print(" ")


    print("===== AddCLSTokenAndMask test =====")
    # 模拟输入
    B = 1
    N = config.num_patches  # 256
    D = config.hidden_size  # 768
    patch_embeddings = np.random.randn(B, N, D).astype(np.float32)
    # 模拟掩码布尔数组（虽然我们不使用）
    bool_masked_pos = np.random.choice([False, True], size=(B, N))
    # 模块初始化（不使用mask）
    cls_mask_module = AddCLSTokenAndMask(config=config, use_mask=False)
    # 前向传播
    start = time.time()
    output = cls_mask_module.forward(patch_embeddings, bool_masked_pos)
    end = time.time()
    print("输入patch shape:", patch_embeddings.shape)  # [1, 256, 768]
    print("输出 shape (含CLS):", output.shape)  # [1, 257, 768]
    print("耗时: {:.4f} 秒".format(end - start))
    print(" ")


    print("===== AddPositionEncoding test =====")
    # 模拟输入
    batch_size = 1
    hidden_size = config.hidden_size
    num_patches = config.num_patches
    patch_embeddings_with_CLS = np.random.randn(batch_size, num_patches + 1, hidden_size).astype(np.float32) # 两个模块调换顺序后，patch_embeddings_with_CLS的数量为N+1(CLS token)
    # 模块初始化
    position_encoding = AddPositionEncoding(config)
    # 前向传播
    start = time.time()
    encoded = position_encoding.forward(patch_embeddings_with_CLS)
    end = time.time()
    print("输入 patch 形状:", patch_embeddings_with_CLS.shape)   # [1, 257, 768]
    print("输出位置编码后形状:", encoded.shape)       # [1, 257, 768]
    print("耗时: {:.4f} 秒".format(end - start))
    print(" ")


    print("===== TransformerEncoderBlock test =====")
    # 模拟输入
    B = 1
    N = config.num_patches + 1  # CLS + patches = 257
    D = config.hidden_size
    x = np.random.randn(B, N, D).astype(np.float32)
    # 模块初始化
    block = TransformerEncoderBlock(config)
    # 前向传播
    start = time.time()
    out = block.forward(x)
    end = time.time()
    print("输入 shape:", x.shape)    # [1, 257, 768]
    print("输出 shape:", out.shape)  # [1, 257, 768]
    print("耗时: {:.4f} 秒".format(end - start))
    print(" ")


    print("===== TransformerEncoder test =====")
    # 模拟输入
    B = 1
    N = config.num_patches + 1
    D = config.hidden_size
    x = np.random.randn(B, N, D).astype(np.float32)
    # 模块初始化
    encoder = TransformerEncoder(config)
    # 前向传播
    start = time.time()
    out = encoder.forward(x)
    end = time.time()
    print("输入形状:", x.shape)  # [1, 257, 768]
    print("输出形状:", out.shape)  # [1, 257, 768]
    print("耗时: {:.4f} 秒".format(end - start))
    print(" ")


    print("===== VitModel test =====")
    # 模拟输入图像
    dummy_image = np.random.rand(3, 224, 224, 3).astype(np.float32)
    # 模块初始化
    model = VitModel(config)
    # 前向传播
    start = time.time()
    out = model.forward(dummy_image)
    end = time.time()
    print("输入形状:", dummy_image.shape)  # [3, 224, 224, 3]
    print("输出形状:", out.shape)  # [3, 257, 768]
    print("耗时: {:.4f} 秒".format(end - start))
    print(" ")
