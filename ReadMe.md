# 🐶🐱 ViT Cat-Dog Classification

NumPy Implementation of ViT + DINOv2-Base Pretrained Weights + Frozen Backbone + MLP Head Classifier

本项目基于 NumPy 手搓 Vision Transformer (ViT-B/14)，迁移 DINOv2-Base 预训练权重，在冻结 backbone 的情况下训练轻量 MLP Head，最终实现 猫狗二分类任务（Test Acc ≈ 99.8%）。

## Environment
```bash
pip install -r requirements.txt
```


## Project Structure
```bash
ViT-CatDogClassification
├── data                # 数据集和数据预处理
├── evaluate            # 定义损失函数和评估指标
├── extract_weights     # 迁移dinov2-base预训练权重
├── features            # 经过ViT网络提取出的特征
├── inference           # 推理脚本提取图片特征
├── model               # 模型实现
├── model_weights       # 给模型加载预训练权重
├── test_utils          # 测试工具
└── traintest           # 训练测试
```


## Usages
以下命令均在项目根目录下执行。

1. 单张图片推理
```bash
python inference/sample_inference.py --image data/sample/xxx.jpg
```

2. 数据预处理
```bash
python data/data_process.py
```

3. 提取特征
```bash
python inference/feature_extractor.py
```

4. 训练和测试
```bash
python traintest/feature_trainer.py
```

5. 使用工具测试模块
```bash
python test_utils/xxx.py
```