# ViT Cat-Dog Classification

实现ViT网络模型，迁移dinov2-base预训练权重，冻结backbone，训练MLP Head分类头，实现猫狗图片二分类任务

## process
工作流：data_process -> 
        

## 项目结构
ViT-CatDogClassification
├── data                    数据集和数据预处理
├── evaluate                定义损失函数和评估指标
├── extract_weights         迁移dinov2-base预训练权重
├── features                经过ViT网络提取出的特征
├── inference               推理脚本提取图片特征            
├── model                   模型实现
├── model_weights           给模型加载预训练权重 
├── test_utils              测试工具
└── traintest               训练测试


## data
data
├── data_process.py     数据预处理脚本
├── data_to_npy         存储预处理数据
│   ├── X_test.npy      猫狗图片训练数据合并
│   ├── X_train.npy     猫狗图片测试数据合并
│   ├── y_test.npy      训练数据标签
│   └── y_train.npy     测试数据标签
└── dataset             原始数据
    ├── test            训练数据
    └── train           测试数据


## evaluate
evaluate
├── __init__.py
└── eval.py             定义损失函数和评估指标


## extract_weights
extract_weights
├── dinov2-base/                                预训练模型
├── extract_pretrain_weights-dinov2-base.py     迁移预训练权重脚本
└── weights_vit_base_224/                       存储迁移的权重


## features



## inference
inference/
└── feature_extractor.py    推理脚本


## model
model/
├── mlp_head.py     分类头
├── vit_config.py   模型配置
└── vit_model.py    ViT模型


## model_weights
model_weights/
└── load_weight.py  加载预训练权重


## test_utils
test_utils/                  测试工具
├── test_evaluate.py        
├── test_load_weights.py    
├── test_mlp_head.py
└── test_vit.py


## traintest
traintest
└── feature_trainer.py      训练测试