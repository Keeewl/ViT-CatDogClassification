class VITConfig:
    def __init__(self):
        self.image_size = 224            # 输入图像尺寸（224 x 224）
        self.patch_size = 14             # 每个 patch 是 14x14
        self.in_channels = 3             # 输入通道数（RGB）
        self.hidden_size = 768           # patch -> embedding 映射维度
        self.num_heads = 12              # 多头注意力数量
        self.num_layers = 12             # 编码器层数
        self.mlp_hidden_size = 3072      # FFN 隐藏层维度
        self.num_classes = 2             # 猫狗分类任务
        self.use_cls_token = True        # 是否拼接 CLS token
        self.patch_grid = (16, 16)       # 初始化占位，运行时动态计算
        self.num_patches = 256           # 分块的数量
        self.grid_h = 16                 # patch高度的数量
        self.grid_w = 16                 # patch宽度的数量
        self.ln_eps = 1e-6               # 或 1e-5，和HF保持一致就好

        self.learning_rate = 1e-2
        self.batch_size = 32
        self.num_epochs = 50
        self.dropout_rate = 0.5
        self.weight_decay = 1e-2