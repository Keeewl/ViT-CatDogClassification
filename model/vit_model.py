import numpy as np
from skimage.util import view_as_blocks
from skimage.util import view_as_windows
from skimage.transform import resize

"""
ViT_numpy_CatDogClassify-base-14-new:
1. 初始化numpy手搓的ViT模型
2. 加载预训练权重
3. 冻结backbone，输入预处理数据，走ViT模型的前向，得到特征features
4. 用特征训练MLP Head
5. 得到结果和结果分析
"""



# === Conv2dPatchEmbedding 模块 ===
"""
维度变化：
输入名称                                   维度                        描述
images                                   [B, H, W, C]               归一化后的原图（channels-last，C=3）
blocks/windows                           [B, gh, gw, C, ph, pw]     提取Patch（blocks 或 windows）
patches                                  [B, N, C, ph, pw]          合并网格后得到的 patch 集合
patches_flat                             [B, N, C*ph*pw]            展平成向量（例如 [B, 256, 588]）
weight                                   [768, C, ph, pw]           卷积权重（Conv2d等价表示）
filters                                  [768, C*ph*pw]             展平后的卷积核（例如 [768, 588]）
out = patches_flat @ filters.T           [B, N, 768]                每个 patch 与每个 filter 点乘
out += bias                              [B, N, 768]                bias 形状 [768]
patch_embeddings                         [B, N, D]                  D=768（例如 [B, 256, 768]），供后续位置编码/CLS拼接
"""
class Conv2dPatchEmbedding:
    def __init__(self, config):
        """
        用 Conv2d 实现的 Patch Embedding（适配 DINOv2）
        相当于一个大步长的卷积：Conv2d(in=3, out=768, kernel_size=14, stride=14)
        :param config: ViTConfig 配置对象
        """
        self.patch_size = config.patch_size     # patch大小 = kernel大小
        self.stride = config.patch_size         # DINOv2 的 stride = kernel_size
        self.out_channels = config.hidden_size  # 输出维度 = 768
        self.in_channels = config.in_channels   # 输入通道 = 3

        # weight shape = [out_channels, in_channels, patch_size, patch_size]
        self.weight = np.random.randn(
            self.out_channels,
            self.in_channels,
            self.patch_size,
            self.patch_size
        ).astype(np.float32) * (1 / np.sqrt(self.patch_size * self.patch_size * self.in_channels))

        self.bias = np.zeros((self.out_channels,), dtype=np.float32)

    def extract_patches(self, images):
        """
        :param images: numpy数组 [B, H, W, C] (C=3)
        :return: patch窗口 [B, gh, gw, C, ph, pw]
        """
        B, H, W, C = images.shape
        ph = pw = self.patch_size

        if self.stride == self.patch_size:  # 非重叠 → blocks 路径 （这个方法更快，省内存）
            blks = view_as_blocks(images, block_shape=(1, ph, pw, C))
            # blks: [B, gh, gw, 1, 1, ph, pw, C]
            gh, gw = H // ph, W // pw
            blks = blks.reshape(B, gh, gw, ph, pw, C).transpose(0, 1, 2, 5, 3, 4)
            # -> [B, gh, gw, C, ph, pw]
            return blks
        else:  # 有重叠 → windows 路径
            patches = view_as_windows(images, (1, ph, pw, C), step=(1, self.stride, self.stride, 1))
            # [B, gh, gw, 1, 1, ph, pw, C]
            gh = patches.shape[1];
            gw = patches.shape[2]
            patches = patches.reshape(B, gh, gw, ph, pw, C).transpose(0, 1, 2, 5, 3, 4)
            # -> [B, gh, gw, C, ph, pw]
            return patches

    def forward(self, images):
        """
        前向传播：模拟Conv2d提取patch embedding
        :param images: [B, H, W, C]
        :return: [B, num_patches, 768]
        """
        B, H, W, C = images.shape
        assert C == self.in_channels, "输入图像通道数不匹配"

        # === 提取滑窗 Patch ===
        patches = self.extract_patches(images)  # [B, gh, gw, C, ph, pw]
        B, gh, gw, C, ph, pw = patches.shape
        num_patches = gh * gw

        # 展平每个 patch 为向量：[B, num_patches, C, ph, pw]
        patches = patches.reshape(B, gh * gw, C, ph, pw)  # [B, 256, 3, 14, 14]

        # 卷积权重：[768, 3, 14, 14] → reshape 为 [768, 588]
        filters = self.weight.reshape(self.out_channels, -1)  # [768, 588]
        patches_flat = patches.reshape(B, num_patches, -1)    # [B, 256, 588]

        # 矩阵乘法模拟卷积：每个 patch 与每个 filter 点乘
        out = np.matmul(patches_flat, filters.T)  # [B, 256, 768]
        out += self.bias  # 加上偏置

        return out  # 输出：patch embedding


# === CLS & Mask 拼接模块 ===
"""
维度变化：
输入名称	            维度	            描述
patch_embeddings	[B, N, D]	    patch嵌入序列（例如 [1, 256, 768]）
bool_masked_pos	    [B, N]	        可选，掩码标记（布尔）
output              [B, N+1, D]	    拼接了CLS的完整序列
"""
class AddCLSTokenAndMask:
    def __init__(self, config, use_mask=False):
        """
        拼接CLS令牌，可选：处理MASK掩码（自监督任务）
        :param config: 配置对象（提供 hidden_size）
        :param use_mask: 是否启用mask机制（训练DINO时使用）
        """
        self.hidden_size = config.hidden_size
        self.use_mask = use_mask

        # 可学习的CLS token，初始化为标准正态分布
        self.cls_token = np.random.randn(1, 1, self.hidden_size).astype(np.float32)  # [1, 1, D]

        # 可学习的mask token（如果使用mask）
        self.mask_token = np.random.randn(self.hidden_size).astype(np.float32)  # [D]

    def forward(self, patch_embeddings, bool_masked_pos=None):
        """
        拼接CLS令牌（和可选的掩码替换）
        :param patch_embeddings: 输入 patch 向量 [B, N, D]
        :param bool_masked_pos: 掩码布尔数组 [B, N]，为True的地方将被mask token替代
        :return: 拼接CLS后的输出 [B, N+1, D]
        """
        B, N, D = patch_embeddings.shape
        assert D == self.hidden_size, "维度不匹配"

        # === Step 1: 掩码处理（如果启用）===
        if self.use_mask and bool_masked_pos is not None:
            masked = patch_embeddings.copy()  # 深拷贝
            mask_token_expanded = np.broadcast_to(self.mask_token, (B, N, D))
            masked = np.where(bool_masked_pos[..., np.newaxis], mask_token_expanded, masked)
        else:
            masked = patch_embeddings  # 不使用mask，直接通过

        # === Step 2: 添加CLS token ===
        cls_tokens = np.broadcast_to(self.cls_token, (B, 1, D))  # [B, 1, D]
        output = np.concatenate([cls_tokens, masked], axis=1)   # [B, N+1, D]
        return output


# === AddPositionEncoding 模块 ===
"""
维度变化:
1. 输入嵌入：
embedded_patches：[batch_size, 1+256, 768]
2. 位置编码逻辑：
原始预训练分辨率：224 × 224，对应 16 × 16 = 256 个 patch
加上（数值相加） [1, 1+256, 768] 的位置编码后仍为 [batch_size, 1+256, 768]
"""
class AddPositionEncoding:
    def __init__(self, config):
        """
        添加可学习的位置编码
        参数:
            config: ViTConfig 配置对象
        """
        self.num_patches = config.num_patches          # 256
        self.hidden_size = config.hidden_size          # 768
        self.grid_h = config.grid_h                    # 16
        self.grid_w = config.grid_w                    # 16

        # 初始化位置编码 (形状: [16,16,768])
        self.pos_embed_2d = np.random.randn(self.grid_h, self.grid_w, self.hidden_size).astype(np.float32)

        """
        对齐官方：在拼 CLS 之后，为完整序列 [B, 1+N, D] 一次性添加位置编码：
        [pos_cls(1,1,D), pos_patches(1,N,D)]
        """
        # 初始化CLS的位置编码(形状: [1,1,D])
        self.cls_pos_embed = np.random.randn(1, 1, self.hidden_size).astype(np.float32)

    def interpolate_pos_encoding(self, target_grid_h, target_grid_w):
        """
        插值位置编码为目标尺寸
        :param target_grid_h: 新网格的高度（如24）
        :param target_grid_w: 新网格的宽度（如24）
        :return: 插值后的二维位置编码 [target_h * target_w, hidden_size]
        """
        # 插值算法
        resized = resize(
            self.pos_embed_2d,  # 原始 [16,16,768]
            (target_grid_h, target_grid_w, self.hidden_size),
            order=3,                # 使用双三次(bicubic)插值算法   计算新位置像素值时考虑周围 4×4 区域的加权平均
            mode='reflect',         # 边界处理方式：镜像反射边界值
            anti_aliasing=False,    # 禁用抗锯齿
            preserve_range=True     # 保持原始数值范围不进行归一化
        ).astype(np.float32)

        return resized.reshape(-1, self.hidden_size)  # 展平成 [target_h * target_w, hidden_size]

    def forward(self, patch_embeddings_with_CLS, image_size=None, patch_size=14):
        """
        添加位置编码
        :param patch_embeddings_with_CLS: [batch, 1 + num_patches, hidden_size]
        :param image_size: tuple (H, W)，用于动态插值
        :param patch_size: patch 的尺寸（默认 14）
        :return: 加上位置编码后的输出 [batch, 1 + num_patches, hidden_size]
        """
        B, N, D = patch_embeddings_with_CLS.shape
        # 当前实际的 patch 数（不含 CLS）
        cur_num_patches = N - 1

        if image_size is not None:
            h, w = image_size
            gh, gw = h // patch_size, w // patch_size
            assert h % patch_size == 0 and w % patch_size == 0
            if (gh, gw) == (self.grid_h, self.grid_w):
                pos_embed = self.pos_embed_2d.reshape(-1, D)  # 不插值
            else:
                pos_embed = self.interpolate_pos_encoding(gh, gw)  # 仅在尺寸变化时插
        else:
            pos_embed = self.pos_embed_2d.reshape(-1, D)  # 默认 [256,768]

        # 检查形状是否匹配
        assert pos_embed.shape[0] + 1 == N, f"位置编码数量不匹配：{pos_embed.shape[0]} vs {N}"
        assert pos_embed.shape[1] == D, f"位置编码维度不匹配：{pos_embed.shape[1]} vs {D}"

        # patch的位置编码   用“当前”patch 数
        pos_embed = pos_embed.reshape(1, cur_num_patches, D)  # [1, 256, 768]
        pos_embed_batch = np.broadcast_to(pos_embed, (B, cur_num_patches, D))

        # 加入CLS的位置编码pos_cls（对齐官方）
        assert self.cls_pos_embed is not None and self.cls_pos_embed.shape == (1, 1, D), \
            f"缺失或维度错误的 cls_pos_embed，得到: {None if self.cls_pos_embed is None else self.cls_pos_embed.shape}"
        pos_cls = np.broadcast_to(self.cls_pos_embed.astype(np.float32), (B, 1, D))  # [B,1,D]

        # 拼接完整 pos 并相加
        pos_full = np.concatenate([pos_cls, pos_embed_batch], axis=1)  # [B,1+cur_num_patches,D]
        out = patch_embeddings_with_CLS + pos_full

        return out


# === Transformer Encoder Block 模块 ===
"""
模块名	                    输入维度	        输出维度	        功能
LayerNorm	                [B, N, D]	    [B, N, D]	    特征归一化
MultiHeadSelfAttention	    [B, N, D]	    [B, N, D]	    全局建模注意力
MLPBlock	                [B, N, D]	    [B, N, D]	    非线性变换
TransformerEncoderBlock	    [B, N, D]	    [B, N, D]	    单层Transformer Encoder
"""

"""
1. LayerNorm里面的 γ（gamma/weight）和 β（beta/bias）是可学习的缩放与偏移
2. LayerScale 有助于稳定深层训练（初值可能很小），作用是把子层输出按通道缩放后再加回主分支。
"""

# 子模块一：LayerNorm
class LayerNorm:
    def __init__(self, config):
        self.hidden_size = config.hidden_size
        self.gamma = np.ones((1, 1, self.hidden_size), dtype=np.float32)
        self.beta = np.zeros((1, 1, self.hidden_size), dtype=np.float32)
        self.ln_eps = config.ln_eps

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        norm = (x - mean) / np.sqrt(var + self.ln_eps)
        return self.gamma * norm + self.beta

    def __call__(self, x):
        return self.forward(x)

# 子模块二：Multi-Head Self Attention
class MultiHeadSelfAttention:
    def __init__(self, config):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        assert self.hidden_size % self.num_heads == 0
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        # 权重（右乘）
        self.W_q = np.random.randn(self.hidden_size, self.hidden_size).astype(np.float32) / np.sqrt(self.hidden_size)
        self.W_k = np.random.randn(self.hidden_size, self.hidden_size).astype(np.float32) / np.sqrt(self.hidden_size)
        self.W_v = np.random.randn(self.hidden_size, self.hidden_size).astype(np.float32) / np.sqrt(self.hidden_size)
        self.W_o = np.random.randn(self.hidden_size, self.hidden_size).astype(np.float32) / np.sqrt(self.hidden_size)

        # === 新增：bias（与 HF 对齐；形状均为 [768]，广播加到最后一维）===
        self.b_q = np.zeros((self.hidden_size,), dtype=np.float32)
        self.b_k = np.zeros((self.hidden_size,), dtype=np.float32)
        self.b_v = np.zeros((self.hidden_size,), dtype=np.float32)
        self.b_o = np.zeros((self.hidden_size,), dtype=np.float32)

    def _split_heads(self, x):
        B, N, D = x.shape
        x = x.reshape(B, N, self.num_heads, self.head_dim)
        return np.transpose(x, (0, 2, 1, 3))  # [B, H, N, D]

    def _combine_heads(self, x):
        B, H, N, D = x.shape
        x = np.transpose(x, (0, 2, 1, 3))  # [B, N, H, D]
        return x.reshape(B, N, H * D)

    def forward(self, x):
        # 线性投影 + bias
        Q = x @ self.W_q + self.b_q
        K = x @ self.W_k + self.b_k
        V = x @ self.W_v + self.b_v

        # 分头
        Q = self._split_heads(Q)   # [B,H,N,D]
        K = self._split_heads(K)
        V = self._split_heads(V)

        # 打分 + 缩放 + softmax（axis=-1）
        scores = (Q @ K.transpose(0, 1, 3, 2)) * self.scale   # [B,H,N,N]
        scores -= np.max(scores, axis=-1, keepdims=True)      # 数值稳定
        weights = np.exp(scores)
        weights /= np.sum(weights, axis=-1, keepdims=True) + 1e-12

        # 上下文 + 合并 + 输出投影
        attention = weights @ V                 # [B,H,N,D]
        out = self._combine_heads(attention)    # [B,N,768]
        out = out @ self.W_o + self.b_o         # 输出投影 + bias
        return out

# 子模块三：MLP Block
class MLPBlock:
    def __init__(self, config):
        self.hidden_size = config.hidden_size
        self.mlp_hidden_size = config.mlp_hidden_size
        self.fc1 = np.random.randn(self.hidden_size, self.mlp_hidden_size).astype(np.float32) / np.sqrt(self.hidden_size)
        self.fc2 = np.random.randn(self.mlp_hidden_size, self.hidden_size).astype(np.float32) / np.sqrt(self.mlp_hidden_size)

        # === 新增：bias（与 HF 对齐）===
        self.b1 = np.zeros((self.mlp_hidden_size,), dtype=np.float32)  # [3072]
        self.b2 = np.zeros((self.hidden_size,), dtype=np.float32)      # [768]

    # 近似版 GELU
    def gelu(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    def forward(self, x):
        x = x @ self.fc1 + self.b1
        x = self.gelu(x)
        x = x @ self.fc2 + self.b2
        return x

# 整合模块：TransformerEncoderBlock
class TransformerEncoderBlock:
    def __init__(self, config):
        self.ln1 = LayerNorm(config)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2 = LayerNorm(config)
        self.mlp = MLPBlock(config)

        # === LayerScale（新增）===
        D = config.hidden_size
        # 占位：加载器会把它们替换成真实的 γ 向量（[1,1,D]）
        self.ls1 = np.ones((1, 1, D), dtype=np.float32)
        self.ls2 = np.ones((1, 1, D), dtype=np.float32)

    def forward(self, x):
        # Attn 分支
        residual = x
        x = self.ln1(x)
        x = self.attn.forward(x)
        x = residual + x * self.ls1   # ← 逐通道 LayerScale γ1

        # MLP 分支
        residual = x
        x = self.ln2(x)
        x = self.mlp.forward(x)
        x = residual + x * self.ls2   # ← 逐通道 LayerScale γ2
        return x


# === TransformerEncoder 模块 ===
"""
模块	                    输入	            输出	            功能
TransformerEncoder	    [B, N, D]	    [B, N, D]	    多层堆叠 Transformer 编码器（带残差+归一化）
"""
class TransformerEncoder:
    def __init__(self, config):
        """
        多层 Transformer Encoder 堆叠模块
        :param config: ViTConfig 对象，提供所有超参数
        """
        self.config = config
        self.num_layers = config.num_layers

        # 构建多个编码器块
        self.layers = [TransformerEncoderBlock(config) for _ in range(self.num_layers)]
        self.norm = LayerNorm(config)

    def forward(self, x):
        """
        前向传播
        :param x: 输入 token 序列 [B, N, D]
        :return: 输出 token 序列 [B, N, D]
        """
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
        x = self.norm.forward(x)
        return x


class VitModel:
    def __init__(self, config):
        """
        构建Vit主干模型
        :param config: ViTConfig 对象，提供所有超参数
        """
        self.patch_embed = Conv2dPatchEmbedding(config)
        self.cls_module = AddCLSTokenAndMask(config)
        self.pos_embed = AddPositionEncoding(config)
        self.encoder = TransformerEncoder(config)

    def forward(self, image, training: bool = True):
        """
        ViT前向传播
        :param image: numpy数组 [B, H, W, C]
        :param training: 是否为训练模式
        :return: 编码器输出 [B, N+1, D]
        """
        x = self.patch_embed.forward(image)                         # [B, 256, 768]
        x = self.cls_module.forward(x)                              # 拼接CLS [B, 257, 768]
        x = self.pos_embed.forward(x, image_size=None)              # +位置编码 [B, 257, 768]
        x = self.encoder.forward(x)                                 # 编码器 [B, 257, 768]
        return x
