import numpy as np

# === MLP Head 模块 ===
"""
自定义MLP Head：
[CLS向量] → Linear(hidden_size → hidden_size) → GELU → Dropout → Linear(hidden_size → 2)
"""
class MLPHead:
    def __init__(self, config):
        """
        自定义分类器 Head：只训练这部分
        :param config: ViTConfig 对象，提供所有超参数
        """
        self.hidden_size = config.hidden_size
        self.num_classes = config.num_classes
        self.dropout_rate = config.dropout_rate

        # 初始化权重
        self.W1 = np.random.randn(self.hidden_size, self.hidden_size).astype(np.float32) / np.sqrt(self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size), dtype=np.float32)
        self.W2 = np.random.randn(self.hidden_size, self.num_classes).astype(np.float32) / np.sqrt(self.hidden_size)
        self.b2 = np.zeros((1, self.num_classes), dtype=np.float32)

        # 训练时需要保存中间变量以用于反向传播
        self.hidden_input = None   # GELU前的 x1
        self.hidden = None         # Dropout后（或前）的 x1
        self.dropout_mask = None   # Dropout mask（如果使用）

    def gelu(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    def dropout(self, x, training: bool):
        # Dropout mask：保留的地方是1，否则是0（按比例缩放）
        if (not training) or self.dropout_rate == 0:
            self.dropout_mask = np.ones_like(x, dtype=np.float32)
            return x
        self.dropout_mask = (np.random.rand(*x.shape) > self.dropout_rate).astype(np.float32)
        return x * self.dropout_mask / (1.0 - self.dropout_rate)

    def forward(self, cls_token, training: bool = True):
        """
        前向传播
        :param cls_token: [B, D]，每个样本的 CLS token
        :return: logits [B, num_classes]
        """
        # Linear1: [B, D] → [B, D]
        x1 = cls_token @ self.W1 + self.b1

        # 记录 GELU 前的输入（反向时要用）
        self.hidden_input = x1.copy()

        # GELU激活
        x1 = self.gelu(x1)

        # 记录 GELU 后的中间值（反向用）
        self.hidden = x1.copy()

        # Dropout（训练阶段有效）
        x1 = self.dropout(x1, training=training)

        # Linear2: [B, D] → [B, num_classes]
        logits = x1 @ self.W2 + self.b2
        return logits