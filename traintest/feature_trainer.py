import sys
import os

# 获取当前脚本的绝对路径，再向上跳转一级（回到项目根目录）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import numpy as np
import time
import matplotlib.pyplot as plt

from model.vit_config import VITConfig
from model.mlp_head import MLPHead


config = VITConfig()


# === FeatureTrainer（使用提取的CLS特征训练MLP Head） ===
class FeatureTrainer:
    def __init__(self, head, X_train, y_train, X_test, y_test, config):
        """
        只训练MLP Head，不依赖ViT主干模块
        :param head: MLPHead 实例
        :param X_train: 提取后的训练特征 [N, 768]
        :param y_train: 训练标签 [N]
        :param X_test: 提取后的测试特征 [N, 768]
        :param y_test: 测试标签 [N]
        :param config: 配置对象（学习率、batch size 等）
        """
        self.head = head
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.lr = config.learning_rate
        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs

        self.weight_decay = config.weight_decay

        # 保存权重引用
        self.params = {
            'W1': head.W1,
            'b1': head.b1,
            'W2': head.W2,
            'b2': head.b2
        }

        # 初始化梯度缓存
        self.grads = {
            'W1': np.zeros_like(head.W1),
            'b1': np.zeros_like(head.b1),
            'W2': np.zeros_like(head.W2),
            'b2': np.zeros_like(head.b2)
        }

        # 记录
        self.train_loss_history = []
        self.train_acc_history = []
        self.test_acc_history = []

    def _forward_pass(self, X_batch, y_batch):
        """
        前向传播：MLP Head 计算 logits、loss、acc
        """
        logits = self.head.forward(X_batch, training=True)

        # Softmax
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)

        # Cross entropy loss
        B = y_batch.shape[0]
        correct_logprobs = -np.log(probs[np.arange(B), y_batch])
        loss = np.mean(correct_logprobs)

        preds = np.argmax(probs, axis=1)
        acc = np.mean(preds == y_batch)

        return logits, loss, acc, probs

    def _backward_pass(self, X_batch, probs, y_batch):
        """
        反向传播更新 MLP Head 的梯度
        :param X_batch: CLS特征 [B, 768]
        :param probs: Softmax 概率 [B, 2]
        """
        B = y_batch.shape[0]
        dlogits = probs.copy()
        dlogits[np.arange(B), y_batch] -= 1
        dlogits /= B

        # W2, b2
        h1 = self.head.hidden
        self.grads['W2'] = h1.T @ dlogits
        self.grads['b2'] = np.sum(dlogits, axis=0, keepdims=True)

        # # GELU 反传
        # dh1 = dlogits @ self.head.W2.T
        # x1 = self.head.hidden_input
        # gelu_grad = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x1 + 0.044715 * x1**3)))
        # dh1 *= gelu_grad
        #
        # if self.head.dropout_rate > 0:
        #     dh1 *= self.head.dropout_mask / (1.0 - self.head.dropout_rate)

        # GELU 反传（tanh 近似的完整导数）
        dh1 = dlogits @ self.head.W2.T
        x1 = self.head.hidden_input

        c = np.float32(np.sqrt(2.0 / np.pi))
        t = c * (x1 + 0.044715 * (x1 ** 3))
        tanh_t = np.tanh(t)
        sech2_t = 1.0 - tanh_t ** 2  # = sech(t)^2
        dt_dx = c * (1.0 + 3.0 * 0.044715 * (x1 ** 2))

        gelu_grad = 0.5 * (1.0 + tanh_t) + 0.5 * x1 * sech2_t * dt_dx
        dh1 *= gelu_grad

        # Dropout 反传（与前向的缩放一致）
        if self.head.dropout_rate > 0:
            dh1 *= self.head.dropout_mask / (1.0 - self.head.dropout_rate)

        # W1, b1
        self.grads['W1'] = X_batch.T @ dh1
        self.grads['b1'] = np.sum(dh1, axis=0, keepdims=True)

    def _update_parameters(self):
        # """
        # SGD 更新参数
        # """
        # for name in self.params:
        #     self.params[name] -= self.lr * self.grads[name]

        wd = self.weight_decay # e.g. 1e-4
        # 对权重做 L2（不对偏置）
        for name in ('W1', 'W2'):
            self.params[name] -= self.lr * (self.grads[name] + wd * self.params[name])
        for name in ('b1', 'b2'):
            self.params[name] -= self.lr * self.grads[name]

    def train(self):
        """
        主训练循环
        """
        num_samples = self.X_train.shape[0]
        num_batches = int(np.ceil(num_samples / self.batch_size))

        print(f"开始训练 MLP Head，共 {self.num_epochs} 个 epoch，每个 epoch 有 {num_batches} 个 batch")

        for epoch in range(1, self.num_epochs + 1):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_train_shuffled = self.X_train[indices]
            y_train_shuffled = self.y_train[indices]

            epoch_loss, epoch_acc = 0.0, 0.0

            for i in range(num_batches):
                start = i * self.batch_size
                end = min(start + self.batch_size, num_samples)
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]

                logits, loss, acc, probs = self._forward_pass(X_batch, y_batch)
                self._backward_pass(X_batch, probs, y_batch)
                self._update_parameters()

                epoch_loss += loss
                epoch_acc += acc

            avg_loss = epoch_loss / num_batches
            avg_acc = epoch_acc / num_batches
            self.train_loss_history.append(avg_loss)
            self.train_acc_history.append(avg_acc)

            test_acc = self.evaluate()
            self.test_acc_history.append(test_acc)

            print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}, Test Acc: {test_acc:.4f}")

    def evaluate(self):
        logits = self.head.forward(self.X_test, training=False)  # ← 关闭dropout
        preds = np.argmax(logits, axis=1)
        acc = np.mean(preds == self.y_test)
        return acc

    def plot_history(self):
        """
        训练可视化
        """
        epochs = range(1, self.num_epochs + 1)
        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_loss_history, label='Train Loss', color='red')
        plt.plot(epochs, self.train_acc_history, label='Train Acc', color='blue')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_acc_history, label='Train Acc', color='blue')
        plt.plot(epochs, self.test_acc_history, label='Test Acc', color='green')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curve")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def save_mlp_weights(self, save_path):
        """保存训练后的MLP Head权重到指定路径"""
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        np.savez(save_path,
                 W1=self.head.W1,
                 b1=self.head.b1,
                 W2=self.head.W2,
                 b2=self.head.b2)
        print(f"💾 MLP Head权重已保存至: {save_path}")


# # FeatureTrainer特征训练
# if __name__ == "__main__":
#     print("===== FeatureTrainer特征训练 =====")

#     # 选一种特征：'gap' 或 'concat'（也可先跑 gap，看曲线后再换 concat）
#     FEAT_KIND = "cls"  # "gap" 或 "concat" 或 "cls"
#     # FEAT_KIND = "gap"  # "gap" 或 "concat" 或 "cls"

#     use_zscore = True  # 建议 True

#     base = "features"
#     Xtr = np.load(f"{base}/{FEAT_KIND}_train_features{'_z' if use_zscore else ''}.npy")
#     Xte = np.load(f"{base}/{FEAT_KIND}_test_features{'_z' if use_zscore else ''}.npy")
#     ytr = np.load(f"{base}/y_train.npy")
#     yte = np.load(f"{base}/y_test.npy")

#     print(f"训练特征形状: {Xtr.shape} | 测试特征形状: {Xte.shape} | 种类: {FEAT_KIND}{'_z' if use_zscore else ''}")

#     # 配置
#     # config = ViTConfig()
#     head = MLPHead(config)

#     start = time.time()
#     trainer = FeatureTrainer(head, Xtr, ytr, Xte, yte, config)
#     trainer.train()
#     end = time.time()
#     print("耗时: {:.4f} 秒".format(end - start))

#     trainer.plot_history()


# FeatureTrainer特征训练
if __name__ == "__main__":
    print("===== FeatureTrainer特征训练 =====")

    base = "feature_test_50samples"
    Xtr = np.load(f"{base}/my_train_features.npy")
    Xte = np.load(f"{base}/my_test_features.npy")
    ytr = np.load(f"{base}/my_train_labels.npy")
    yte = np.load(f"{base}/my_test_labels.npy")

    print(f"训练特征形状: {Xtr.shape} | 测试特征形状: {Xte.shape}")

    head = MLPHead(config)

    start = time.time()
    trainer = FeatureTrainer(head, Xtr, ytr, Xte, yte, config)
    trainer.train()
    end = time.time()
    print("耗时: {:.4f} 秒".format(end - start))

    weights_path = os.path.join(project_root, "extract_weights", "mlp_head_trained_weights.npz")
    trainer.save_mlp_weights(weights_path)

    trainer.plot_history()
