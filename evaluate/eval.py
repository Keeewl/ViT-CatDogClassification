import numpy as np

# === 定义损失函数 logsumexp ===
def cross_entropy_loss(logits, labels):
    # logits: [B, 2], labels: [B] (int)
    m = np.max(logits, axis=1, keepdims=True)
    logsumexp = m + np.log(np.sum(np.exp(logits - m), axis=1, keepdims=True))
    log_probs = logits - logsumexp                    # [B, 2] = log_softmax
    nll = -log_probs[np.arange(labels.shape[0]), labels.astype(np.int64)]
    return np.mean(nll)

# === 定义评估指标 ===
def accuracy(logits, labels):
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == labels)