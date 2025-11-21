import os
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Dinov2Model

"""
Dinov2-Base 预训练权重导出（统一 & 更新版）
- 一次性导出：Patch Embedding、CLS token、Position Embedding(插值到16x16)、
  每层Encoder的 Q/K/V、O(out proj)、FC1/FC2、LayerNorm1/2、Encoder尾部LayerNorm，还有残差缩放LayerScale
- 保持与你 NumPy 手搓ViT（右乘）一致的权重方向：所有线性层权重均已转置后保存
- 调用方式（默认从本地 ./dinov2-base 加载，保存到 ./weights_vit_base_224）：
    python PretrainWeights-dinov2-base.py
  或者（从HuggingFace在线加载）：
    python PretrainWeights-dinov2-base.py --model-id facebook/dinov2-bas
"""

# === 工具函数 ===
"""
Python 的类型注解（type hints）语法：
t: torch.Tensor：给参数 t 做类型标注，表示这个参数期望是 torch.Tensor。
-> np.ndarray：给返回值做类型标注，表示函数应当返回 numpy.ndarray。
"""
def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float32) # detach()：切断计算图，防止 Autograd 跟踪到这份数据（不需要梯度）


def _save_np(path, arr, msg=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)
    if msg is not None:
        print(msg, "shape =", tuple(arr.shape))


def _get_attr(obj, names): # obj是layer层对象
    """鲁棒获取属性：按 names 顺序尝试，直到有一个存在"""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


def interpolate_pos_embed_to_grid(pos_tokens_patch_np: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    将 [1, N, C] 的 patch 部分位置编码重排到 [1, C, H, W]，用双三次插值到 (target_h, target_w)，
    再还原为 [target_h, target_w, C] 的二维网格（便于 NumPy 版直接使用）。
    """

    assert pos_tokens_patch_np.ndim == 3 and pos_tokens_patch_np.shape[0] == 1
    B, N, C = pos_tokens_patch_np.shape  # [1, 1369, 768] for 37x37
    src_hw = int(round(math.sqrt(N)))
    assert src_hw * src_hw == N, f"位置编码patch数不是完美平方：{N}"

    # [1, N, C] -> [1, H, W, C] -> [1, C, H, W]
    pos_hw = pos_tokens_patch_np.reshape(1, src_hw, src_hw, C)
    pos_chw = np.transpose(pos_hw, (0, 3, 1, 2))  # [1, C, H, W]
    pos_chw_t = torch.from_numpy(pos_chw)

    # 双三次插值
    with torch.no_grad():
        pos_resized = F.interpolate(pos_chw_t, size=(target_h, target_w), mode="bicubic", align_corners=False) # 按空间维 (H, W) 双三次插值到 (target_h, target_w)。
        # 稍作裁剪以防极端数值
        pos_resized = pos_resized.clamp_(min=pos_chw_t.min().item(), max=pos_chw_t.max().item()) # 双三次插值可能“过冲”（overshoot），这里把数值裁回到原 min/max 范围，保持分布稳定

    # [1, C, H, W] -> [1, H, W, C] -> [H, W, C]
    pos_hwC = pos_resized.permute(0, 2, 3, 1).contiguous().numpy()
    return pos_hwC[0].astype(np.float32)


def _find_final_layernorm(_model):
    """在多条兼容路径中查找最终 LayerNorm，返回(名字, 模块)"""
    candidates = [
        ("model.encoder.layernorm", getattr(_model.encoder, "layernorm", None)),
        ("model.encoder.layer_norm", getattr(_model.encoder, "layer_norm", None)),
        ("model.encoder.norm", getattr(_model.encoder, "norm", None)),
        ("model.layernorm", getattr(_model, "layernorm", None)),
        ("model.layer_norm", getattr(_model, "layer_norm", None)),
        ("model.norm", getattr(_model, "norm", None)),
        ("model.post_layernorm", getattr(_model, "post_layernorm", None)),
        ("model.post_layer_norm", getattr(_model, "post_layer_norm", None)),
    ]
    for name, mod in candidates:
        if mod is not None and hasattr(mod, "weight") and hasattr(mod, "bias"):
            return name, mod
    return None, None



# === 提取dinov2-base预训练权重 ===
def export_dinov2_base(
    model_id_or_path: str,
    save_dir: str = "./weights_vit_base_224",
    image_size: int = 224,
    patch_size: int = 14,
):
    # === Step 0: 基础配置 & 目录 ===
    print(" ")
    print("===== config =====")
    os.makedirs(save_dir, exist_ok=True)
    assert image_size % patch_size == 0, "image_size 必须能被 patch_size 整除"
    grid_h = grid_w = image_size // patch_size  # 224/14=16
    print(f"【config】image_size={image_size}, patch_size={patch_size}, target_grid={grid_h}×{grid_w}")

    # 获取脚本所在目录（确保无论在哪运行都能找到dinov2-base）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 拼接模型路径：脚本目录 + 传入的模型名/路径
    model_path = os.path.join(script_dir, model_id_or_path)

    # === Step 1: 加载预训练模型（默认 ./dinov2-base；也可传入 facebook/dinov2-base） ===
    print(" ")
    print("===== load Dinov2Model =====")
    model = Dinov2Model.from_pretrained(model_path, local_files_only=os.path.isdir(model_path))
    model.eval()
    print("✅ 模型加载完成！hidden_size =", model.config.hidden_size, ", num_hidden_layers =", model.config.num_hidden_layers)

    # === Step 2: 提取 Patch Embedding（Conv2d 14×14，stride=14）===
    print(" ")
    print("===== Patch Embedding =====")
    patch_proj = model.embeddings.patch_embeddings.projection  # nn.Conv2d
    patch_w = _to_numpy(patch_proj.weight)  # [768, 3, 14, 14]
    _save_np(os.path.join(save_dir, "patch_embed_W.npy"), patch_w, "✅ patch_embed_W.npy 保存为 Conv2d 格式，")
    # print("【debug】patch_embed 权重范数:", float(np.linalg.norm(patch_w)))

    # 保存bias
    if patch_proj.bias is not None:
        patch_b = _to_numpy(patch_proj.bias)  # [768]
        _save_np(os.path.join(save_dir, "patch_embed_b.npy"), patch_b, "✅ patch_embed_b.npy 保存，")

    # === Step 3: 提取 Position Embedding（从 37×37 插值到 16×16）+ CLS Token ===
    print(" ")
    print("===== position_embeddings =====")
    # 3.1 位置编码（含CLS）
    # 通常形状为 [1, 1+N, 768]，其中 N=预训练patch数(如 37×37=1369)
    pos_full = _to_numpy(model.embeddings.position_embeddings)  # [1, 1+N, C]
    print("position_embeddings 原始 shape:", tuple(pos_full.shape))

    # 3.2 分离 CLS / patch
    cls_pos = pos_full[:, :1, :]            # [1,1,C]
    patch_pos = pos_full[:, 1:, :]          # [1,N,C]
    print("patch_pos token 数 =", patch_pos.shape[1], "（应为平方数，如37×37=1369）")

    # 3.3 插值到目标网格（16×16）
    pos_2d = interpolate_pos_embed_to_grid(patch_pos, grid_h, grid_w)  # [16,16,768]
    _save_np(os.path.join(save_dir, "position_embed.npy"), pos_2d, "✅ 已导出 Position Embedding(二维网格)")

    # 3.4 单独导出 CLS token（注意：这是 CLS 向量，不是 patch 的位置编码 CLS 位）
    cls_token = _to_numpy(model.embeddings.cls_token)  # [1,1,768]
    _save_np(os.path.join(save_dir, "cls_token.npy"), cls_token, "✅ 已导出 CLS Token，")

    # 3.5 导出 “CLS 的位置编码” —— 对齐官方做法
    cls_pos_embed = cls_pos.astype(np.float32)  # [1,1,768]
    _save_np(os.path.join(save_dir, "position_embed_cls.npy"), cls_pos_embed, "✅ 已导出 CLS Pos Embedding，")

    # === Step 4: 提取 Encoder 每层参数（全部转置为“右乘”方向）===
    print(" ")
    print("===== Encoder =====")
    """
    说明：你的 NumPy 实现中统一使用 X @ W，所以这里将所有 nn.Linear 的权重 .t() 再保存：
    Q/K/V: [out,in] → [in,out]，堆叠成 [3,768,768]
    out_proj(O): [out,in] → [in,out] = [768,768]
    fc1: [3072,768] → [768,3072]
    fc2: [768,3072] → [3072,768]
    """
    # 遍历 ViT 的第 i 个 Encoder Block。理论上每层包含：LN → Self-Attention → 残差 → LN → MLP → 残差（DINOv2 常见为 Pre-LN，并带残差缩放 LayerScale）。
    for i, layer in enumerate(model.encoder.layer):
        # --- Self-Attention Projections ---
        attn = layer.attention.attention  # 内部包含 query/key/value（nn.Linear）
        Wq = _to_numpy(attn.query.weight.t())
        Wk = _to_numpy(attn.key.weight.t())
        Wv = _to_numpy(attn.value.weight.t())
        qkv = np.stack([Wq, Wk, Wv], axis=0)  # [3,768,768]
        _save_np(os.path.join(save_dir, f"encoder_layer_{i}_qkv.npy"), qkv,
                 f"✅ L{i:02d} QKV 已保存，")

        # out projection
        out_dense = layer.attention.output.dense
        Wo = _to_numpy(out_dense.weight.t())  # [768,768] 右乘
        _save_np(os.path.join(save_dir, f"encoder_layer_{i}_output.npy"), Wo,
                 f"✅ L{i:02d} Attention Output(投影) 已保存，")

        # --- LayerNorm1 / LayerNorm2 ---
        ln1 = _get_attr(layer, ["layernorm_before", "ln1", "norm1", "pre_layernorm", "layer_norm1"])
        ln2 = _get_attr(layer, ["layernorm_after", "ln2", "norm2", "post_layernorm", "layer_norm2"])
        assert ln1 is not None and ln2 is not None, "未找到 LayerNorm1/2，请检查 transformers 版本"
        ln1_np = np.stack([_to_numpy(ln1.weight), _to_numpy(ln1.bias)], axis=0)  # [2,768]
        ln2_np = np.stack([_to_numpy(ln2.weight), _to_numpy(ln2.bias)], axis=0)  # [2,768]
        _save_np(os.path.join(save_dir, f"encoder_layer_{i}_ln1.npy"), ln1_np,
                 f"✅ L{i:02d} LayerNorm1 已保存，")
        _save_np(os.path.join(save_dir, f"encoder_layer_{i}_ln2.npy"), ln2_np,
                 f"✅ L{i:02d} LayerNorm2 已保存，")

        # --- MLP: fc1 / fc2 ---
        fc1 = layer.mlp.fc1
        fc2 = layer.mlp.fc2
        W1 = _to_numpy(fc1.weight.t())  # [768,3072]
        W2 = _to_numpy(fc2.weight.t())  # [3072,768]
        _save_np(os.path.join(save_dir, f"encoder_layer_{i}_fc1.npy"), W1,
                 f"✅ L{i:02d} MLP fc1 已保存，")
        _save_np(os.path.join(save_dir, f"encoder_layer_{i}_fc2.npy"), W2,
                 f"✅ L{i:02d} MLP fc2 已保存，")

        # --- Q/K/V bias ---
        attn = layer.attention.attention  # 或你解析出来的 attn_core
        q = _get_attr(attn, ["query"])
        k = _get_attr(attn, ["key"])
        v = _get_attr(attn, ["value"])
        qkv_linear = _get_attr(attn, ["qkv"])
        # 有的实现 Q/K/V 是三层 Linear，各自有 bias:[768]；有的实现是 融合的 qkv 一层，bias:[3*768]，需要切成三段。
        if q is not None and k is not None and v is not None:
            bqkv = np.stack([_to_numpy(q.bias), _to_numpy(k.bias), _to_numpy(v.bias)], axis=0)  # [3,768]
            _save_np(os.path.join(save_dir, f"encoder_layer_{i}_qkv_bias.npy"), bqkv, f"✅ L{i:02d} QKV bias 已保存，") # 实际模型qkv bias
        elif qkv_linear is not None and getattr(qkv_linear, "bias", None) is not None:
            b_qkv = _to_numpy(qkv_linear.bias)  # [3*768]
            D = b_qkv.shape[0] // 3
            bqkv = np.stack([b_qkv[:D], b_qkv[D:2 * D], b_qkv[2 * D:]], axis=0)
            _save_np(os.path.join(save_dir, f"encoder_layer_{i}_qkv_bias.npy"), bqkv,
                     f"✅ L{i:02d} QKV bias 已保存（fused），")

        # --- out proj bias ---
        out_dense = _get_attr(layer.attention, ["output"])
        saved_out_bias = False
        if out_dense is not None and hasattr(out_dense, "dense") and getattr(out_dense.dense, "bias", None) is not None:
            _save_np(os.path.join(save_dir, f"encoder_layer_{i}_output_bias.npy"), _to_numpy(out_dense.dense.bias),
                     f"✅ L{i:02d} out bias 已保存，")
            saved_out_bias = True
        if not saved_out_bias:
            out_proj = _get_attr(attn, ["out_proj"])
            if out_proj is not None and getattr(out_proj, "bias", None) is not None:
                _save_np(os.path.join(save_dir, f"encoder_layer_{i}_output_bias.npy"), _to_numpy(out_proj.bias),
                         f"✅ L{i:02d} out bias 已保存，")

        # --- MLP bias ---
        if getattr(layer.mlp.fc1, "bias", None) is not None:
            _save_np(os.path.join(save_dir, f"encoder_layer_{i}_fc1_bias.npy"), _to_numpy(layer.mlp.fc1.bias),
                     f"✅ L{i:02d} fc1 bias 已保存，")
        if getattr(layer.mlp.fc2, "bias", None) is not None:
            _save_np(os.path.join(save_dir, f"encoder_layer_{i}_fc2_bias.npy"), _to_numpy(layer.mlp.fc2.bias),
                     f"✅ L{i:02d} fc2 bias 已保存，")

        # --- LayerScale（新增） ---
        # 兼容命名：layer_scale1/layer_scale2，参数名通常叫 lambda1
        ls1_mod = _get_attr(layer, ["layer_scale1", "ls1", "gamma1"])
        ls2_mod = _get_attr(layer, ["layer_scale2", "ls2", "gamma2"])

        def _export_ls(mod, name):
            if mod is None:
                print(f"【warn】未找到 {name}（本层无 LayerScale？）")
                return
            # 常见结构：有一个 Parameter 叫 lambda1
            lam = getattr(mod, "lambda1", None)
            if lam is None:
                # 有些实现可能直接是 weight，做个兜底
                lam = getattr(mod, "weight", None)
            if lam is None:
                print(f"【warn】{name} 未找到 lambda1/weight")
                return
            arr = _to_numpy(lam).reshape(-1).astype(np.float32)  # [768]
            _save_np(os.path.join(save_dir, f"encoder_layer_{i}_{name}.npy"), arr,
                     f"✅ L{i:02d} {name} 已保存，")

        _export_ls(ls1_mod, "ls1")
        _export_ls(ls2_mod, "ls2")

    # === Step 5: Encoder 尾部 LayerNorm（更鲁棒的多路径查找） ===
    print(" ")
    print("===== Encoder 尾部 LayerNorm =====")
    ln_name, enc_norm_mod = _find_final_layernorm(model)
    assert enc_norm_mod is not None, "未找到 Encoder 尾部 LayerNorm（已尝试多条兼容路径）"
    print(f"最终LayerNorm模块 = {ln_name}")

    enc_ln = np.stack([_to_numpy(enc_norm_mod.weight), _to_numpy(enc_norm_mod.bias)], axis=0)  # [2,768]
    _save_np(os.path.join(save_dir, "encoder_ln_final.npy"), enc_ln, "✅ Encoder 尾部 LayerNorm 已保存，")

    print("\n🎉 全部权重导出完成！保存目录：", save_dir)


def main():
    """
    把脚本做成一个可配置的命令行工具（默认值即是合理配置），你可以在命令行覆盖其中任意参数，最终把对应模型的权重按你前面定义的格式导出为 .npy。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="dinov2-base",
                        help="预训练模型目录或HuggingFace的模型名，如 ./dinov2-base 或 facebook/dinov2-base")
    parser.add_argument("--save-dir", type=str, default="./weights_vit_base_224",
                        help="导出的 .npy 权重保存目录")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=14)
    args = parser.parse_args()

    export_dinov2_base(
        model_id_or_path=args.model_id,
        save_dir=args.save_dir,
        image_size=args.image_size,
        patch_size=args.patch_size,
    )


if __name__ == "__main__":
    # 小型 debug：打印 torch / transformers 版本，确认环境
    print("torch =", torch.__version__)
    try:
        import transformers
        print("transformers =", transformers.__version__)
    except Exception:
        pass
    main()
