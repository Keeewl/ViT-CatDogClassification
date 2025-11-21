import sys
import os

# 获取当前脚本的绝对路径，再向上跳转一级（回到项目根目录）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import numpy as np
import os

# === 加载Dinov2预训练权重 ===
"""
权重文件                         模块                                             模型变量名
patch_embed_W.npy               PatchEmbedding                                  self.patch_embed.weight
                                （Conv2d权重，形状[768,3,14,14]）
position_embed.npy              AddPositionEncoding                             self.pos_embed.pos_embed_2d
                                （二维网格位置编码，形状[16,16,768]；动态插值以此为源）
position_embed_cls.npy          AddPositionEncoding                             self.pos_embed.cls_pos_embed
                                （CLS专属位置编码，形状[1,1,768]；先拼CLS后一次性相加）
cls_token.npy                   AddCLSTokenAndMask                              self.cls_module.cls_token
                                （CLS起始向量参数，形状[1,1,768]，与图像无关）
encoder_layer_{i}_qkv.npy       TransformerEncoderBlock                         self.encoder.layers[i].attn.W_q / W_k / W_v
                                （每个为[768,768]；已转为右乘方向：X @ W_*）
encoder_layer_{i}_output.npy    TransformerEncoderBlock                         self.encoder.layers[i].attn.W_o
                                （[768,768]；右乘方向：(...heads...) @ W_o）
encoder_layer_{i}_fc1.npy       MLPBlock                                        self.encoder.layers[i].mlp.fc1
                                （[768,3072]；右乘方向：X @ W1）
encoder_layer_{i}_fc2.npy       MLPBlock                                        self.encoder.layers[i].mlp.fc2
                                （[3072,768]；右乘方向：X @ W2）
encoder_layer_{i}_ln1.npy       LayerNorm1                                      gamma/beta → self.encoder.layers[i].ln1.gamma / ln1.beta
                                （保存格式为[2,768]，加载后各自reshape到[1,1,768]）
encoder_layer_{i}_ln2.npy       LayerNorm2                                      gamma/beta → self.encoder.layers[i].ln2.gamma / ln2.beta
                                （同上）
encoder_ln_final.npy            Encoder尾部LayerNorm                              self.encoder.norm.gamma / self.encoder.norm.beta
                                （保存格式为[2,768]，加载后各自reshape到[1,1,768]）
"""
# weights_vit_base_224文件夹下有dinov2-base预训练权重  <--  PretrainWeights-dinov2-base.py
def load_dinov2_weights(model, config, weight_dir="./weights_vit_base_224"):
    # === Patch Embedding ===
    model.patch_embed.weight = np.load(os.path.join(weight_dir, "patch_embed_W.npy")).astype(np.float32)
    model.patch_embed.bias = np.load(os.path.join(weight_dir, "patch_embed_b.npy")).astype(np.float32)
    # print("【debug】✅ patch_embed.weight shape:", model.patch_embed.weight.shape)  # (768,3,14,14)
    # print("【debug】✅ patch_embed.bias shape:", model.patch_embed.bias.shape)  # (768,)

    # === CLS & Pos ===（保持你原状）
    model.cls_module.cls_token = np.load(os.path.join(weight_dir, "cls_token.npy")).astype(np.float32)
    # print("【debug】✅ CLS Token shape:", model.cls_module.cls_token.shape)  # (1, 1, 768)
    model.pos_embed.pos_embed_2d = np.load(os.path.join(weight_dir, "position_embed.npy")).astype(np.float32)
    # print("【debug】✅ Position Embedding shape:", model.pos_embed.pos_embed_2d.shape) # (16, 16, 768)
    cls_pos_path = os.path.join(weight_dir, "position_embed_cls.npy")
    if os.path.exists(cls_pos_path):
        model.pos_embed.cls_pos_embed = np.load(cls_pos_path).astype(np.float32)
        # print("【debug】✅ CLS Position Embedding shape:", model.pos_embed.cls_pos_embed.shape) # (1, 1, 768)
    else:
        model.pos_embed.cls_pos_embed = None
        # print("【warn】未找到 position_embed_cls.npy")

    # === Transformer Encoder Layers ===
    for i, layer in enumerate(model.encoder.layers):
        # QKV 权重
        qkv = np.load(os.path.join(weight_dir, f"encoder_layer_{i}_qkv.npy")).astype(np.float32)  # [3, 768, 768]
        layer.attn.W_q = qkv[0]
        layer.attn.W_k = qkv[1]
        layer.attn.W_v = qkv[2]

        # Attention Output
        layer.attn.W_o = np.load(os.path.join(weight_dir, f"encoder_layer_{i}_output.npy")).astype(np.float32)

        # LayerNorm1
        ln1 = np.load(os.path.join(weight_dir, f"encoder_layer_{i}_ln1.npy")).astype(np.float32)  # [2, 768]
        layer.ln1.gamma = ln1[0].reshape(1, 1, -1)
        layer.ln1.beta = ln1[1].reshape(1, 1, -1)

        # LayerNorm2
        ln2 = np.load(os.path.join(weight_dir, f"encoder_layer_{i}_ln2.npy")).astype(np.float32)  # [2, 768]
        layer.ln2.gamma = ln2[0].reshape(1, 1, -1)
        layer.ln2.beta = ln2[1].reshape(1, 1, -1)

        # MLP
        layer.mlp.fc1 = np.load(os.path.join(weight_dir, f"encoder_layer_{i}_fc1.npy")).astype(np.float32)  # [768, 3072]
        layer.mlp.fc2 = np.load(os.path.join(weight_dir, f"encoder_layer_{i}_fc2.npy")).astype(np.float32)  # [3072, 768]

        # === 新增：bias 加载 ===
        qkvb_path = os.path.join(weight_dir, f"encoder_layer_{i}_qkv_bias.npy")
        if os.path.exists(qkvb_path):
            bq, bk, bv = np.load(qkvb_path).astype(np.float32)
            layer.attn.b_q = bq
            layer.attn.b_k = bk
            layer.attn.b_v = bv

        ob_path = os.path.join(weight_dir, f"encoder_layer_{i}_output_bias.npy")
        if os.path.exists(ob_path):
            layer.attn.b_o = np.load(ob_path).astype(np.float32)

        fc1b_path = os.path.join(weight_dir, f"encoder_layer_{i}_fc1_bias.npy")
        fc2b_path = os.path.join(weight_dir, f"encoder_layer_{i}_fc2_bias.npy")
        if os.path.exists(fc1b_path):
            layer.mlp.b1 = np.load(fc1b_path).astype(np.float32)
        if os.path.exists(fc2b_path):
            layer.mlp.b2 = np.load(fc2b_path).astype(np.float32)

        # --- LayerScale γ（新增） ---
        D = layer.ln1.gamma.shape[-1]  # 768
        ls1_path = os.path.join(weight_dir, f"encoder_layer_{i}_ls1.npy")
        ls2_path = os.path.join(weight_dir, f"encoder_layer_{i}_ls2.npy")

        # 若层对象还没有属性，先放默认 1（兼容旧权重）
        if not hasattr(layer, "ls1"):
            layer.ls1 = np.ones((1, 1, D), dtype=np.float32)
        if not hasattr(layer, "ls2"):
            layer.ls2 = np.ones((1, 1, D), dtype=np.float32)

        # 读取 ls1
        if os.path.exists(ls1_path):
            gamma1 = np.load(ls1_path).astype(np.float32)         # [768]
            assert gamma1.ndim == 1 and gamma1.shape[0] == D, f"L{i} ls1 形状异常: {gamma1.shape}"
            layer.ls1 = gamma1.reshape(1, 1, D)                   # [1,1,768]
            # print(f"【debug】✅ L{i:02d} ls1 shape: {layer.ls1.shape}  ||ls1||₂={np.linalg.norm(gamma1):.4f}")
        else:
            print(f"【warn】未找到 {ls1_path}，L{i:02d} ls1 使用全 1。")

        # 读取 ls2
        if os.path.exists(ls2_path):
            gamma2 = np.load(ls2_path).astype(np.float32)         # [768]
            assert gamma2.ndim == 1 and gamma2.shape[0] == D, f"L{i} ls2 形状异常: {gamma2.shape}"
            layer.ls2 = gamma2.reshape(1, 1, D)                   # [1,1,768]
            # print(f"【debug】✅ L{i:02d} ls2 shape: {layer.ls2.shape}  ||ls2||₂={np.linalg.norm(gamma2):.4f}")
        else:
            print(f"【warn】未找到 {ls2_path}，L{i:02d} ls2 使用全 1。")

    # === Encoder尾部LayerNorm ===
    ln_final = np.load(os.path.join(weight_dir, "encoder_ln_final.npy")).astype(np.float32)  # [2, 768]
    # print("【debug】✅ Encoder尾部 LayerNorm shape:", ln_final.shape) # (2, 768)
    model.encoder.norm.gamma = ln_final[0].reshape(1, 1, -1)
    model.encoder.norm.beta = ln_final[1].reshape(1, 1, -1)

    g_final = model.encoder.norm.gamma.reshape(-1)  # 或 model.encoder.norm.gamma[0,0,:]
    b_final = model.encoder.norm.beta.reshape(-1)
    # print("【debug】✅ Encoder尾部 LayerNorm gamma shape:", g_final.shape) # (768,)
    # print("【debug】✅ Encoder尾部 LayerNorm beta shape:", b_final.shape) # (768,)

    # === 确认最终LN的γ / β尺度 ===
    # print("【debug】[final LN] ||gamma||₂ =", np.linalg.norm(g_final),
    #       " mean=", g_final.mean(), " min=", g_final.min(), " max=", g_final.max())
    # print("【debug】[final LN] ||beta||₂  =", np.linalg.norm(b_final),
    #       " mean=", b_final.mean(), " min=", b_final.min(), " max=", b_final.max())

    print("✅ 所有预训练主干权重加载完毕！")
