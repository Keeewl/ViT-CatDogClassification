import sys
import os

# 获取当前脚本的绝对路径，再向上跳转一级（回到项目根目录）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

"""
图像数据加载与预处理
目标：将原始图像数据读取、预处理，并保存为 .npy 文件供 NumPy ViT 模型使用

1. 统一成 短边 256 等比缩放 
2. 中心裁剪 224×224 →
3. 按 ImageNet 均值/方差做通道归一化（和 DINOv2/ViT 推理端一致）→
4. 组织成 NHWC（[N, 224, 224, 3]）浮点数组，写到 .npy（X_train.npy, y_train.npy, X_test.npy, y_test.npy）
"""

# === 参数 ===
image_size = 224                            # 图片要求固定尺寸(224,224,3)
short_edge = 256                            # 等比缩放的最短边
# data_root = "dataset"                       # 原始数据根目录
# output_dir = "data_to_npy"                  # 预处理输出目录  
data_root = os.path.join(project_root, "data", "dataset")  
output_dir = os.path.join(project_root, "data", "data_to_npy") # <- 相对项目根目录
os.makedirs(output_dir, exist_ok=True)      # 创建输出目录

# === ImageNet 归一化（与 DINOv2 预处理一致） ===
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# === ImageNet mean/std 归一化 ===
def preprocess_image_pil(img: Image.Image, target=image_size, short=short_edge): # 参数 img 期望是 Pillow 的 PIL.Image.Image 实例。
    """
    等比缩放到短边=short（默认256），中心裁剪到 target（默认224），
    然后做 /255 + ImageNet mean/std 归一化。
    返回：H×W×C，float32
    """
    # 1.确保 RGB
    img = img.convert("RGB") # 把图像强制转为 3通道 RGB
    w, h = img.size

    # 2.等比缩放：最短边=short
    scale = float(short) / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale)) # round 可以把 255.6 之类的小数四舍五入到最近的整数，避免系统性偏小或偏大，然后再int
    img = img.resize((new_w, new_h), resample=Image.BICUBIC) # 双三次插值（Bicubic） 进行缩放

    # 3.中心裁剪到 target×target
    left = (new_w - target) // 2
    top  = (new_h - target) // 2
    img = img.crop((left, top, left + target, top + target)) # 中心裁剪为 target×target

    # 4.转 numpy 并做 rescale + normalize
    x = np.asarray(img).astype(np.float32) / 255.0   # [0,1] 把 PIL.Image 转为 NumPy 数组，再归一化到 [0,1]，并保证 float32 精度
    x = (x - IMAGENET_MEAN) / IMAGENET_STD          # 按通道标准化
    return x

# === 按文件夹加载图片 ===
def load_images_from_folder(folder_path, label, max_images=None):
    """
    从指定文件夹中加载图片，图片预处理（尺寸归一化，数值归一化），并附上标签
    :param folder_path: 文件夹路径
    :param label: 对应标签（0或1）
    :param max_images: 限制加载数量（可选）
    :return: (images, labels)
    """
    images = []
    labels = []

    files = os.listdir(folder_path) # 列出目录所有项（文件/子目录）
    if max_images:
        files = files[:max_images] # 可选数量限制

    for filename in tqdm(files, desc=f"Loading {folder_path}"): # 用 tqdm 显示进度条；desc 作为前缀说明正在加载哪个目录
        # 文件过滤
        if not filename.endswith(".jpg"):
            print("【debug】跳过了一个图像")
            continue # 跳过非jpg文件
        path = os.path.join(folder_path, filename)

        try:
            with Image.open(path) as img: # 进入块时打开文件并创建图像对象；块结束时自动关闭文件句柄/释放资源。
                x = preprocess_image_pil(img)  # <<<<<< 关键改为调用预处理

        # 异常处理
        except Exception as e:
            print(f"跳过损坏文件: {filename}, 错误: {e}")
            continue

        images.append(x.astype(np.float32))      # H×W×C
        labels.append(label)

        if (max_images is not None) and (len(images) >= max_images):
            break

    images = np.asarray(images, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)

    return images, labels

# === 处理并保存数据 ===
def process_and_save(split):
    """
    加载 cat/dog 图像，保存为 numpy 文件
    :param split: 'train' or 'test'
    """
    # 路径构建
    cat_folder = os.path.join(data_root, split, "cats")
    dog_folder = os.path.join(data_root, split, "dogs")

    # 加载图片和标签
    cat_images, cat_labels = load_images_from_folder(cat_folder, label=0)
    dog_images, dog_labels = load_images_from_folder(dog_folder, label=1)

    # 数据合并
    X = np.concatenate([cat_images, dog_images], axis=0).astype(np.float32)
    y = np.concatenate([cat_labels, dog_labels], axis=0).astype(np.int32)

    # 打乱数据
    np.random.seed(24) # 固定随机种子，保证可复现
    idx = np.random.permutation(len(X)) # 生成随机索引，避免同类数据连续排列
    X = X[idx]
    y = y[idx]

    # 文件保存
    np.save(os.path.join(output_dir, f"X_{split}.npy"), X)
    np.save(os.path.join(output_dir, f"y_{split}.npy"), y)
    print(f"✅ Saved {split} set: {X.shape}, labels: {y.shape}")


if __name__ == "__main__":
    process_and_save("train")
    process_and_save("test")

    # === 加载保存的文件并打印形状 ===
    X_train = np.load(os.path.join(output_dir, "X_train.npy"))
    y_train = np.load(os.path.join(output_dir, "y_train.npy"))
    X_test = np.load(os.path.join(output_dir, "X_test.npy"))
    y_test = np.load(os.path.join(output_dir, "y_test.npy"))

    # print("\n=== 数据形状检查 ===")
    # print("X_train:", X_train.shape)
    # print("y_train:", y_train.shape)
    # print("X_test:", X_test.shape)
    # print("y_test:", y_test.shape)

    # # 小批量自检
    # print("\n=== 小批量自检 ===")
    # X = np.load("data_to_npy/X_train.npy")  # [N,224,224,3]
    # print("per-channel mean:", X.mean(axis=(0, 1, 2)))
    # print("per-channel std: ", X.std(axis=(0, 1, 2)))