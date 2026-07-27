#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进版手写数字图片预处理工具
==============================
核心改进：
  1. 初始中值滤波 → 预先去除椒盐噪声，保护边缘
  2. Retinex 光照归一化 → 大幅提升阴影去除效果
  3. Sauvola 局部自适应阈值 → 完美应对非均匀光照
  4. 自动伽马校正 → 调整整体对比度和亮度
  5. 边缘保持去噪 → 去噪同时不模糊数字边缘
  6. 连通域分析 (8连通) → 确保数字完整性，防止断笔分割
  7. 形态学闭运算和膨胀 → 修复笔画断裂和填充小孔
  8. 多策略自动选择 → 自适应不同图片质量

输出：背景纯白(255)、数字纯黑(0)
"""

import cv2
import numpy as np
import os
import sys # Added for preview
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ============================================================
#                        核心处理函数
# ============================================================

def read_image_grayscale(input_path):
    """读取图片并转为灰度图"""
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {input_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def illumination_normalize(gray, kernel_ratio=0.04):
    """
    Retinex 光照归一化
    原理：用大核高斯模糊估计光照背景，然后用除法归一化。
    参数:
        kernel_ratio: 高斯核大小 = max(h,w) * kernel_ratio
                      值越大，估计的背景越平滑，0.04 是经验最优值
    """
    h, w = gray.shape[:2]
    # 核大小必须是奇数，至少为 3
    kernel_size = max(3, int(min(h, w) * kernel_ratio))
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    # 用大高斯模糊估计光照背景
    background = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # 除法归一化（避免除零）
    background = np.where(background < 1, 1, background)
    normalized = gray.astype(np.float32) / background.astype(np.float32)

    # 缩放到 [0, 255]
    normalized = normalized * 128  # 中位亮度
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    return normalized


def auto_gamma_correction(gray):
    """
    自动伽马校正
    根据图像的亮度分布自动决定是否需要伽马校正。
    """
    mean_val = np.mean(gray)

    if mean_val < 80:
        gamma = 0.6  # 大幅提亮暗图
    elif mean_val < 110:
        gamma = 0.8  # 轻微提亮
    elif mean_val > 170:
        gamma = 1.4  # 压暗过亮图
    elif mean_val > 140:
        gamma = 1.2  # 轻微压暗
    else:
        return gray  # 亮度合适，不需校正

    # 应用伽马校正
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255
                      for i in range(256)]).astype(np.uint8)
    return cv2.LUT(gray, table)


def denoise_bilateral(gray):
    """
    边缘保持去噪 (双边滤波)
    去噪的同时保留边缘，适合手写数字。
    """
    return cv2.bilateralFilter(gray, d=5, sigmaColor=20, sigmaSpace=20)


def sauvola_threshold(gray, window_size=25, k=0.2, r=128):
    """
    Sauvola 局部自适应阈值
    对每个像素，在其局部窗口内计算阈值，擅长处理非均匀光照文档。
    """
    # 局部均值
    mean = cv2.boxFilter(gray, cv2.CV_32F, (window_size, window_size))
    # 局部平方均值
    sq_mean = cv2.boxFilter(gray.astype(np.float32) ** 2, cv2.CV_32F,
                            (window_size, window_size))
    # 局部标准差
    variance = sq_mean - mean ** 2
    variance = np.maximum(variance, 0)  # 防止浮点误差导致负数
    std = np.sqrt(variance)

    # Sauvola 阈值公式
    threshold = mean * (1.0 + k * (std / r - 1.0))

    # 二值化：像素值 < 阈值 → 前景（数字），否则 → 背景
    binary = (gray.astype(np.float32) < threshold).astype(np.uint8) * 255

    return binary


def clean_binary(binary, min_area=50, close_kernel_size=3, dilate_strength=1, connectivity=8):
    """
    二值图清理与笔画修复

    步骤：
      1. 去除小面积噪声（连通域分析）
      2. 形态学闭运算填充笔画中的小孔
      3. 可选膨胀修复断裂笔画

    参数:
        binary:              输入二值图像 (前景255, 背景0)
        min_area:            最小连通域面积，小于此值的被视为噪声
        close_kernel_size:   闭运算核大小，用于填充内部小孔
        dilate_strength:     膨胀迭代次数，用于加粗笔画和连接断裂
        connectivity:        连通域分析的连通性 (4 或 8)。8连通更适合手写数字。
    """
    # 此时 binary: 数字=255(白)，背景=0(黑)

    # --- 1. 连通域分析，去除小噪声 ---
    # 使用 8 连通，更能连接对角线相连的笔画，减少数字被拆分成多个小块
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=connectivity)
    cleaned = np.zeros_like(binary)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    # --- 2. 闭运算：先膨胀后腐蚀，填充笔画内部小孔 ---
    if close_kernel_size > 0:
        # 核大小必须是奇数
        k = close_kernel_size if close_kernel_size % 2 == 1 else close_kernel_size + 1
        if k < 1: k = 1 # Minimum kernel size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    # --- 3. 可选膨胀，修复断裂笔画 ---
    if dilate_strength > 0:
        # 使用 2x2 核进行膨胀，避免过粗
        kernel_d = np.ones((2, 2), np.uint8)
        cleaned = cv2.dilate(cleaned, kernel_d, iterations=dilate_strength)

    # --- 4. 反转为：背景白(255)，数字黑(0) ---
    final = cv2.bitwise_not(cleaned)

    return final


# ============================================================
#                        质量评估
# ============================================================

def assess_binary_quality(binary):
    """
    评估二值化质量，返回 0~1 之间的分数
    指标：
      - 前景占比是否合理（5%~40%，过高说明噪声多，过低说明数字丢失）
    """
    # binary: 数字=255(前景)，背景=0
    fg_ratio = np.mean(binary > 0)

    # 前景占比评分 (最佳 5%~40%)
    if 0.05 <= fg_ratio <= 0.40:
        ratio_score = 1.0
    elif fg_ratio < 0.05:
        ratio_score = fg_ratio / 0.05
    else:
        ratio_score = max(0, 1.0 - (fg_ratio - 0.40) / 0.30)

    return ratio_score


# ============================================================
#                        主处理函数
# ============================================================

def preprocess_handwritten_image(
        input_path,
        output_path=None,
        strategy='auto',  # 'auto', 'sauvola', 'otsu', 'adaptive'
        median_blur_ksize=3, # NEW: 初始中值滤波核大小
        remove_illumination=True,
        auto_gamma=True,
        denoise=True,
        sauvola_k=0.25,
        sauvola_window=31,
        dilate_strength=1,
        min_noise_area=30,
        close_morph_kernel=3, # NEW: 闭运算核大小
        connectivity=8, # NEW: 连通域分析连通性
        show_preview=True,
        verbose=True,
):
    """
    手写数字图片预处理主函数（改进版）

    参数:
        input_path:          输入图片路径
        output_path:         输出路径 (None 则自动生成)
        strategy:            二值化策略 ('sauvola' | 'otsu' | 'adaptive' | 'auto')
        median_blur_ksize:   初始中值滤波核大小 (奇数，>=3)。设置为 0 或 1 则跳过。
        remove_illumination: 是否做光照归一化（强烈建议开启）
        auto_gamma:          是否自动伽马校正
        denoise:             是否去噪 (双边滤波)
        sauvola_k:           Sauvola灵敏度参数 (0.1~0.5，越小数字越粗)
        sauvola_window:      Sauvola窗口大小 (必须奇数，越大越平滑)
        dilate_strength:     形态学膨胀强度 (0=不膨胀，1=轻度，2=中度)
        min_noise_area:      最小连通域面积（小于此值的视为噪声）
        close_morph_kernel:  形态学闭运算核大小 (奇数，>=1)，用于填充笔画内部小孔。
        connectivity:        连通域分析的连通性 (4 或 8)。8连通更适合手写数字。
        show_preview:        是否显示处理预览图 (需要 matplotlib)
        verbose:             是否打印详细日志

    返回:
        (final_image, metadata_dict)
    """
    if verbose:
        print("=" * 55)
        print("  手写数字预处理 (改进版)")
        print("=" * 55)

    # ---- Step 1: 读取图片 ----
    if verbose:
        print("[1/7] 读取图片...")
    original, gray = read_image_grayscale(input_path)
    h, w = gray.shape[:2]
    if verbose:
        print(f"      尺寸: {w}x{h}, 亮度均值: {np.mean(gray):.1f}")

    # ---- Step 2: 初始中值滤波去噪 (NEW) ----
    if median_blur_ksize > 1 and median_blur_ksize % 2 == 1:
        if verbose:
            print(f"[2/7] 初始中值滤波 (ksize={median_blur_ksize})...")
        gray = cv2.medianBlur(gray, median_blur_ksize)
    else:
        if verbose:
            print("[2/7] 跳过初始中值滤波")

    # ---- Step 3: 光照归一化 (Retinex) ----
    if remove_illumination:
        if verbose:
            print("[3/7] 光照归一化 (Retinex)...")
        gray = illumination_normalize(gray)
    else:
        if verbose:
            print("[3/7] 跳过光照归一化")

    # ---- Step 4: 伽马校正 + 边缘保持去噪 ----
    if verbose:
        print("[4/7] 对比度调整 + 边缘保持去噪...")
    if auto_gamma:
        old_mean = np.mean(gray)
        gray = auto_gamma_correction(gray)
        new_mean = np.mean(gray)
        if verbose and abs(old_mean - new_mean) > 3:
            print(f"      自动伽马校正: 均值 {old_mean:.0f} → {new_mean:.0f}")

    if denoise:
        gray = denoise_bilateral(gray)

    # ---- Step 5: 二值化 (多策略自动选择) ----
    if verbose:
        print(f"[5/7] 二值化 (策略: {strategy})...")

    if strategy == 'auto':
        candidates = {}
        # 策略A: Sauvola (k=0.2, 较敏感)
        bin_a = sauvola_threshold(gray, window_size=sauvola_window, k=0.20)
        candidates['sauvola_k0.20'] = (bin_a, assess_binary_quality(bin_a))
        # 策略B: Sauvola (k=0.35, 较保守)
        bin_b = sauvola_threshold(gray, window_size=sauvola_window, k=0.35)
        candidates['sauvola_k0.35'] = (bin_b, assess_binary_quality(bin_b))
        # 策略C: Otsu
        _, bin_c = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        candidates['otsu'] = (bin_c, assess_binary_quality(bin_c))
        # 策略D: 自适应阈值 (高斯加权)
        bin_d = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
        candidates['adaptive'] = (bin_d, assess_binary_quality(bin_d))

        # 选择得分最高的
        best_name = max(candidates, key=lambda k: candidates[k][1])
        binary = candidates[best_name][0]
        if verbose:
            for name, (_, score) in candidates.items():
                marker = " ✓" if name == best_name else ""
                print(f"      {name}: 质量评分 {score:.3f}{marker}")
            print(f"      自动选择: {best_name}")

    elif strategy == 'sauvola':
        binary = sauvola_threshold(gray, window_size=sauvola_window, k=sauvola_k)
    elif strategy == 'otsu':
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif strategy == 'adaptive':
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
    else:
        raise ValueError(f"未知策略: {strategy}")

    # ---- Step 6: 形态学清理 & 笔画修复 ----
    if verbose:
        print("[6/7] 形态学清理 & 笔画修复...")
    cleaned = clean_binary(binary,
                           min_area=min_noise_area,
                           close_kernel_size=close_morph_kernel,
                           dilate_strength=dilate_strength,
                           connectivity=connectivity)

    # ---- Step 7: 保存结果 ----
    if verbose:
        print("[7/7] 保存结果...")
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_processed{ext}"

    cv2.imwrite(output_path, cleaned)

    if verbose:
        fg_ratio = 1.0 - np.mean(cleaned > 127)
        print(f"\n  ✅ 完成! 输出: {output_path}")
        print(f"     前景(数字)占比: {fg_ratio:.1%}")
        print(f"     背景颜色: 白色(255), 数字颜色: 黑色(0)")
        print("=" * 55)

    # ---- 可选：预览 ----
    if show_preview and MATPLOTLIB_AVAILABLE:
        preview(original, cleaned)
    elif show_preview and not MATPLOTLIB_AVAILABLE:
        print("\n[WARN] 无法显示预览图：未安装 matplotlib 库。请安装：pip install matplotlib\n")


    metadata = {
        'output_path': output_path,
        'strategy': strategy,
        'image_size': (w, h),
        'fg_ratio': 1.0 - np.mean(cleaned > 127),
        'params': {
            'median_blur_ksize': median_blur_ksize,
            'remove_illumination': remove_illumination,
            'auto_gamma': auto_gamma,
            'denoise': denoise,
            'sauvola_k': sauvola_k,
            'sauvola_window': sauvola_window,
            'dilate_strength': dilate_strength,
            'min_noise_area': min_noise_area,
            'close_morph_kernel': close_morph_kernel,
            'connectivity': connectivity,
        }
    }

    return cleaned, metadata


# ============================================================
#                        可视化预览
# ============================================================

def preview(original, processed):
    """并排显示原图与处理结果"""
    if not MATPLOTLIB_AVAILABLE:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 原图
    orig_rgb = (cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                if len(original.shape) == 3 else original)
    axes[0].imshow(orig_rgb, cmap='gray')
    axes[0].set_title('原始图片', fontsize=14)
    axes[0].axis('off')

    # 处理结果
    axes[1].imshow(processed, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('处理后 (背景白·数字黑)', fontsize=14)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


# ============================================================
#                        批量处理
# ============================================================

def batch_process(input_dir, output_dir, **kwargs):
    """批量处理目录中所有图片"""
    os.makedirs(output_dir, exist_ok=True)
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')

    files = [f for f in os.listdir(input_dir)
             if f.lower().endswith(exts)]

    if not files:
        print(f"❌ 目录 {input_dir} 中没有找到图片文件")
        return

    print(f"\n批量处理 {len(files)} 张图片...\n")

    for i, fname in enumerate(files, 1):
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, f"proc_{fname}")
        print(f"\n[{i}/{len(files)}] {fname}")
        try:
            preprocess_handwritten_image(
                in_path, out_path,
                show_preview=False, verbose=True, **kwargs
            )
        except Exception as e:
            print(f"  ❌ 失败: {e}")


# ============================================================
#                      使用示例
# ============================================================

if __name__ == "__main__":
    # =========== 单张图片处理 ===========
    preprocess_handwritten_image(
        input_path=r"C:\Develop\python\python_study\lab10\test_figs\test.jpg",  # ← 改成你的图片路径
        output_path=r"C:\Develop\python\python_study\lab10\test_figs\pre_out.jpg",  # 输出路径（可选）
        strategy='auto',  # 'sauvola' | 'otsu' | 'adaptive' | 'auto'
        median_blur_ksize=3, # NEW: 初始中值滤波核大小 (奇数，>=3)。设置为 0 或 1 则跳过。
        remove_illumination=True,  # 光照归一化（强烈推荐）
        auto_gamma=True,  # 自动伽马校正
        denoise=True,  # 边缘保持去噪
        sauvola_k=0.25,  # Sauvola 灵敏度 (0.15~0.40)
        sauvola_window=31,  # 窗口大小 (15~51, 奇数)
        dilate_strength=1,  # 笔画加粗 (0~2)
        min_noise_area=30,  # 噪声过滤阈值
        close_morph_kernel=3, # NEW: 闭运算核大小 (奇数，>=1)
        connectivity=8, # NEW: 连通域分析连通性 (4 或 8)。8连通更适合手写数字。
        show_preview=True,  # 显示对比图
    )

    # =========== 快速调参建议 ===========
    #
    # 如果数字笔画断裂、太细：
    #   → 尝试增大 dilate_strength=2, 减小 sauvola_k=0.15
    #   → 尝试增大 close_morph_kernel=5, 确保 connectivity=8
    #
    # 如果背景有大量噪声：
    #   → 尝试增大 min_noise_area=80, 增大 sauvola_k=0.35
    #   → 尝试增大 median_blur_ksize=5
    #
    # 如果阴影仍然存在：
    #   → 确保 remove_illumination=True, 尝试增大 sauvola_window=51
    #
    # 如果数字边缘模糊：
    #   → 确保 denoise=True (双边滤波)，尝试减小 median_blur_ksize=3
    #   → 尝试 strategy='auto' 让程序自动选择最佳策略
    #
    # ====================================