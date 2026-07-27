"""
Improved Multi-Digit Detector

功能：
- 从输入图片中检测多个手写数字区域
- 自动判断白底黑字 / 黑底白字
- 使用 Otsu 自适应阈值进行二值化
- 使用形态学 closing 连接断裂笔画
- 使用 opening 去除小噪声
- 连通域分析提取每个数字
- 合并过近或重叠的候选区域
- 按阅读顺序排序：从上到下，从左到右
- 将每个数字预处理成 MNIST 风格的 28x28 浮点输入
- 输出 pix_region_N.dat
- 输出 digit_regions.csv / digit_regions.json
- 输出预览图 pix_region_N_preview.png
- 输出检测框 debug_detect_overlay.png

用法：
  python multi_digit_detector.py

或：
  python multi_digit_detector.py --input ./digit.png --out_dir ./hls_1/src

如果数字太小，可以试：
  python multi_digit_detector.py --fit_size 24

如果断笔较多，可以试：
  python multi_digit_detector.py --close_radius 2 --merge_gap 10

如果噪声较多，可以试：
  python multi_digit_detector.py --open_radius 1 --min_area 80
"""

import os
import csv
import json
import glob
import argparse
from collections import deque

import numpy as np
from PIL import Image, ImageOps, ImageDraw


# ============================================================
# Default paths
# ============================================================

# DEFAULT_INPUT_IMAGE = "./test_figs/digit.png"
DEFAULT_INPUT_IMAGE = "./test_figs/test_preprocessed.jpg"
DEFAULT_OUT_DIR = "./hls_1/src/digit_detector_out"


# ============================================================
# PIL compatibility
# ============================================================

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS
    RESAMPLE_NEAREST = Image.NEAREST


# ============================================================
# Utility
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def clean_old_outputs(out_dir):
    """
    清理旧的检测结果，避免新检测数字数量减少时，旧文件残留导致 C/HLS 测试读错。
    """
    patterns = [
        "pix_region_*.dat",
        "pix_region_*_preview.png",
        "digit_regions.json",
        "digit_regions.csv",
        "debug_detect_overlay.png",
        "debug_binary_mask.png",
    ]

    for pat in patterns:
        for f in glob.glob(os.path.join(out_dir, pat)):
            try:
                os.remove(f)
            except OSError:
                pass


def load_grayscale_image(path):
    """
    读取图片并转换为灰度图。
    如果图片带 alpha 通道，先用白色背景合成，避免透明区域变成黑色噪声。
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)

    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.alpha_composite(img.convert("RGBA"))
        img = bg.convert("L")
    else:
        img = img.convert("L")

    return img


def detect_background_is_light(gray_arr):
    """
    根据图片边缘像素判断背景是亮色还是暗色。
    True  表示白底黑字
    False 表示黑底白字
    """
    h, w = gray_arr.shape

    border_pixels = np.concatenate([
        gray_arr[0, :],
        gray_arr[-1, :],
        gray_arr[:, 0],
        gray_arr[:, -1],
    ])

    border_median = float(np.median(border_pixels))
    return border_median >= 128.0


# ============================================================
# Otsu threshold
# ============================================================

def otsu_threshold_uint8(arr):
    """
    对 uint8 图像计算 Otsu 阈值。
    arr 可以是任意 shape。
    """
    flat = arr.astype(np.uint8).ravel()
    hist = np.bincount(flat, minlength=256).astype(np.float64)

    total = flat.size
    if total == 0:
        return 0

    sum_total = np.dot(np.arange(256), hist)

    weight_bg = 0.0
    sum_bg = 0.0
    max_var = -1.0
    threshold = 0

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue

        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += t * hist[t]

        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        between_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

        if between_var > max_var:
            max_var = between_var
            threshold = t

    return int(threshold)


# ============================================================
# Binary morphology
# ============================================================

def binary_dilate(mask, radius=1):
    """
    二值膨胀。
    mask: bool ndarray, True 表示 ink。
    """
    if radius <= 0:
        return mask.copy()

    h, w = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)

    out = np.zeros_like(mask, dtype=bool)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            y0 = radius + dy
            x0 = radius + dx
            out |= padded[y0:y0 + h, x0:x0 + w]

    return out


def binary_erode(mask, radius=1):
    """
    二值腐蚀。
    mask: bool ndarray, True 表示 ink。
    """
    if radius <= 0:
        return mask.copy()

    h, w = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)

    out = np.ones_like(mask, dtype=bool)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            y0 = radius + dy
            x0 = radius + dx
            out &= padded[y0:y0 + h, x0:x0 + w]

    return out


def binary_close(mask, radius=1):
    """
    closing = dilate then erode
    用于连接断裂笔画。
    """
    if radius <= 0:
        return mask.copy()
    return binary_erode(binary_dilate(mask, radius), radius)


def binary_open(mask, radius=1):
    """
    opening = erode then dilate
    用于去除小噪声。
    """
    if radius <= 0:
        return mask.copy()
    return binary_dilate(binary_erode(mask, radius), radius)


# ============================================================
# Connected components
# ============================================================

def find_connected_components(mask):
    """
    连通域检测。
    mask: bool ndarray, True 表示 ink。
    使用 8 连通。
    返回：
      [
        {
          "bbox": [x0, y0, x1, y1],  # x1/y1 inclusive
          "width": ...,
          "height": ...,
          "area": ...,
          "cx": ...,
          "cy": ...
        },
        ...
      ]
    """
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []

    ys, xs = np.nonzero(mask)

    for start_y, start_x in zip(ys, xs):
        if visited[start_y, start_x]:
            continue

        q = deque()
        q.append((start_x, start_y))
        visited[start_y, start_x] = True

        area = 0
        min_x = max_x = start_x
        min_y = max_y = start_y

        while q:
            x, y = q.popleft()

            area += 1

            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y

            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue

                    nx = x + dx
                    ny = y + dy

                    if nx < 0 or nx >= w or ny < 0 or ny >= h:
                        continue

                    if visited[ny, nx]:
                        continue

                    if not mask[ny, nx]:
                        continue

                    visited[ny, nx] = True
                    q.append((nx, ny))

        bw = max_x - min_x + 1
        bh = max_y - min_y + 1

        components.append({
            "bbox": [int(min_x), int(min_y), int(max_x), int(max_y)],
            "width": int(bw),
            "height": int(bh),
            "area": int(area),
            "cx": int(round((min_x + max_x) / 2)),
            "cy": int(round((min_y + max_y) / 2)),
        })

    return components


def filter_components(components, image_w, image_h, args):
    """
    过滤掉过小、过大、过扁、过细、贴边噪声等候选区域。
    """
    filtered = []

    if args.min_area is None:
        min_area = max(20, int(image_w * image_h * 0.00003))
    else:
        min_area = args.min_area

    max_area = int(image_w * image_h * args.max_area_ratio)

    for c in components:
        x0, y0, x1, y1 = c["bbox"]
        bw = c["width"]
        bh = c["height"]
        area = c["area"]

        if area < min_area:
            continue

        if area > max_area:
            continue

        if bw < args.min_width or bh < args.min_height:
            continue

        if bw > image_w * args.max_width_ratio:
            continue

        if bh > image_h * args.max_height_ratio:
            continue

        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if aspect > args.max_aspect:
            continue

        # 过滤紧贴图片边缘的横线、边框等
        if args.skip_edge > 0:
            if x0 < args.skip_edge or y0 < args.skip_edge:
                continue
            if x1 > image_w - 1 - args.skip_edge:
                continue
            if y1 > image_h - 1 - args.skip_edge:
                continue

        filtered.append(c)

    return filtered


# ============================================================
# Merge bboxes
# ============================================================

def bboxes_overlap_or_close(a, b, gap):
    """
    判断两个 bbox 是否重叠或距离足够近。
    bbox 格式：[x0, y0, x1, y1]，x1/y1 inclusive。
    """
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    ax0 -= gap
    ay0 -= gap
    ax1 += gap
    ay1 += gap

    bx0 -= gap
    by0 -= gap
    bx1 += gap
    by1 += gap

    if ax1 < bx0:
        return False
    if bx1 < ax0:
        return False
    if ay1 < by0:
        return False
    if by1 < ay0:
        return False

    return True


def merge_components(components, merge_gap=6):
    """
    合并重叠或相近的候选区域。
    主要用于同一个数字因为断笔被拆成多个连通域的情况。
    """
    n = len(components)
    if n <= 1:
        return components

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if bboxes_overlap_or_close(
                components[i]["bbox"],
                components[j]["bbox"],
                merge_gap,
            ):
                union(i, j)

    groups = {}

    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(components[i])

    merged = []

    for group in groups.values():
        x0 = min(g["bbox"][0] for g in group)
        y0 = min(g["bbox"][1] for g in group)
        x1 = max(g["bbox"][2] for g in group)
        y1 = max(g["bbox"][3] for g in group)

        area = sum(g["area"] for g in group)
        bw = x1 - x0 + 1
        bh = y1 - y0 + 1

        merged.append({
            "bbox": [int(x0), int(y0), int(x1), int(y1)],
            "width": int(bw),
            "height": int(bh),
            "area": int(area),
            "cx": int(round((x0 + x1) / 2)),
            "cy": int(round((y0 + y1) / 2)),
        })

    return merged


# ============================================================
# Sorting
# ============================================================

def sort_regions_reading_order(regions):
    """
    按阅读顺序排序：
    - 先按行从上到下
    - 每行从左到右

    行间距自适应，基于候选数字高度估计。
    """
    if not regions:
        return []

    heights = [r["height"] for r in regions]
    median_h = float(np.median(heights))
    row_threshold = max(20.0, median_h * 0.65)

    regions_sorted_y = sorted(regions, key=lambda r: r["cy"])

    rows = []

    for r in regions_sorted_y:
        placed = False

        for row in rows:
            row_cy = np.mean([x["cy"] for x in row])
            if abs(r["cy"] - row_cy) <= row_threshold:
                row.append(r)
                placed = True
                break

        if not placed:
            rows.append([r])

    rows.sort(key=lambda row: np.mean([r["cy"] for r in row]))

    ordered = []
    for row in rows:
        row.sort(key=lambda r: r["cx"])
        ordered.extend(row)

    return ordered


# ============================================================
# Image shifting
# ============================================================

def shift_image_zero_fill(arr, shift_y, shift_x):
    """
    平移二维数组，空出来的区域补 0，不循环。
    """
    h, w = arr.shape
    out = np.zeros_like(arr)

    src_y0 = max(0, -shift_y)
    src_y1 = min(h, h - shift_y)

    dst_y0 = max(0, shift_y)
    dst_y1 = min(h, h + shift_y)

    src_x0 = max(0, -shift_x)
    src_x1 = min(w, w - shift_x)

    dst_x0 = max(0, shift_x)
    dst_x1 = min(w, w + shift_x)

    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]

    return out


# ============================================================
# MNIST-style preprocessing
# ============================================================

def preprocess_digit_to_mnist(gray_arr, region, background_is_light, args):
    """
    将检测出的数字区域转成 28x28 MNIST 风格浮点数组。

    输出 normalized：
      paper = 0.0
      ink   = 1.0
    """
    h, w = gray_arr.shape

    x0, y0, x1, y1 = region["bbox"]

    digit_w = x1 - x0 + 1
    digit_h = y1 - y0 + 1
    larger = max(digit_w, digit_h)

    pad = int(round(larger * args.pad_ratio))

    crop_x0 = max(0, x0 - pad)
    crop_y0 = max(0, y0 - pad)
    crop_x1 = min(w, x1 + 1 + pad)
    crop_y1 = min(h, y1 + 1 + pad)

    crop_gray = gray_arr[crop_y0:crop_y1, crop_x0:crop_x1].astype(np.float32)

    # 转为 ink 强度，paper=0, ink=1
    if background_is_light:
        ink = 1.0 - crop_gray / 255.0
    else:
        ink = crop_gray / 255.0

    ink = np.clip(ink, 0.0, 1.0)

    # 对 ROI 做对比度拉伸，增强浅色笔迹
    hi = float(np.percentile(ink, args.contrast_high_percentile))
    lo = float(np.percentile(ink, args.contrast_low_percentile))

    if hi > lo + 1e-6:
        ink = (ink - lo) / (hi - lo)
        ink = np.clip(ink, 0.0, 1.0)

    # 抑制非常弱的背景噪声
    if args.suppress_background > 0:
        ink[ink < args.suppress_background] = 0.0

    ch, cw = ink.shape

    if cw <= 0 or ch <= 0:
        canvas = np.zeros((28, 28), dtype=np.float32)
        return canvas.flatten().tolist(), [crop_x0, crop_y0, crop_x1, crop_y1]

    # 将 crop 按比例缩放，使最大边为 fit_size
    fit_size = int(args.fit_size)
    fit_size = max(1, min(28, fit_size))

    scale = fit_size / float(max(cw, ch))
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))

    ink_img = Image.fromarray(np.uint8(np.clip(ink * 255.0, 0, 255)), mode="L")
    ink_resized = ink_img.resize((new_w, new_h), RESAMPLE_LANCZOS)

    resized = np.asarray(ink_resized).astype(np.float32) / 255.0
    resized = np.clip(resized, 0.0, 1.0)

    # 先居中放入 28x28
    canvas = np.zeros((28, 28), dtype=np.float32)

    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2

    canvas[paste_y:paste_y + new_h, paste_x:paste_x + new_w] = resized

    # 根据质心再居中，更接近 MNIST
    mass = float(canvas.sum())

    if mass > 1e-6 and args.center_by_mass:
        yy, xx = np.indices(canvas.shape)
        cx = float((canvas * xx).sum() / mass)
        cy = float((canvas * yy).sum() / mass)

        target = 13.5

        shift_x = int(round(target - cx))
        shift_y = int(round(target - cy))

        # 限制最大平移，避免极端情况下把数字移出画布
        shift_x = max(-args.max_center_shift, min(args.max_center_shift, shift_x))
        shift_y = max(-args.max_center_shift, min(args.max_center_shift, shift_y))

        canvas = shift_image_zero_fill(canvas, shift_y, shift_x)

    canvas = np.clip(canvas, 0.0, 1.0)

    return canvas.flatten().tolist(), [crop_x0, crop_y0, crop_x1, crop_y1]


# ============================================================
# Save outputs
# ============================================================

def save_dat_file(path, values):
    """
    保存 784 个 float。
    """
    with open(path, "w", newline="") as f:
        for i, v in enumerate(values):
            f.write(f"{float(v):.6f}")
            if i < len(values) - 1:
                f.write(",")


def save_preview(path, values, scale=5):
    """
    保存预览图。
    values:
      paper = 0
      ink   = 1

    预览图：
      paper = white
      ink   = black
    """
    arr = np.asarray(values, dtype=np.float32).reshape(28, 28)
    img_arr = np.uint8(np.clip((1.0 - arr) * 255.0, 0, 255))
    img = Image.fromarray(img_arr, mode="L")
    img = img.resize((28 * scale, 28 * scale), RESAMPLE_NEAREST)
    img.save(path)


def save_binary_mask(path, mask):
    """
    保存二值 mask，用于调试。
    ink=True 显示为黑色，背景显示为白色。
    """
    arr = np.where(mask, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def save_overlay(path, original_img, regions):
    """
    保存检测框 overlay。
    """
    overlay = original_img.convert("RGB")
    draw = ImageDraw.Draw(overlay)

    for i, r in enumerate(regions):
        x0, y0, x1, y1 = r["bbox"]
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)
        draw.text((x0, max(0, y0 - 14)), str(i), fill=(255, 0, 0))

    overlay.save(path)


# ============================================================
# Main detection pipeline
# ============================================================

def detect_digits(img, args, out_dir):
    gray_arr = np.asarray(img).astype(np.uint8)
    h, w = gray_arr.shape

    background_is_light = detect_background_is_light(gray_arr)

    if background_is_light:
        ink_score = 255 - gray_arr
    else:
        ink_score = gray_arr.copy()

    # Otsu 阈值
    otsu_t = otsu_threshold_uint8(ink_score)

    if args.threshold is not None:
        threshold = int(args.threshold)
    else:
        threshold = int(max(args.threshold_min, otsu_t * args.otsu_scale))

    threshold = max(0, min(255, threshold))

    raw_mask = ink_score >= threshold

    # 形态学处理
    mask = raw_mask.copy()

    if args.close_radius > 0:
        mask = binary_close(mask, args.close_radius)

    if args.open_radius > 0:
        mask = binary_open(mask, args.open_radius)

    # 保存二值图调试
    save_binary_mask(os.path.join(out_dir, "debug_binary_mask.png"), mask)

    components = find_connected_components(mask)

    print(f"Image: {w}x{h}")
    print(f"Background: {'light' if background_is_light else 'dark'}")
    print(f"Otsu threshold on ink_score: {otsu_t}")
    print(f"Used threshold: {threshold}")
    print(f"Raw connected components: {len(components)}")

    filtered = filter_components(components, w, h, args)
    print(f"After filtering: {len(filtered)}")

    merged = merge_components(filtered, merge_gap=args.merge_gap)
    print(f"After merging: {len(merged)}")

    # 合并后再过滤一次，防止合并出异常大区域
    merged = filter_components(merged, w, h, args)
    print(f"After post-filtering: {len(merged)}")

    # 最多保留 N 个
    if args.max_digits > 0 and len(merged) > args.max_digits:
        merged.sort(key=lambda r: r["area"], reverse=True)
        merged = merged[:args.max_digits]
        print(f"Selected top {args.max_digits} by area")

    regions = sort_regions_reading_order(merged)

    return regions, mask, background_is_light


def main():
    parser = argparse.ArgumentParser(
        description="Improved multi-digit detector for HLS MNIST inference"
    )

    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_IMAGE)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)

    # Threshold
    parser.add_argument("--threshold", type=int, default=None,
                        help="Manual ink threshold. If not set, use Otsu.")
    parser.add_argument("--threshold_min", type=int, default=25,
                        help="Minimum threshold for ink_score.")
    parser.add_argument("--otsu_scale", type=float, default=0.9,
                        help="Use threshold = max(threshold_min, otsu * otsu_scale).")

    # Morphology
    parser.add_argument("--close_radius", type=int, default=0,
                        help="Morphological closing radius. Larger connects broken strokes.")
    parser.add_argument("--open_radius", type=int, default=0,
                        help="Morphological opening radius. Larger removes noise but may damage thin strokes.")

    # Component filtering
    parser.add_argument("--min_area", type=int, default=None)
    parser.add_argument("--min_width", type=int, default=4)
    parser.add_argument("--min_height", type=int, default=4)
    parser.add_argument("--max_area_ratio", type=float, default=0.50)
    parser.add_argument("--max_width_ratio", type=float, default=0.95)
    parser.add_argument("--max_height_ratio", type=float, default=0.95)
    parser.add_argument("--max_aspect", type=float, default=12.0)
    parser.add_argument("--skip_edge", type=int, default=0)

    # Merge and count
    parser.add_argument("--merge_gap", type=int, default=6)
    parser.add_argument("--max_digits", type=int, default=20)

    # MNIST preprocessing
    parser.add_argument("--pad_ratio", type=float, default=0.08)
    parser.add_argument("--fit_size", type=int, default=18,
                        help="Resize digit max side to this size before centering into 28x28. MNIST-like default is 20.")
    parser.add_argument("--center_by_mass", action="store_true", default=True)
    parser.add_argument("--no_center_by_mass", dest="center_by_mass", action="store_false")
    parser.add_argument("--max_center_shift", type=int, default=6)
    parser.add_argument("--contrast_low_percentile", type=float, default=1.0)
    parser.add_argument("--contrast_high_percentile", type=float, default=99.5)
    parser.add_argument("--suppress_background", type=float, default=0.02)

    args = parser.parse_args()

    ensure_dir(args.out_dir)
    clean_old_outputs(args.out_dir)

    regions_json_path = os.path.join(args.out_dir, "digit_regions.json")
    regions_csv_path = os.path.join(args.out_dir, "digit_regions.csv")

    print("=" * 70)
    print("Improved Multi-Digit Detector")
    print("=" * 70)
    print(f"Input:  {args.input}")
    print(f"Output: {args.out_dir}")
    print()

    img = load_grayscale_image(args.input)
    w, h = img.size

    regions, mask, background_is_light = detect_digits(img, args, args.out_dir)

    print()
    print(f"Detected digit candidates: {len(regions)}")

    for i, r in enumerate(regions):
        b = r["bbox"]
        print(
            f"  Digit {i}: "
            f"bbox=({b[0]},{b[1]})-({b[2]},{b[3]}), "
            f"size={r['width']}x{r['height']}, "
            f"area={r['area']}, "
            f"center=({r['cx']},{r['cy']})"
        )

    gray_arr = np.asarray(img).astype(np.uint8)

    # 生成每个 digit 的 .dat 和 preview
    for i, region in enumerate(regions):
        values, crop_bbox = preprocess_digit_to_mnist(
            gray_arr=gray_arr,
            region=region,
            background_is_light=background_is_light,
            args=args,
        )

        dat_name = f"pix_region_{i}.dat"
        preview_name = f"pix_region_{i}_preview.png"

        dat_path = os.path.join(args.out_dir, dat_name)
        preview_path = os.path.join(args.out_dir, preview_name)

        save_dat_file(dat_path, values)
        save_preview(preview_path, values, scale=5)

        arr = np.asarray(values, dtype=np.float32)

        region["crop_bbox"] = crop_bbox
        region["dat_file"] = dat_name
        region["preview_file"] = preview_name
        region["value_range"] = [float(arr.min()), float(arr.max())]
        region["pix_mean"] = float(arr.mean())
        region["pix_gt_half"] = int(np.sum(arr > 0.5))
        region["pix_gt_tenth"] = int(np.sum(arr > 0.1))
        region["pix_max"] = float(arr.max())

    # 保存 overlay
    overlay_path = os.path.join(args.out_dir, "debug_detect_overlay.png")
    save_overlay(overlay_path, img, regions)

    # 输出质量检查
    if regions:
        all_gt_half = [r["pix_gt_half"] for r in regions]
        all_gt_tenth = [r["pix_gt_tenth"] for r in regions]
        all_means = [r["pix_mean"] for r in regions]

        avg_half = sum(all_gt_half) / len(all_gt_half)
        avg_tenth = sum(all_gt_tenth) / len(all_gt_tenth)
        avg_mean = sum(all_means) / len(all_means)

        print()
        print("Preprocessing quality check:")
        print("  MNIST rough baseline: mean≈0.13, pixels>0.5≈100, pixels>0.1≈130")
        print(f"  pixels > 0.5: avg={avg_half:.1f}, min={min(all_gt_half)}, max={max(all_gt_half)}")
        print(f"  pixels > 0.1: avg={avg_tenth:.1f}, min={min(all_gt_tenth)}, max={max(all_gt_tenth)}")
        print(f"  pixel mean:   avg={avg_mean:.4f}")

        if avg_half < 35:
            print("  WARNING: ink pixels are very sparse. Recognition may be poor.")
            print("           Try: --fit_size 24")
            print("           Or use darker/thicker handwriting.")
        elif avg_half > 180:
            print("  WARNING: ink pixels are too dense. Recognition may be poor.")
            print("           Try: --fit_size 18 or reduce stroke thickness.")

    # 保存 CSV
    with open(regions_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index",
            "center_x",
            "center_y",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "dat_file",
        ])

        for i, r in enumerate(regions):
            writer.writerow([
                i,
                r["cx"],
                r["cy"],
                r["bbox"][0],
                r["bbox"][1],
                r["width"],
                r["height"],
                r["dat_file"],
            ])

    # 保存 JSON
    metadata = {
        "image": args.input,
        "image_size": [w, h],
        "output_dir": args.out_dir,
        "num_digits": len(regions),
        "background": "light" if background_is_light else "dark",
        "params": {
            "threshold": args.threshold,
            "threshold_min": args.threshold_min,
            "otsu_scale": args.otsu_scale,
            "close_radius": args.close_radius,
            "open_radius": args.open_radius,
            "merge_gap": args.merge_gap,
            "pad_ratio": args.pad_ratio,
            "fit_size": args.fit_size,
            "center_by_mass": args.center_by_mass,
            "suppress_background": args.suppress_background,
        },
        "regions": regions,
    }

    with open(regions_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print()
    print("[OK] Extraction finished")
    print(f"  Digits:      {len(regions)}")
    print(f"  DAT files:   {os.path.join(args.out_dir, 'pix_region_*.dat')}")
    print(f"  CSV:         {regions_csv_path}")
    print(f"  JSON:        {regions_json_path}")
    print(f"  Overlay:     {overlay_path}")
    print(f"  Binary mask: {os.path.join(args.out_dir, 'debug_binary_mask.png')}")


if __name__ == "__main__":
    main()