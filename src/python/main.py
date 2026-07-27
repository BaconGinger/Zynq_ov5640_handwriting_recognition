"""
One-click Multi-Digit Recognition Pipeline

新版流程：
  可选 0. 调用 bp_train.py 训练 / 测试 / 导出权重
  可选 0.5. 调用 preprocess.py 对输入图片进行预处理（去阴影、增强对比度、二值化等）
  1. 调用 digit_detector.py 检测 digit.png 中多个数字
  2. 调用 vivado_csim_runner.py 运行 Vivado HLS C Simulation
  3. test_bench.cpp 内部生成最终标注图，不再调用 watermark_overlay.py

不训练 + 预处理 + 检测 + HLS:
  python main.py

带预处理但不训练：
  python main.py --preprocess

训练后再预处理+检测+HLS：
  python main.py --train --preprocess --epochs 50 --lr 0.01

只跑检测 + HLS（跳过预处理）：
  python main.py --no_preprocess

只训练并导出权重：
  python main.py --train_only --epochs 50 --lr 0.01

只测试已有权重：
  python main.py --train --train_test

只导出初始化权重：
  python main.py --train --train_export_only

跳过检测：
  python main.py --skip_detect

跳过 HLS：
  python main.py --skip_hls
"""

import argparse
import datetime
import json
import locale
import os
import platform
import subprocess
import sys
from pathlib import Path


# ============================================================
# Default paths
# ============================================================

DEFAULT_BASE_DIR = "./"
# DEFAULT_INPUT_IMAGE = Path(r"C:\Develop\python\python_study\lab10\test_figs\digit.png")
DEFAULT_INPUT_IMAGE = Path(r"C:\Develop\python\python_study\lab10\test_figs\test.jpg")
DEFAULT_HLS_DIR = "./hls_1"
DEFAULT_SRC_DIR = "./hls_1/src"

DEFAULT_TCL_FILE = "./hls_1/solution1/csim_only.tcl"

DEFAULT_VIVADO_SETTINGS = Path(
    r"C:\Develop\Vivado2018.3\Vivado\2018.3\settings64.bat"
)

DEFAULT_VIVADO_HLS = Path(
    r"C:\Develop\Vivado2018.3\Vivado\2018.3\bin\vivado_hls"
)

DEFAULT_RESULTS_TXT = "./hls_1/output/csim_results.txt"

# 注意：这个图片由 test_bench.cpp 生成
DEFAULT_OUTPUT_IMAGE = "./hls_1/output/csim_watermark_result.jpg"


# ============================================================
# Utility
# ============================================================

def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def step(title):
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def path_exists_or_error(path, name):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def command_to_string(cmd):
    if isinstance(cmd, (list, tuple)):
        return subprocess.list2cmdline([str(x) for x in cmd])
    return str(cmd)


def run_command(cmd, cwd=None, log_file=None, shell=False):
    cwd = Path(cwd) if cwd else None
    cmd_display = command_to_string(cmd)

    print()
    print(f"[CMD] {cmd_display}")
    if cwd:
        print(f"[CWD] {cwd}")

    log_fp = None

    try:
        if log_file is not None:
            ensure_dir(Path(log_file).parent)
            log_fp = open(log_file, "w", encoding="utf-8", errors="replace")
            log_fp.write(f"Time: {now_str()}\n")
            log_fp.write(f"CWD : {cwd}\n")
            log_fp.write(f"CMD : {cmd_display}\n")
            log_fp.write("-" * 72 + "\n")

        encoding = locale.getpreferredencoding(False) or "utf-8"

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            shell=shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding=encoding,
            errors="replace",
        )

        assert proc.stdout is not None

        for line in proc.stdout:
            print(line, end="")
            if log_fp:
                log_fp.write(line)

        proc.wait()

        if log_fp:
            log_fp.write("-" * 72 + "\n")
            log_fp.write(f"Return code: {proc.returncode}\n")

        return proc.returncode

    finally:
        if log_fp:
            log_fp.close()


def write_json(path, data):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def open_file(path):
    path = Path(path)

    try:
        if platform.system().lower().startswith("win"):
            os.startfile(str(path))
            return True

        if platform.system().lower() == "darwin":
            subprocess.Popen(["open", str(path)])
            return True

        subprocess.Popen(["xdg-open", str(path)])
        return True

    except Exception as e:
        print(f"  [WARN] Could not auto-open file: {e}")
        return False


def load_regions_json(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"digit_regions.json not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    regions = data.get("regions", [])
    num_digits = data.get("num_digits", len(regions))

    try:
        num_digits = int(num_digits)
    except Exception:
        num_digits = len(regions)

    return data, num_digits


def count_result_lines(path):
    path = Path(path)

    if not path.exists():
        return 0

    count = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue

            parts = s.split()
            if len(parts) >= 3:
                count += 1

    return count


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="One-click multi-digit recognition pipeline."
    )

    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_IMAGE))

    parser.add_argument("--hls_dir", type=str, default=None)
    parser.add_argument("--src_dir", type=str, default=None)

    parser.add_argument("--tcl", type=str, default=None)
    parser.add_argument("--vivado_settings", type=str, default=str(DEFAULT_VIVADO_SETTINGS))
    parser.add_argument("--vivado_hls", type=str, default=str(DEFAULT_VIVADO_HLS))

    parser.add_argument("--detector_script", type=str, default=None)
    parser.add_argument("--vivado_runner_script", type=str, default=None)
    parser.add_argument("--bp_train_script", type=str, default=None)
    parser.add_argument("--preprocess_script", type=str, default=None)

    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)

    # 流程控制
    parser.add_argument("--train", action="store_true",
                        help="Run bp_train.py before detection/HLS.")
    parser.add_argument("--train_only", action="store_true",
                        help="Only run bp_train.py, then exit.")
    parser.add_argument("--preprocess", action="store_true",
                        help="Run preprocess.py to clean input image before detection.")
    parser.add_argument("--no_preprocess", action="store_true",
                        help="Skip preprocessing even if preprocess.py exists.")
    parser.add_argument("--skip_detect", action="store_true")
    parser.add_argument("--skip_hls", action="store_true")
    parser.add_argument("--no_open", action="store_true")
    parser.add_argument("--allow_zero", action="store_true")
    parser.add_argument("--strict_hls_return", action="store_true")
    parser.add_argument("--no_clean_hls", action="store_true")

    # bp_train.py 参数
    parser.add_argument("--train_test", action="store_true",
                        help="Forward to bp_train.py --test.")
    parser.add_argument("--train_export_only", action="store_true",
                        help="Forward to bp_train.py --export_only.")
    parser.add_argument("--cpu", action="store_true",
                        help="Forward to bp_train.py --cpu.")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--lr_decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--train_labels_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--test_labels_path", type=str, default=None)
    parser.add_argument("--weights_out_dir", type=str, default=None)

    # digit_detector.py 参数
    parser.add_argument("--fit_size", type=int, default=None)
    parser.add_argument("--close_radius", type=int, default=None)
    parser.add_argument("--open_radius", type=int, default=None)
    parser.add_argument("--merge_gap", type=int, default=None)
    parser.add_argument("--min_area", type=int, default=None)
    parser.add_argument("--otsu_scale", type=float, default=None,
                        help="[digit_detector] Use threshold = max(threshold_min, otsu * otsu_scale).")

    # preprocess.py 参数
    parser.add_argument("--pre_threshold_method", type=str, default='auto',
                        choices=['auto', 'sauvola', 'otsu', 'adaptive'],
                        help="Binarization strategy for preprocessing. 'auto' tries several and picks best.")
    parser.add_argument("--pre_sauvola_k", type=float, default=0.25,
                        help="Sauvola sensitivity (0.1~0.5, smaller = thicker strokes).")
    parser.add_argument("--pre_sauvola_window", type=int, default=31,
                        help="Sauvola window size (odd, 15~51).")
    parser.add_argument("--pre_dilate_strength", type=int, default=1,
                        help="Stroke dilation iterations (0~2).")
    parser.add_argument("--pre_min_noise_area", type=int, default=30,
                        help="Min connected component area to keep.")
    parser.add_argument("--pre_no_preview", action="store_true",
                        help="Disable matplotlib preview after preprocessing.")
    # 保留原有的 skip 开关，但映射含义不变
    parser.add_argument("--pre_skip_shadow", action="store_true",
                        help="Skip illumination normalization (formerly shadow removal).")
    parser.add_argument("--pre_skip_contrast", action="store_true",
                        help="Skip automatic gamma correction.")
    parser.add_argument("--pre_skip_denoise", action="store_true",
                        help="Skip edge-preserving denoising.")
    # preprocess.py 参数 (新增部分)
    parser.add_argument("--pre_median_blur_ksize", type=int, default=3,
                        help="Median blur kernel size before main processing (odd, >=3). Set to 0 or 1 to skip.")
    parser.add_argument("--pre_close_morph_kernel", type=int, default=3,
                        help="Kernel size for morphological closing in cleanup (odd, >=1).")
    parser.add_argument("--pre_connectivity", type=int, default=8, choices=[4, 8],
                        help="Connectivity for connected components in cleanup (4 or 8). 8 is generally better for digits.")
    args = parser.parse_args()

    if args.train_only:
        args.train = True

    base_dir = Path(args.base_dir).resolve()
    input_image = Path(args.input).resolve()

    hls_dir = Path(args.hls_dir).resolve() if args.hls_dir else base_dir / "hls_1"
    src_dir = Path(args.src_dir).resolve() if args.src_dir else hls_dir / "src"

    detector_out_dir = src_dir / "digit_detector_out"
    weights_out_dir = (
        Path(args.weights_out_dir).resolve()
        if args.weights_out_dir
        else src_dir / "weights_export"
    )

    tcl_file = Path(args.tcl).resolve() if args.tcl else hls_dir / "solution1" / "csim_only.tcl"

    detector_script = (
        Path(args.detector_script).resolve()
        if args.detector_script
        else base_dir / "digit_detector.py"
    )

    vivado_runner_script = (
        Path(args.vivado_runner_script).resolve()
        if args.vivado_runner_script
        else base_dir / "vivado_csim_runner.py"
    )

    bp_train_script = (
        Path(args.bp_train_script).resolve()
        if args.bp_train_script
        else base_dir / "bp_train.py"
    )

    preprocess_script = (
        Path(args.preprocess_script).resolve()
        if args.preprocess_script
        else base_dir / "preprocess.py"
    )

    vivado_settings = Path(args.vivado_settings).resolve()
    vivado_hls = Path(args.vivado_hls).resolve()

    output_image = (
        Path(args.output).resolve()
        if args.output
        else hls_dir / "output" / "csim_watermark_result.jpg"
    )

    log_dir = (
        Path(args.log_dir).resolve()
        if args.log_dir
        else hls_dir / "output" / "logs"
    )

    regions_json = detector_out_dir / "digit_regions.json"
    results_txt = hls_dir / "output" / "csim_results.txt"
    csim_dir = hls_dir / "solution1" / "csim"
    summary_json = hls_dir / "output" / "pipeline_summary.json"

    # 预处理输出路径：在输入图片同目录生成 *_preprocessed.* 文件
    preprocessed_image = input_image.parent / f"{input_image.stem}_preprocessed{input_image.suffix}"

    ensure_dir(src_dir)
    ensure_dir(detector_out_dir)
    ensure_dir(weights_out_dir)
    ensure_dir(output_image.parent)
    ensure_dir(log_dir)

    summary = {
        "start_time": now_str(),
        "base_dir": str(base_dir),
        "input_image": str(input_image),
        "preprocessed_image": str(preprocessed_image),
        "hls_dir": str(hls_dir),
        "src_dir": str(src_dir),
        "detector_out_dir": str(detector_out_dir),
        "weights_out_dir": str(weights_out_dir),
        "regions_json": str(regions_json),
        "results_txt": str(results_txt),
        "output_image": str(output_image),
        "steps": {},
    }

    print("=" * 72)
    print("One-click Multi-Digit Recognition Pipeline")
    print("=" * 72)
    print(f"Time                 : {now_str()}")
    print(f"Python               : {sys.executable}")
    print(f"Base dir             : {base_dir}")
    print(f"Input image          : {input_image}")
    print(f"Preprocessed image   : {preprocessed_image}")
    print(f"HLS dir              : {hls_dir}")
    print(f"Src dir              : {src_dir}")
    print(f"Detector out dir     : {detector_out_dir}")
    print(f"Weights out dir      : {weights_out_dir}")
    print(f"TCL file             : {tcl_file}")
    print(f"Vivado settings      : {vivado_settings}")
    print(f"Vivado HLS           : {vivado_hls}")
    print(f"Detector script      : {detector_script}")
    print(f"Vivado runner script : {vivado_runner_script}")
    print(f"BP train script      : {bp_train_script}")
    print(f"Preprocess script    : {preprocess_script}")
    print(f"Output image         : {output_image}")
    print(f"Log dir              : {log_dir}")

    try:
        # ------------------------------------------------------------
        # Pre-check
        # ------------------------------------------------------------
        step("Pre-check")

        path_exists_or_error(base_dir, "BASE_DIR")
        path_exists_or_error(input_image, "Input image")
        path_exists_or_error(src_dir, "SRC_DIR")

        if args.train:
            path_exists_or_error(bp_train_script, "bp_train.py")

        if not args.skip_detect and not args.train_only:
            path_exists_or_error(detector_script, "digit_detector.py")

        if not args.skip_hls and not args.train_only:
            path_exists_or_error(vivado_runner_script, "vivado_csim_runner.py")
            path_exists_or_error(tcl_file, "csim_only.tcl")
            path_exists_or_error(vivado_settings, "Vivado settings64.bat")
            path_exists_or_error(vivado_hls, "vivado_hls")

        # 检查预处理脚本是否存在（如果用户要求预处理）
        if args.preprocess and not args.no_preprocess:
            if not preprocess_script.exists():
                print(f"  [WARN] Preprocess script not found: {preprocess_script}")
                print("         Will skip preprocessing.")
                args.no_preprocess = True

        print("  Pre-check passed.")

        # ------------------------------------------------------------
        # Optional Step 0: Training
        # ------------------------------------------------------------
        if args.train:
            step("Optional Step 0: Running bp_train.py")

            train_cmd = [
                sys.executable,
                str(bp_train_script),
                "--out_dir",
                str(weights_out_dir),
            ]

            if args.train_test:
                train_cmd.append("--test")

            if args.train_export_only:
                train_cmd.append("--export_only")

            if args.cpu:
                train_cmd.append("--cpu")

            if args.epochs is not None:
                train_cmd += ["--epochs", str(args.epochs)]

            if args.lr is not None:
                train_cmd += ["--lr", str(args.lr)]

            if args.batch_size is not None:
                train_cmd += ["--batch_size", str(args.batch_size)]

            if args.eval_batch_size is not None:
                train_cmd += ["--eval_batch_size", str(args.eval_batch_size)]

            if args.momentum is not None:
                train_cmd += ["--momentum", str(args.momentum)]

            if args.lr_decay is not None:
                train_cmd += ["--lr_decay", str(args.lr_decay)]

            if args.seed is not None:
                train_cmd += ["--seed", str(args.seed)]

            if args.train_path is not None:
                train_cmd += ["--train_path", str(args.train_path)]

            if args.train_labels_path is not None:
                train_cmd += ["--train_labels_path", str(args.train_labels_path)]

            if args.test_path is not None:
                train_cmd += ["--test_path", str(args.test_path)]

            if args.test_labels_path is not None:
                train_cmd += ["--test_labels_path", str(args.test_labels_path)]

            ret = run_command(
                train_cmd,
                cwd=base_dir,
                log_file=log_dir / "00_bp_train.log",
                shell=False,
            )

            summary["steps"]["train"] = {
                "returncode": ret,
                "log": str(log_dir / "00_bp_train.log"),
                "weights_out_dir": str(weights_out_dir),
            }

            if ret != 0:
                raise RuntimeError("bp_train.py failed.")

            if args.train_only:
                step("DONE")

                summary["end_time"] = now_str()
                summary["status"] = "ok"
                summary["mode"] = "train_only"

                write_json(summary_json, summary)

                print(f"  Training finished.")
                print(f"  Weights out dir : {weights_out_dir}")
                print(f"  Summary JSON    : {summary_json}")
                print(f"  Logs            : {log_dir}")

                return 0
        else:
            summary["steps"]["train"] = {
                "skipped": True,
            }

        # ------------------------------------------------------------
        # Optional Step 0.5: Preprocess input image
        # ------------------------------------------------------------
        if args.preprocess and not args.no_preprocess and not args.train_only:
            step("Optional Step 0.5: Preprocessing input image")

            # 先检查 preprocess.py 中是否有 main 函数可以调用
            # 我们通过 subprocess 调用 preprocess.py 作为独立脚本
            # 但 preprocess.py 的 __main__ 里有写死的路径，
            # 所以我们这里直接 import 并调用函数，更干净

            # 方法：将 preprocess.py 所在目录加入 sys.path，然后 import 函数
            preprocess_dir = str(preprocess_script.parent)
            if preprocess_dir not in sys.path:
                sys.path.insert(0, preprocess_dir)

            try:
                # 动态导入 preprocess 模块
                import importlib.util
                spec = importlib.util.spec_from_file_location("preprocess_module", str(preprocess_script))
                preprocess_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(preprocess_mod)

                # 调用新版预处理函数
                print(f"  Preprocessing: {input_image} -> {preprocessed_image}")
                # ... (前面的参数不变) ...
                result_img, _ = preprocess_mod.preprocess_handwritten_image(
                    input_path=str(input_image),
                    output_path=str(preprocessed_image),
                    strategy=args.pre_threshold_method,
                    median_blur_ksize=args.pre_median_blur_ksize,  # 新参数
                    remove_illumination=not args.pre_skip_shadow,
                    auto_gamma=not args.pre_skip_contrast,
                    denoise=not args.pre_skip_denoise,
                    sauvola_k=args.pre_sauvola_k,
                    sauvola_window=args.pre_sauvola_window,
                    dilate_strength=args.pre_dilate_strength,
                    min_noise_area=args.pre_min_noise_area,
                    close_morph_kernel=args.pre_close_morph_kernel,  # 新参数
                    connectivity=args.pre_connectivity,  # 新参数
                    show_preview=not args.pre_no_preview,
                    verbose=True,
                )
                print(f"  ✅ Preprocessing completed: {preprocessed_image}")

                summary["steps"]["preprocess"] = {
                    "input": str(input_image),
                    "output": str(preprocessed_image),
                    "strategy": args.pre_threshold_method,
                    "sauvola_k": args.pre_sauvola_k,
                    "sauvola_window": args.pre_sauvola_window,
                    "dilate_strength": args.pre_dilate_strength,
                    "min_noise_area": args.pre_min_noise_area,
                    "illumination_normalization": not args.pre_skip_shadow,
                    "auto_gamma": not args.pre_skip_contrast,
                    "denoise": not args.pre_skip_denoise,
                    "status": "ok",
                }

                # 后续检测使用预处理后的图片
                input_image_for_detect = preprocessed_image

            except Exception as e:
                print(f"  [WARN] Preprocessing failed: {e}")
                print("         Will use original input image for detection.")
                input_image_for_detect = input_image
                summary["steps"]["preprocess"] = {
                    "input": str(input_image),
                    "error": str(e),
                    "status": "failed",
                    "fallback_to_original": True,
                }

        else:
            # 不使用预处理，直接用原始图片
            input_image_for_detect = input_image
            summary["steps"]["preprocess"] = {
                "skipped": True,
                "reason": "not requested" if not args.preprocess else "explicitly skipped by --no_preprocess",
            }

        # ------------------------------------------------------------
        # Step 1: Detect digits
        # ------------------------------------------------------------
        if not args.skip_detect:
            step("Step 1/2: Detecting digits")

            detect_cmd = [
                sys.executable,
                str(detector_script),
                "--input",
                str(input_image_for_detect),  # 使用预处理后的图片（或原图）
                "--out_dir",
                str(detector_out_dir),
            ]

            if args.fit_size is not None:
                detect_cmd += ["--fit_size", str(args.fit_size)]

            if args.close_radius is not None:
                detect_cmd += ["--close_radius", str(args.close_radius)]

            if args.open_radius is not None:
                detect_cmd += ["--open_radius", str(args.open_radius)]

            if args.merge_gap is not None:
                detect_cmd += ["--merge_gap", str(args.merge_gap)]

            if args.min_area is not None:
                detect_cmd += ["--min_area", str(args.min_area)]

            ret = run_command(
                detect_cmd,
                cwd=base_dir,
                log_file=log_dir / "01_detect_digits.log",
                shell=False,
            )

            summary["steps"]["detect"] = {
                "returncode": ret,
                "log": str(log_dir / "01_detect_digits.log"),
                "input_used": str(input_image_for_detect),
            }

            if ret != 0:
                raise RuntimeError("Digit detection failed.")

        else:
            step("Step 1/2: Detecting digits")
            print("  Skipped by --skip_detect")

            summary["steps"]["detect"] = {
                "skipped": True,
            }

        # ------------------------------------------------------------
        # Load regions
        # ------------------------------------------------------------
        data, num_digits = load_regions_json(regions_json)

        print()
        print(f"  Detected digits: {num_digits}")

        summary["num_digits"] = num_digits

        if num_digits <= 0 and not args.allow_zero:
            raise RuntimeError(
                "No digit regions detected. "
                "Try parameters such as --fit_size 24 --close_radius 2 --merge_gap 10"
            )

        # ------------------------------------------------------------
        # Step 2: Run Vivado HLS C simulation
        # ------------------------------------------------------------
        if not args.skip_hls and num_digits > 0:
            step(f"Step 2/2: Running Vivado HLS C Simulation ({num_digits} digits)")

            hls_cmd = [
                sys.executable,
                str(vivado_runner_script),
                "--base_dir",
                str(base_dir),
                "--hls_dir",
                str(hls_dir),
                "--tcl",
                str(tcl_file),
                "--vivado_settings",
                str(vivado_settings),
                "--vivado_hls",
                str(vivado_hls),
                "--results",
                str(results_txt),
                "--csim_dir",
                str(csim_dir),
                "--log_dir",
                str(log_dir),
                "--summary",
                str(log_dir / "vivado_csim_summary.json"),
            ]

            if not args.no_clean_hls:
                hls_cmd.append("--clean")

            if args.strict_hls_return:
                hls_cmd.append("--strict_return")

            ret = run_command(
                hls_cmd,
                cwd=base_dir,
                log_file=log_dir / "02_call_vivado_runner.log",
                shell=False,
            )

            result_lines = count_result_lines(results_txt)

            summary["steps"]["hls"] = {
                "returncode": ret,
                "runner_script": str(vivado_runner_script),
                "runner_call_log": str(log_dir / "02_call_vivado_runner.log"),
                "runner_summary": str(log_dir / "vivado_csim_summary.json"),
                "results_exists": results_txt.exists(),
                "result_lines": result_lines,
                "output_image_exists": output_image.exists(),
            }

            if ret != 0:
                raise RuntimeError("Vivado HLS simulation failed.")

            if not results_txt.exists():
                raise RuntimeError("csim_results.txt was not generated.")

            if not output_image.exists():
                print()
                print(f"  [WARN] Expected output image was not generated: {output_image}")
                print("         Please check OUT_IMG in test_bench.cpp.")

            print()
            print("  Vivado HLS C Simulation completed.")
            print(f"  Results file : {results_txt}")
            print(f"  Result lines : {result_lines}")
            print(f"  Output image : {output_image}")

            if result_lines < num_digits:
                print()
                print(
                    f"  [WARN] Result lines ({result_lines}) are fewer than "
                    f"detected digits ({num_digits})."
                )

        elif args.skip_hls:
            step("Step 2/2: Running Vivado HLS C Simulation")
            print("  Skipped by --skip_hls")

            result_lines = count_result_lines(results_txt)

            summary["steps"]["hls"] = {
                "skipped": True,
                "results_exists": results_txt.exists(),
                "result_lines": result_lines,
                "output_image_exists": output_image.exists(),
            }

        else:
            step("Step 2/2: Running Vivado HLS C Simulation")
            print("  Skipped because num_digits is 0.")

            summary["steps"]["hls"] = {
                "skipped": True,
                "reason": "num_digits is 0",
            }

        # ------------------------------------------------------------
        # Done
        # ------------------------------------------------------------
        step("DONE")

        summary["end_time"] = now_str()
        summary["status"] = "ok"

        write_json(summary_json, summary)

        print(f"  Input image        : {input_image}")
        print(f"  Preprocessed image : {preprocessed_image}")
        print(f"  Detected digits    : {num_digits}")
        print(f"  Regions JSON       : {regions_json}")
        print(f"  Results TXT        : {results_txt}")
        print(f"  Output image       : {output_image}")
        print(f"  Summary JSON       : {summary_json}")
        print(f"  Logs               : {log_dir}")

        # 通常 test_bench.cpp 已经在 CSim 结束时打开图片。
        # 这里默认不重复打开，避免弹两次。
        if output_image.exists() and not args.no_open:
            print()
            print("  Note: test_bench.cpp should have opened the saved image already.")
            print("        If it did not, opening it from Python now...")
            open_file(output_image)

        return 0

    except Exception as e:
        print()
        print("=" * 72)
        print("[ERROR] Pipeline failed")
        print("=" * 72)
        print(f"Reason: {e}")

        summary["end_time"] = now_str()
        summary["status"] = "failed"
        summary["error"] = str(e)

        try:
            write_json(summary_json, summary)
            print(f"Summary saved to: {summary_json}")
        except Exception:
            pass

        print()
        print("Useful checks:")
        print(f"  1. Input image exists:          {input_image}")
        print(f"  2. BP train script exists:      {bp_train_script}")
        print(f"  3. Preprocess script exists:    {preprocess_script}")
        print(f"  4. Detector script exists:      {detector_script}")
        print(f"  5. Vivado runner script exists: {vivado_runner_script}")
        print(f"  6. Regions JSON generated:      {regions_json}")
        print(f"  7. Vivado settings exists:      {vivado_settings}")
        print(f"  8. Vivado HLS exists:           {vivado_hls}")
        print(f"  9. TCL file exists:             {tcl_file}")
        print(f" 10. Results generated:           {results_txt}")
        print(f" 11. Output image generated:      {output_image}")
        print(f" 12. Logs directory:              {log_dir}")

        return 1


if __name__ == "__main__":
    sys.exit(main())