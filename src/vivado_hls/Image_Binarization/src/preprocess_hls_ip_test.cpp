#include "preprocess_hls_ip.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

// ============================================================
// 路径与测试参数，集中放在文件开头，便于修改
// ============================================================

// 输入图片，相对 C-simulation 运行目录
static const std::string TB_INPUT_IMAGE_PATH =
    "../../../../../test_figs/test.jpg";

// 输出结果
static const std::string TB_HLS_OUTPUT_IMAGE_PATH =
    "../../../test_result/test_processed_hls.jpg";

static const std::string TB_GOLDEN_OUTPUT_IMAGE_PATH =
    "../../../test_result/test_processed_golden_int.jpg";

static const std::string TB_DIFF_OUTPUT_IMAGE_PATH =
    "../../../test_result/test_diff.jpg";

static const std::string TB_RESIZED_INPUT_IMAGE_PATH =
    "../../../test_result/test_input_resized.jpg";

// 测试参数
static const int TB_WIN_SIZE = 25;
static const int TB_MEAN_C   = 30;

// 是否自动缩放过大的图片
static const bool TB_AUTO_RESIZE_IF_TOO_LARGE = true;

// 如果不想等比例缩放，而是直接裁剪左上角区域，可以改成 true
static const bool TB_CROP_INSTEAD_OF_RESIZE = false;


// ============================================================
// 与 preprocess_hls_ip.cpp 中一致的 win_size 归一化逻辑
// ============================================================
static int normalize_win_size(int win_size) {
    int win = win_size;

    if (win < 3) {
        win = 3;
    }

    if (win > 51) {
        win = 51;
    }

    if ((win & 1) == 0) {
        win++;

        if (win > 51) {
            win = 51;
        }
    }

    return win;
}


// ============================================================
// mean_c 归一化逻辑
// ============================================================
static int normalize_mean_c(int mean_c) {
    int c_val = mean_c;

    if (c_val < 0) {
        c_val = 0;
    }

    if (c_val > 255) {
        c_val = 255;
    }

    return c_val;
}


// ============================================================
// 与 HLS 中 read_rgb_to_gray() 一致的灰度转换
//
// HLS 中 DDR 顺序：
//   src[idx + 0] = B
//   src[idx + 1] = G
//   src[idx + 2] = R
// ============================================================
static unsigned char tb_bgr_to_gray(
    unsigned char b,
    unsigned char g,
    unsigned char r
) {
    int gray = 77  * (int)r
             + 150 * (int)g
             + 29  * (int)b;

    gray = (gray + 128) >> 8;

    if (gray < 0) {
        gray = 0;
    }

    if (gray > 255) {
        gray = 255;
    }

    return (unsigned char)gray;
}


// ============================================================
// 自动处理过大的测试图片
//
// 只修改测试文件，不修改 HLS 内核。
// 如果图片尺寸超过 PREPROCESS_MAX_WIDTH/PREPROCESS_MAX_HEIGHT：
//   1. 默认等比例缩放到最大尺寸以内；
//   2. 也可以通过 TB_CROP_INSTEAD_OF_RESIZE 改成裁剪。
// ============================================================
static bool prepare_test_image(
    const cv::Mat &input_img,
    cv::Mat &test_img
) {
    if (input_img.empty()) {
        return false;
    }

    int src_rows = input_img.rows;
    int src_cols = input_img.cols;

    if (src_rows <= PREPROCESS_MAX_HEIGHT &&
        src_cols <= PREPROCESS_MAX_WIDTH) {
        test_img = input_img.clone();
        return true;
    }

    std::cout << "Input image is too large: "
              << src_cols << " x " << src_rows
              << std::endl;

    std::cout << "Max supported size: "
              << PREPROCESS_MAX_WIDTH << " x "
              << PREPROCESS_MAX_HEIGHT
              << std::endl;

    if (!TB_AUTO_RESIZE_IF_TOO_LARGE) {
        std::cerr << "ERROR: Image too large and auto resize is disabled."
                  << std::endl;
        return false;
    }

    if (TB_CROP_INSTEAD_OF_RESIZE) {
        int crop_cols = std::min(src_cols, PREPROCESS_MAX_WIDTH);
        int crop_rows = std::min(src_rows, PREPROCESS_MAX_HEIGHT);

        cv::Rect roi(0, 0, crop_cols, crop_rows);
        test_img = input_img(roi).clone();

        std::cout << "Image cropped to: "
                  << test_img.cols << " x "
                  << test_img.rows
                  << std::endl;
    } else {
        double scale_w =
            (double)PREPROCESS_MAX_WIDTH / (double)src_cols;

        double scale_h =
            (double)PREPROCESS_MAX_HEIGHT / (double)src_rows;

        double scale = std::min(scale_w, scale_h);

        int new_cols = (int)(src_cols * scale);
        int new_rows = (int)(src_rows * scale);

        if (new_cols < 1) {
            new_cols = 1;
        }

        if (new_rows < 1) {
            new_rows = 1;
        }

        cv::resize(
            input_img,
            test_img,
            cv::Size(new_cols, new_rows),
            0,
            0,
            cv::INTER_AREA
        );

        std::cout << "Image resized to: "
                  << test_img.cols << " x "
                  << test_img.rows
                  << std::endl;
    }

    return true;
}


// ============================================================
// CPU Golden
//
// 与当前 preprocess_hls_ip.cpp 的算法保持一致：
//   1. BGR 转 Gray
//   2. 使用前向窗口：
//        y ~ y + win - 1
//        x ~ x + win - 1
//   3. 边界处窗口自动裁剪
//   4. 判断条件：
//        gray * count < sum - mean_c * count
// ============================================================
static void generate_golden_integer_mean(
    const cv::Mat &src_bgr,
    cv::Mat &golden_bin,
    int win_size,
    int mean_c
) {
    int rows = src_bgr.rows;
    int cols = src_bgr.cols;

    int win = normalize_win_size(win_size);
    int c_val = normalize_mean_c(mean_c);

    std::vector<unsigned char> gray(rows * cols);

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            cv::Vec3b bgr = src_bgr.at<cv::Vec3b>(y, x);

            unsigned char b = bgr[0];
            unsigned char g = bgr[1];
            unsigned char r = bgr[2];

            gray[y * cols + x] = tb_bgr_to_gray(b, g, r);
        }
    }

    // 积分图
    std::vector<int> integ((rows + 1) * (cols + 1), 0);

    for (int y = 1; y <= rows; y++) {
        int row_acc = 0;

        for (int x = 1; x <= cols; x++) {
            row_acc += gray[(y - 1) * cols + (x - 1)];

            integ[y * (cols + 1) + x] =
                integ[(y - 1) * (cols + 1) + x] + row_acc;
        }
    }

    golden_bin = cv::Mat(rows, cols, CV_8UC1);

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {

            int y0 = y;
            int x0 = x;
            int y1 = y + win;
            int x1 = x + win;

            if (y1 > rows) {
                y1 = rows;
            }

            if (x1 > cols) {
                x1 = cols;
            }

            int sum =
                integ[y1 * (cols + 1) + x1]
              - integ[y0 * (cols + 1) + x1]
              - integ[y1 * (cols + 1) + x0]
              + integ[y0 * (cols + 1) + x0];

            int count = (y1 - y0) * (x1 - x0);

            unsigned char g = gray[y * cols + x];

            unsigned char bin = 0;

            if (count > 0) {
                int lhs = ((int)g) * count;
                int rhs = sum - c_val * count;

                if (rhs > 0 && lhs < rhs) {
                    bin = 255;
                } else {
                    bin = 0;
                }
            }

            golden_bin.at<unsigned char>(y, x) = bin;
        }
    }
}


// ============================================================
// Main Testbench
// ============================================================
int main() {
    std::cout << "Starting preprocess_hls_ip C-simulation test..."
              << std::endl;

    std::cout << "Input image path : "
              << TB_INPUT_IMAGE_PATH
              << std::endl;

    std::cout << "win_size         : "
              << TB_WIN_SIZE
              << std::endl;

    std::cout << "mean_c           : "
              << TB_MEAN_C
              << std::endl;

    // ------------------------------------------------------------
    // 读取原始图片
    // ------------------------------------------------------------
    cv::Mat input_img_cv =
        cv::imread(TB_INPUT_IMAGE_PATH, cv::IMREAD_COLOR);

    if (input_img_cv.empty()) {
        std::cerr << "ERROR: Could not open input image: "
                  << TB_INPUT_IMAGE_PATH
                  << std::endl;
        return 1;
    }

    std::cout << "Original image size: "
              << input_img_cv.cols
              << " x "
              << input_img_cv.rows
              << std::endl;

    // ------------------------------------------------------------
    // 如果图片过大，在测试文件中缩放或裁剪
    // ------------------------------------------------------------
    cv::Mat src_img_cv;

    if (!prepare_test_image(input_img_cv, src_img_cv)) {
        return 1;
    }

    int rows = src_img_cv.rows;
    int cols = src_img_cv.cols;

    if (rows > PREPROCESS_MAX_HEIGHT ||
        cols > PREPROCESS_MAX_WIDTH) {
        std::cerr << "ERROR: Prepared image is still too large."
                  << std::endl;
        return 1;
    }

    std::cout << "Test image size: "
              << cols
              << " x "
              << rows
              << std::endl;

    // 保存实际送入 HLS 测试的图片，方便查看
    if (!cv::imwrite(TB_RESIZED_INPUT_IMAGE_PATH, src_img_cv)) {
        std::cerr << "WARNING: Failed to save resized input image: "
                  << TB_RESIZED_INPUT_IMAGE_PATH
                  << std::endl;
    } else {
        std::cout << "Prepared input image saved to: "
                  << TB_RESIZED_INPUT_IMAGE_PATH
                  << std::endl;
    }

    // ------------------------------------------------------------
    // OpenCV BGR 图像转换为模拟 DDR buffer
    //
    // preprocess_hls_ip.cpp 中读取方式：
    //   b = src_img[idx + 0]
    //   g = src_img[idx + 1]
    //   r = src_img[idx + 2]
    // ------------------------------------------------------------
    std::vector<unsigned char> src_buf(rows * cols * 3);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            cv::Vec3b bgr = src_img_cv.at<cv::Vec3b>(r, c);

            int idx = (r * cols + c) * 3;

            src_buf[idx + 0] = bgr[0];
            src_buf[idx + 1] = bgr[1];
            src_buf[idx + 2] = bgr[2];
        }
    }

    // ------------------------------------------------------------
    // 输出 buffer
    // ------------------------------------------------------------
    std::vector<unsigned char> bin_img(rows * cols, 0);

    // ------------------------------------------------------------
    // 调用 HLS IP
    // ------------------------------------------------------------
    std::cout << "Calling preprocess_hls_ip..."
              << std::endl;

    preprocess_hls_ip(
        src_buf.data(),
        bin_img.data(),
        rows,
        cols,
        TB_WIN_SIZE,
        TB_MEAN_C
    );

    std::cout << "preprocess_hls_ip finished."
              << std::endl;

    // ------------------------------------------------------------
    // HLS 输出转 OpenCV Mat
    // ------------------------------------------------------------
    cv::Mat hls_img_cv(rows, cols, CV_8UC1);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            hls_img_cv.at<unsigned char>(r, c) =
                bin_img[r * cols + c];
        }
    }

    if (!cv::imwrite(TB_HLS_OUTPUT_IMAGE_PATH, hls_img_cv)) {
        std::cerr << "WARNING: Failed to save HLS output image: "
                  << TB_HLS_OUTPUT_IMAGE_PATH
                  << std::endl;
    } else {
        std::cout << "HLS output image saved to: "
                  << TB_HLS_OUTPUT_IMAGE_PATH
                  << std::endl;
    }

    // ------------------------------------------------------------
    // 生成 Golden
    // ------------------------------------------------------------
    cv::Mat golden_img_cv;

    std::cout << "Generating CPU golden..."
              << std::endl;

    generate_golden_integer_mean(
        src_img_cv,
        golden_img_cv,
        TB_WIN_SIZE,
        TB_MEAN_C
    );

    if (!cv::imwrite(TB_GOLDEN_OUTPUT_IMAGE_PATH, golden_img_cv)) {
        std::cerr << "WARNING: Failed to save golden image: "
                  << TB_GOLDEN_OUTPUT_IMAGE_PATH
                  << std::endl;
    } else {
        std::cout << "Golden image saved to: "
                  << TB_GOLDEN_OUTPUT_IMAGE_PATH
                  << std::endl;
    }

    // ------------------------------------------------------------
    // 对比 HLS 和 Golden
    // ------------------------------------------------------------
    int diff_pixels = 0;
    double mse = 0.0;

    cv::Mat diff_img_cv(rows, cols, CV_8UC1);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            unsigned char hls_val =
                hls_img_cv.at<unsigned char>(r, c);

            unsigned char ref_val =
                golden_img_cv.at<unsigned char>(r, c);

            if (hls_val != ref_val) {
                diff_pixels++;
                diff_img_cv.at<unsigned char>(r, c) = 255;
            } else {
                diff_img_cv.at<unsigned char>(r, c) = 0;
            }

            double d = (double)hls_val - (double)ref_val;
            mse += d * d;
        }
    }

    mse /= (double)(rows * cols);

    if (!cv::imwrite(TB_DIFF_OUTPUT_IMAGE_PATH, diff_img_cv)) {
        std::cerr << "WARNING: Failed to save diff image: "
                  << TB_DIFF_OUTPUT_IMAGE_PATH
                  << std::endl;
    } else {
        std::cout << "Diff image saved to: "
                  << TB_DIFF_OUTPUT_IMAGE_PATH
                  << std::endl;
    }

    std::cout << "Compare with golden:"
              << std::endl;

    std::cout << "  diff_pixels = "
              << diff_pixels
              << std::endl;

    std::cout << "  mse         = "
              << mse
              << std::endl;

    if (diff_pixels == 0) {
        std::cout << "[PASS] HLS output matches golden."
                  << std::endl;
    } else {
        std::cout << "[WARN] HLS output differs from golden."
                  << std::endl;
    }

    std::cout << "C-simulation completed."
              << std::endl;

    return 0;
}
