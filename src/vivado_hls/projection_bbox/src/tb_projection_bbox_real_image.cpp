#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <string>

#include <opencv2/opencv.hpp>

#include "projection_bbox_ip.h"



// 原始输入图片
static const std::string TB_INPUT_IMAGE_PATH =
    "../../../../../test_figs/test.jpg";

static const std::string TB_PREPROCESS_HLS_OUTPUT_IMAGE_PATH =
    "../../../../Image_Binarization/test_result/test_processed_hls.jpg";


static const bool TB_USE_PREPROCESS_OUTPUT_FIRST = true;

// 输出文件全部放在 ../../../test_result/
static const std::string OUT_BINARY_IMAGE =
    "../../../test_result/projection_binary_for_ip.png";

static const std::string OUT_OVERLAY_IMAGE =
    "../../../test_result/projection_overlay.png";

static const std::string OUT_BOX_TXT =
    "../../../test_result/projection_boxes.txt";

static const std::string OUT_PREPARED_INPUT_IMAGE =
    "../../../test_result/projection_input_used.png";

// 如果输入图片超过 projection_bbox_ip 支持的 MAX_WIDTH/MAX_HEIGHT，
// 只在测试文件中进行缩放，不修改 HLS IP。
static const bool TB_AUTO_RESIZE_IF_TOO_LARGE = true;

// 如果图片过大：
//   false：等比例缩放
//   true ：裁剪左上角
static const bool TB_CROP_INSTEAD_OF_RESIZE = false;


// ============================================================
// 自动选择输入图片
// ============================================================

static cv::Mat load_input_image(std::string &used_path)
{
    cv::Mat img;

    if (TB_USE_PREPROCESS_OUTPUT_FIRST) {
        used_path = TB_PREPROCESS_HLS_OUTPUT_IMAGE_PATH;

        img = cv::imread(used_path, cv::IMREAD_GRAYSCALE);

        if (!img.empty()) {
            std::cout << "Use previous HLS output as input: "
                      << used_path << std::endl;
            return img;
        }

        std::cout << "WARNING: Failed to load previous HLS output: "
                  << used_path << std::endl;

        std::cout << "Fallback to original input image: "
                  << TB_INPUT_IMAGE_PATH << std::endl;
    }

    used_path = TB_INPUT_IMAGE_PATH;
    img = cv::imread(used_path, cv::IMREAD_GRAYSCALE);

    return img;
}


// ============================================================
// 如果图片过大，只在测试文件中缩放或裁剪
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

    if (src_rows <= MAX_HEIGHT && src_cols <= MAX_WIDTH) {
        test_img = input_img.clone();
        return true;
    }

    std::cout << "Input image is too large: "
              << src_cols << " x " << src_rows << std::endl;

    std::cout << "Max supported size: "
              << MAX_WIDTH << " x " << MAX_HEIGHT << std::endl;

    if (!TB_AUTO_RESIZE_IF_TOO_LARGE) {
        std::cerr << "[ERROR] Image too large and auto resize is disabled."
                  << std::endl;
        return false;
    }

    if (TB_CROP_INSTEAD_OF_RESIZE) {
        int crop_cols = std::min(src_cols, MAX_WIDTH);
        int crop_rows = std::min(src_rows, MAX_HEIGHT);

        cv::Rect roi(0, 0, crop_cols, crop_rows);
        test_img = input_img(roi).clone();

        std::cout << "Image cropped to: "
                  << test_img.cols << " x "
                  << test_img.rows << std::endl;
    } else {
        double scale_w = (double)MAX_WIDTH / (double)src_cols;
        double scale_h = (double)MAX_HEIGHT / (double)src_rows;
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
                  << test_img.rows << std::endl;
    }

    return true;
}


// ============================================================
// 自动判断背景亮/暗
//
// true  : 浅色背景，深色文字
// false : 深色背景，浅色文字
// ============================================================

static bool detect_background_is_light(const cv::Mat &gray)
{
    std::vector<unsigned char> border;

    int rows = gray.rows;
    int cols = gray.cols;

    border.reserve(rows * 2 + cols * 2);

    for (int x = 0; x < cols; x++) {
        border.push_back(gray.at<unsigned char>(0, x));
        border.push_back(gray.at<unsigned char>(rows - 1, x));
    }

    for (int y = 0; y < rows; y++) {
        border.push_back(gray.at<unsigned char>(y, 0));
        border.push_back(gray.at<unsigned char>(y, cols - 1));
    }

    std::sort(border.begin(), border.end());

    unsigned char median = border[border.size() / 2];

    return median >= 128;
}


// ============================================================
// 将灰度图转换为 projection_bbox_ip 使用的二值图
//
// projection_bbox_ip 统一要求：
//   ink        = 255
//   background = 0
// ============================================================

static void convert_to_binary_for_projection(
    const cv::Mat &gray,
    std::vector<unsigned char> &bin_img,
    int threshold
) {
    int rows = gray.rows;
    int cols = gray.cols;

    bin_img.resize(rows * cols);

    bool background_is_light = detect_background_is_light(gray);

    std::cout << "Background: "
              << (background_is_light ?
                  "light, black ink expected" :
                  "dark, white ink expected")
              << std::endl;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            unsigned char pix = gray.at<unsigned char>(y, x);

            bool is_ink;

            if (background_is_light) {
                // 白底黑字：灰度越小越可能是 ink
                is_ink = pix < threshold;
            } else {
                // 黑底白字：灰度越大越可能是 ink
                is_ink = pix > threshold;
            }

            bin_img[y * cols + x] = is_ink ? 255 : 0;
        }
    }
}


// ============================================================
// 保存传入 projection_bbox_ip 的二值图
// ============================================================

static void save_binary_image(
    const std::string &path,
    const std::vector<unsigned char> &bin_img,
    int rows,
    int cols
) {
    cv::Mat bin(rows, cols, CV_8UC1);

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            bin.at<unsigned char>(y, x) = bin_img[y * cols + x];
        }
    }

    if (!cv::imwrite(path, bin)) {
        std::cerr << "WARNING: Failed to save binary image: "
                  << path << std::endl;
    }
}


// ============================================================
// 在原图上绘制 bbox
// ============================================================

static void draw_overlay(
    const cv::Mat &gray,
    const BBox boxes[MAX_DIGITS],
    int num_boxes,
    const std::string &path
) {
    cv::Mat overlay;

    if (gray.channels() == 1) {
        cv::cvtColor(gray, overlay, cv::COLOR_GRAY2BGR);
    } else {
        overlay = gray.clone();
    }

    for (int i = 0; i < num_boxes; i++) {
        int x0 = boxes[i].x0;
        int y0 = boxes[i].y0;
        int x1 = boxes[i].x1;
        int y1 = boxes[i].y1;

        cv::rectangle(
            overlay,
            cv::Point(x0, y0),
            cv::Point(x1, y1),
            cv::Scalar(0, 0, 255),
            2
        );

        char text[32];
        std::snprintf(text, sizeof(text), "%d", i);

        int text_y = y0 - 5;

        if (text_y < 15) {
            text_y = y0 + 15;
        }

        cv::putText(
            overlay,
            text,
            cv::Point(x0, text_y),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 0, 255),
            1
        );
    }

    if (!cv::imwrite(path, overlay)) {
        std::cerr << "WARNING: Failed to save overlay image: "
                  << path << std::endl;
    }
}


// ============================================================
// 保存 bbox 到 txt
// ============================================================

static void save_boxes_txt(
    const std::string &path,
    const BBox boxes[MAX_DIGITS],
    int num_boxes
) {
    std::ofstream ofs(path.c_str());

    if (!ofs.is_open()) {
        std::cerr << "WARNING: Failed to open boxes txt: "
                  << path << std::endl;
        return;
    }

    ofs << "num_boxes = " << num_boxes << "\n";

    for (int i = 0; i < num_boxes; i++) {
        ofs << i
            << " "
            << boxes[i].x0 << " "
            << boxes[i].y0 << " "
            << boxes[i].x1 << " "
            << boxes[i].y1 << " "
            << boxes[i].area
            << "\n";
    }

    ofs.close();
}


// ============================================================
// main
// ============================================================

int main()
{
    std::cout << "Starting projection_bbox_ip test..."
              << std::endl;

    std::cout << "Original input image path: "
              << TB_INPUT_IMAGE_PATH
              << std::endl;

    std::cout << "Previous HLS output path : "
              << TB_PREPROCESS_HLS_OUTPUT_IMAGE_PATH
              << std::endl;

    std::cout << "Output directory         : "
              << "../../../test_result/"
              << std::endl;

    // ------------------------------------------------------------
    // 读取输入图片
    // ------------------------------------------------------------

    std::string used_input_path;
    cv::Mat input_img = load_input_image(used_input_path);

    if (input_img.empty()) {
        std::cerr << "[ERROR] Failed to load input image." << std::endl;
        std::cerr << "Tried:" << std::endl;

        if (TB_USE_PREPROCESS_OUTPUT_FIRST) {
            std::cerr << "  " << TB_PREPROCESS_HLS_OUTPUT_IMAGE_PATH
                      << std::endl;
        }

        std::cerr << "  " << TB_INPUT_IMAGE_PATH << std::endl;

        return 1;
    }

    std::cout << "Actually used input image: "
              << used_input_path << std::endl;

    std::cout << "Original image size: "
              << input_img.cols
              << " x "
              << input_img.rows
              << std::endl;

    // ------------------------------------------------------------
    // 如果图片过大，只在测试文件中处理
    // ------------------------------------------------------------

    cv::Mat img;

    if (!prepare_test_image(input_img, img)) {
        return 1;
    }

    int rows = img.rows;
    int cols = img.cols;

    std::cout << "Test image size: "
              << cols
              << " x "
              << rows
              << std::endl;

    if (rows > MAX_HEIGHT || cols > MAX_WIDTH) {
        std::cerr << "[ERROR] Prepared image is still too large."
                  << std::endl;

        std::cerr << "MAX_WIDTH  = " << MAX_WIDTH << std::endl;
        std::cerr << "MAX_HEIGHT = " << MAX_HEIGHT << std::endl;

        return 1;
    }

    if (!cv::imwrite(OUT_PREPARED_INPUT_IMAGE, img)) {
        std::cerr << "WARNING: Failed to save prepared input image: "
                  << OUT_PREPARED_INPUT_IMAGE
                  << std::endl;
    } else {
        std::cout << "Prepared input image saved to: "
                  << OUT_PREPARED_INPUT_IMAGE
                  << std::endl;
    }

    // ------------------------------------------------------------
    // 转换为 projection_bbox_ip 需要的二值格式
    //
    // projection_bbox_ip 要求：
    //   ink        = 255
    //   background = 0
    //
    // 如果输入来自 preprocess_hls_ip，通常已经是：
    //   ink = 255, background = 0
    //
    // 这里仍然做一次统一转换，保证原图输入和 HLS 输出输入都能测试。
    // ------------------------------------------------------------

    std::vector<unsigned char> bin_img;

    int bin_threshold = 128;

    convert_to_binary_for_projection(
        img,
        bin_img,
        bin_threshold
    );

    save_binary_image(
        OUT_BINARY_IMAGE,
        bin_img,
        rows,
        cols
    );

    std::cout << "Saved binary image: "
              << OUT_BINARY_IMAGE
              << std::endl;

    // ------------------------------------------------------------
    // 调用 projection_bbox_ip
    // ------------------------------------------------------------

    BBox boxes[MAX_DIGITS];
    std::memset(boxes, 0, sizeof(boxes));

    int num_boxes = 0;

    // ============================================================
    // 参数设置
    //
    // 小图，例如 180x120：
    //   row_threshold = 2
    //   col_threshold = 2
    //
    // 中等图，例如 640x480：
    //   row_threshold = 5~10
    //   col_threshold = 2~5
    //
    // 大图，例如 1280x720：
    //   row_threshold = 8~20
    //   col_threshold = 3~8
    // ============================================================

    int row_threshold = 5;
    int col_threshold = 2;

    int row_gap = 8;
    int col_gap = 6;

    int min_width = 4;
    int min_height = 8;
    int min_area = 20;


    std::cout << "Parameters:" << std::endl;
    std::cout << "  row_threshold = " << row_threshold << std::endl;
    std::cout << "  col_threshold = " << col_threshold << std::endl;
    std::cout << "  row_gap       = " << row_gap << std::endl;
    std::cout << "  col_gap       = " << col_gap << std::endl;
    std::cout << "  min_width     = " << min_width << std::endl;
    std::cout << "  min_height    = " << min_height << std::endl;
    std::cout << "  min_area      = " << min_area << std::endl;

    std::cout << "Calling projection_bbox_ip..."
              << std::endl;

    projection_bbox_ip(
        reinterpret_cast<volatile unsigned char *>(bin_img.data()),
        boxes,
        &num_boxes,

        rows,
        cols,

        row_threshold,
        col_threshold,
        row_gap,
        col_gap,

        min_width,
        min_height,
        min_area
    );

    std::cout << "projection_bbox_ip finished."
              << std::endl;

    // ------------------------------------------------------------
    // 打印检测结果
    // ------------------------------------------------------------

    std::cout << std::endl;
    std::cout << "Detected boxes: "
              << num_boxes
              << std::endl;

    for (int i = 0; i < num_boxes; i++) {
        int w = boxes[i].x1 - boxes[i].x0 + 1;
        int h = boxes[i].y1 - boxes[i].y0 + 1;

        std::cout
            << "Box " << i
            << ": x0=" << boxes[i].x0
            << ", y0=" << boxes[i].y0
            << ", x1=" << boxes[i].x1
            << ", y1=" << boxes[i].y1
            << ", w=" << w
            << ", h=" << h
            << ", area=" << boxes[i].area
            << std::endl;
    }

    // ------------------------------------------------------------
    // 保存输出
    // ------------------------------------------------------------

    draw_overlay(
        img,
        boxes,
        num_boxes,
        OUT_OVERLAY_IMAGE
    );

    save_boxes_txt(
        OUT_BOX_TXT,
        boxes,
        num_boxes
    );

    std::cout << std::endl;
    std::cout << "[OK] projection_bbox_ip test finished."
              << std::endl;

    std::cout << "  Input used    : "
              << used_input_path
              << std::endl;

    std::cout << "  Prepared input: "
              << OUT_PREPARED_INPUT_IMAGE
              << std::endl;

    std::cout << "  Binary for IP : "
              << OUT_BINARY_IMAGE
              << std::endl;

    std::cout << "  Overlay       : "
              << OUT_OVERLAY_IMAGE
              << std::endl;

    std::cout << "  Boxes txt     : "
              << OUT_BOX_TXT
              << std::endl;

    return 0;
}
