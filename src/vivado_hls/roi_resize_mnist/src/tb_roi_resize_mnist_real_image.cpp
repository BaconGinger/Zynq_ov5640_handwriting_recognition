#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <string>

#include <opencv2/opencv.hpp>

#include "roi_resize_mnist_ip.h"

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif


// 原始输入图片
static const std::string TB_ORIGINAL_INPUT_IMAGE =
    "../../../../../test_figs/test.jpg";

// 当前 roi_resize_mnist_ip 的输出目录
static const std::string TB_RESULT_DIR =
    "../../../test_result";

// 引用 projection_bbox_ip 的输出目录
static const std::string TB_PROJECTION_BBOX_RESULT_DIR =
    "../../../../projection_bbox/test_result";

// projection_bbox_ip 输出的二值图
// 该图应满足：ink = 255, background = 0
static const std::string TB_PROJECTION_BINARY_IMAGE =
    TB_PROJECTION_BBOX_RESULT_DIR + "/projection_binary_for_ip.png";

// projection_bbox_ip 输出的 bbox 文件
static const std::string TB_PROJECTION_BOXES_TXT =
    TB_PROJECTION_BBOX_RESULT_DIR + "/projection_boxes.txt";

// 当前测试输出文件
static const std::string OUT_BINARY_IMAGE =
    TB_RESULT_DIR + "/roi_input_binary_for_ip.png";

static const std::string OUT_OVERLAY_IMAGE =
    TB_RESULT_DIR + "/roi_bbox_overlay.png";

static const std::string OUT_PREPARED_INPUT_IMAGE =
    TB_RESULT_DIR + "/roi_input_used.png";

// 是否优先使用 projection_bbox_ip 的二值图作为输入
//
// true:
//   使用 ../../../../projection_bbox_ip/test_result/projection_binary_for_ip.png
//
// false:
//   使用 ../../../../../test_figs/test.jpg
//
// 注意：即使使用原始图片作为输入，bbox 仍然默认从 projection_bbox_ip 的
// projection_boxes.txt 读取。
static const bool TB_USE_PROJECTION_BINARY_FIRST = true;

// 如果输入图像超过 roi_resize_mnist_ip 支持的最大尺寸，
// 只在测试文件中缩放或裁剪，不修改 HLS IP。
static const bool TB_AUTO_RESIZE_IF_TOO_LARGE = true;

// false：等比例缩放
// true ：裁剪左上角
static const bool TB_CROP_INSTEAD_OF_RESIZE = false;


// ============================================================
// Testbench 本地 bbox 结构
// ============================================================

typedef struct {
    int x0;
    int y0;
    int x1;
    int y1;
    int area;
} TBBox;


// ============================================================
// 创建结果目录
// ============================================================

static void create_result_dir()
{
#ifdef _WIN32
    _mkdir(TB_RESULT_DIR.c_str());
#else
    mkdir(TB_RESULT_DIR.c_str(), 0777);
#endif
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

    if (src_rows <= ROI_MAX_HEIGHT && src_cols <= ROI_MAX_WIDTH) {
        test_img = input_img.clone();
        return true;
    }

    std::cout << "Input image is too large: "
              << src_cols << " x " << src_rows
              << std::endl;

    std::cout << "Max supported size: "
              << ROI_MAX_WIDTH << " x "
              << ROI_MAX_HEIGHT
              << std::endl;

    if (!TB_AUTO_RESIZE_IF_TOO_LARGE) {
        std::cerr << "[ERROR] Image too large and auto resize is disabled."
                  << std::endl;
        return false;
    }

    if (TB_CROP_INSTEAD_OF_RESIZE) {
        int crop_cols = std::min(src_cols, ROI_MAX_WIDTH);
        int crop_rows = std::min(src_rows, ROI_MAX_HEIGHT);

        cv::Rect roi(0, 0, crop_cols, crop_rows);
        test_img = input_img(roi).clone();

        std::cout << "Image cropped to: "
                  << test_img.cols << " x "
                  << test_img.rows
                  << std::endl;
    } else {
        double scale_w =
            (double)ROI_MAX_WIDTH / (double)src_cols;

        double scale_h =
            (double)ROI_MAX_HEIGHT / (double)src_rows;

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
// 判断背景亮/暗
//
// true  : 白底黑字
// false : 黑底白字
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
// 转成 roi_resize_mnist_ip 需要的二值图
//
// 输出格式：
//     ink        = 255
//     background = 0
// ============================================================

static void convert_to_binary_for_roi(
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
// 保存输入 IP 的二值图
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
        std::cerr << "[ERROR] Failed to save image: "
                  << path
                  << std::endl;
    }
}


// ============================================================
// 读取 projection_bbox_ip 生成的 projection_boxes.txt
//
// 文件格式兼容：
//     num_boxes = 6
//     0 10 12 27 39 210
//     1 45 12 62 39 145
// ============================================================

static bool read_boxes_txt(
    const std::string &path,
    std::vector<TBBox> &boxes
) {
    std::ifstream ifs(path.c_str());

    if (!ifs.is_open()) {
        std::cerr << "[ERROR] Failed to open boxes file: "
                  << path
                  << std::endl;
        return false;
    }

    boxes.clear();

    std::string line;

    while (std::getline(ifs, line)) {
        if (line.empty()) {
            continue;
        }

        // 跳过 num_boxes = ...
        if (line.find("num_boxes") != std::string::npos) {
            continue;
        }

        std::stringstream ss(line);

        int idx;
        TBBox b;

        b.area = 0;

        if (ss >> idx >> b.x0 >> b.y0 >> b.x1 >> b.y1) {
            if (!(ss >> b.area)) {
                b.area = 0;
            }

            boxes.push_back(b);
        }
    }

    ifs.close();

    return !boxes.empty();
}


// ============================================================
// 保存 28x28 float 为图片
// ============================================================

static void save_mnist_float_as_image(
    const std::string &path_28,
    const std::string &path_big,
    const float out_pix[MNIST_PIXELS]
) {
    cv::Mat img28(MNIST_SIZE, MNIST_SIZE, CV_8UC1);

    for (int y = 0; y < MNIST_SIZE; y++) {
        for (int x = 0; x < MNIST_SIZE; x++) {
            float v = out_pix[y * MNIST_SIZE + x];

            if (v < 0.0f) {
                v = 0.0f;
            }

            if (v > 1.0f) {
                v = 1.0f;
            }

            img28.at<unsigned char>(y, x) =
                static_cast<unsigned char>(v * 255.0f + 0.5f);
        }
    }

    if (!cv::imwrite(path_28, img28)) {
        std::cerr << "[ERROR] Failed to save image: "
                  << path_28
                  << std::endl;
    }

    cv::Mat big;

    cv::resize(
        img28,
        big,
        cv::Size(280, 280),
        0,
        0,
        cv::INTER_NEAREST
    );

    if (!cv::imwrite(path_big, big)) {
        std::cerr << "[ERROR] Failed to save image: "
                  << path_big
                  << std::endl;
    }
}


// ============================================================
// 保存 float[784] 到 txt
// ============================================================

static void save_float_txt(
    const std::string &path,
    const float out_pix[MNIST_PIXELS]
) {
    std::ofstream ofs(path.c_str());

    if (!ofs.is_open()) {
        std::cerr << "[ERROR] Failed to save txt: "
                  << path
                  << std::endl;
        return;
    }

    for (int y = 0; y < MNIST_SIZE; y++) {
        for (int x = 0; x < MNIST_SIZE; x++) {
            ofs << out_pix[y * MNIST_SIZE + x];

            if (x != MNIST_SIZE - 1) {
                ofs << " ";
            }
        }

        ofs << "\n";
    }

    ofs.close();
}


// ============================================================
// 在原图上画 bbox
// ============================================================

static void save_overlay(
    const cv::Mat &gray,
    const std::vector<TBBox> &boxes,
    const std::string &path
) {
    cv::Mat overlay;

    if (gray.channels() == 1) {
        cv::cvtColor(gray, overlay, cv::COLOR_GRAY2BGR);
    } else {
        overlay = gray.clone();
    }

    for (size_t i = 0; i < boxes.size(); i++) {
        const TBBox &b = boxes[i];

        cv::rectangle(
            overlay,
            cv::Point(b.x0, b.y0),
            cv::Point(b.x1, b.y1),
            cv::Scalar(0, 0, 255),
            2
        );

        char text[32];

        std::snprintf(
            text,
            sizeof(text),
            "%d",
            static_cast<int>(i)
        );

        int ty = b.y0 - 5;

        if (ty < 15) {
            ty = b.y0 + 15;
        }

        cv::putText(
            overlay,
            text,
            cv::Point(b.x0, ty),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 0, 255),
            1
        );
    }

    if (!cv::imwrite(path, overlay)) {
        std::cerr << "[ERROR] Failed to save overlay: "
                  << path
                  << std::endl;
    }
}


// ============================================================
// 选择输入图像
// ============================================================

static cv::Mat load_input_image(std::string &used_path)
{
    cv::Mat gray;

    if (TB_USE_PROJECTION_BINARY_FIRST) {
        used_path = TB_PROJECTION_BINARY_IMAGE;

        gray = cv::imread(used_path, cv::IMREAD_GRAYSCALE);

        if (!gray.empty()) {
            std::cout << "Use projection_bbox_ip output image: "
                      << used_path
                      << std::endl;
            return gray;
        }

        std::cout << "WARNING: Failed to load projection_bbox_ip output image: "
                  << used_path
                  << std::endl;

        std::cout << "Fallback to original input image: "
                  << TB_ORIGINAL_INPUT_IMAGE
                  << std::endl;
    }

    used_path = TB_ORIGINAL_INPUT_IMAGE;

    gray = cv::imread(used_path, cv::IMREAD_GRAYSCALE);

    return gray;
}


// ============================================================
// main
// ============================================================

int main()
{
    create_result_dir();

    std::cout << "Starting roi_resize_mnist_ip test..."
              << std::endl;

    std::cout << "Original input image          : "
              << TB_ORIGINAL_INPUT_IMAGE
              << std::endl;

    std::cout << "Projection bbox result dir    : "
              << TB_PROJECTION_BBOX_RESULT_DIR
              << std::endl;

    std::cout << "Projection binary image       : "
              << TB_PROJECTION_BINARY_IMAGE
              << std::endl;

    std::cout << "Projection boxes txt          : "
              << TB_PROJECTION_BOXES_TXT
              << std::endl;

    std::cout << "Current HLS result dir        : "
              << TB_RESULT_DIR
              << std::endl;

    // ------------------------------------------------------------
    // 读取输入图像
    // ------------------------------------------------------------

    std::string used_input_path;

    cv::Mat input_gray = load_input_image(used_input_path);

    if (input_gray.empty()) {
        std::cerr << "[ERROR] Failed to load input image."
                  << std::endl;

        std::cerr << "Tried:" << std::endl;

        if (TB_USE_PROJECTION_BINARY_FIRST) {
            std::cerr << "  "
                      << TB_PROJECTION_BINARY_IMAGE
                      << std::endl;
        }

        std::cerr << "  "
                  << TB_ORIGINAL_INPUT_IMAGE
                  << std::endl;

        return 1;
    }

    std::cout << "Actually used input image     : "
              << used_input_path
              << std::endl;

    std::cout << "Original loaded image size    : "
              << input_gray.cols
              << " x "
              << input_gray.rows
              << std::endl;

    // ------------------------------------------------------------
    // 如果图片过大，只在测试文件中缩放或裁剪
    // ------------------------------------------------------------

    cv::Mat gray;

    if (!prepare_test_image(input_gray, gray)) {
        return 1;
    }

    int rows = gray.rows;
    int cols = gray.cols;

    std::cout << "Test image size               : "
              << cols
              << " x "
              << rows
              << std::endl;

    if (rows > ROI_MAX_HEIGHT || cols > ROI_MAX_WIDTH) {
        std::cerr << "[ERROR] Prepared image is still too large."
                  << std::endl;

        std::cerr << "ROI_MAX_WIDTH  = "
                  << ROI_MAX_WIDTH
                  << std::endl;

        std::cerr << "ROI_MAX_HEIGHT = "
                  << ROI_MAX_HEIGHT
                  << std::endl;

        return 1;
    }

    if (!cv::imwrite(OUT_PREPARED_INPUT_IMAGE, gray)) {
        std::cerr << "[ERROR] Failed to save prepared input image: "
                  << OUT_PREPARED_INPUT_IMAGE
                  << std::endl;
    } else {
        std::cout << "Saved prepared input image    : "
                  << OUT_PREPARED_INPUT_IMAGE
                  << std::endl;
    }

    // ------------------------------------------------------------
    // 转成 IP 输入二值格式
    //
    // roi_resize_mnist_ip 要求：
    //   ink        = 255
    //   background = 0
    //
    // 如果输入来自 projection_bbox_ip 的 projection_binary_for_ip.png，
    // 理论上已经满足该格式。这里仍保留统一转换，方便直接用原图测试。
    // ------------------------------------------------------------

    std::vector<unsigned char> bin_img;

    int threshold = 128;

    convert_to_binary_for_roi(
        gray,
        bin_img,
        threshold
    );

    save_binary_image(
        OUT_BINARY_IMAGE,
        bin_img,
        rows,
        cols
    );

    std::cout << "Saved binary image            : "
              << OUT_BINARY_IMAGE
              << std::endl;

    // ------------------------------------------------------------
    // 从 projection_bbox_ip 的 test_result 中读取 bbox
    // ------------------------------------------------------------

    std::vector<TBBox> boxes;

    if (!read_boxes_txt(TB_PROJECTION_BOXES_TXT, boxes)) {
        std::cerr << "[ERROR] Failed to read boxes from: "
                  << TB_PROJECTION_BOXES_TXT
                  << std::endl;

        std::cerr << "Please run projection_bbox_ip test first."
                  << std::endl;

        return 1;
    }

    std::cout << "Read boxes from               : "
              << TB_PROJECTION_BOXES_TXT
              << std::endl;

    std::cout << "Number of boxes to process    : "
              << boxes.size()
              << std::endl;

    for (size_t i = 0; i < boxes.size(); i++) {
        const TBBox &b = boxes[i];

        std::cout
            << "Box " << i
            << ": x0=" << b.x0
            << ", y0=" << b.y0
            << ", x1=" << b.x1
            << ", y1=" << b.y1
            << ", area=" << b.area
            << std::endl;
    }

    // ------------------------------------------------------------
    // 保存 bbox 叠加图
    // ------------------------------------------------------------

    save_overlay(
        gray,
        boxes,
        OUT_OVERLAY_IMAGE
    );

    std::cout << "Saved bbox overlay            : "
              << OUT_OVERLAY_IMAGE
              << std::endl;

    // ------------------------------------------------------------
    // 调用 roi_resize_mnist_ip
    // ------------------------------------------------------------

    int fit_size = 20;
    int pad = 2;
    int invert = 0;

    std::cout << "ROI parameters:" << std::endl;
    std::cout << "  fit_size = " << fit_size << std::endl;
    std::cout << "  pad      = " << pad << std::endl;
    std::cout << "  invert   = " << invert << std::endl;

    for (size_t i = 0; i < boxes.size(); i++) {
        float out_pix[MNIST_PIXELS];

        std::memset(out_pix, 0, sizeof(out_pix));

        TBBox b = boxes[i];

        roi_resize_mnist_ip(
            reinterpret_cast<volatile unsigned char *>(bin_img.data()),
            reinterpret_cast<volatile float *>(out_pix),

            rows,
            cols,

            b.x0,
            b.y0,
            b.x1,
            b.y1,

            fit_size,
            pad,
            invert
        );

        char path28_buf[512];
        char pathBig_buf[512];
        char pathTxt_buf[512];

        std::snprintf(
            path28_buf,
            sizeof(path28_buf),
            "%s/roi_mnist_%02d_28.png",
            TB_RESULT_DIR.c_str(),
            static_cast<int>(i)
        );

        std::snprintf(
            pathBig_buf,
            sizeof(pathBig_buf),
            "%s/roi_mnist_%02d_280.png",
            TB_RESULT_DIR.c_str(),
            static_cast<int>(i)
        );

        std::snprintf(
            pathTxt_buf,
            sizeof(pathTxt_buf),
            "%s/roi_mnist_%02d_float.txt",
            TB_RESULT_DIR.c_str(),
            static_cast<int>(i)
        );

        std::string path28(path28_buf);
        std::string pathBig(pathBig_buf);
        std::string pathTxt(pathTxt_buf);

        save_mnist_float_as_image(
            path28,
            pathBig,
            out_pix
        );

        save_float_txt(
            pathTxt,
            out_pix
        );

        std::cout << "Saved ROI "
                  << i
                  << ":"
                  << std::endl;

        std::cout << "  "
                  << path28
                  << std::endl;

        std::cout << "  "
                  << pathBig
                  << std::endl;

        std::cout << "  "
                  << pathTxt
                  << std::endl;
    }

    std::cout << std::endl;
    std::cout << "[OK] roi_resize_mnist_ip test finished."
              << std::endl;

    std::cout << "  Input used     : "
              << used_input_path
              << std::endl;

    std::cout << "  Binary for IP  : "
              << OUT_BINARY_IMAGE
              << std::endl;

    std::cout << "  Overlay        : "
              << OUT_OVERLAY_IMAGE
              << std::endl;

    std::cout << "  Result dir     : "
              << TB_RESULT_DIR
              << std::endl;

    return 0;
}
