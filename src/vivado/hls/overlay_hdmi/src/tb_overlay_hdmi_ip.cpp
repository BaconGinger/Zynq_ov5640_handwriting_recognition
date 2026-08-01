#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <cstdio>
#include <string>

#include <opencv2/opencv.hpp>

#include "overlay_hdmi_ip.h"

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif


// ============================================================
// 路径配置
//
// 统一约定：
//   1. 原始输入图片：
//        ../../../../../test_figs/test.jpg
//
//   2. 当前 HLS 的输出目录：
//        ../../../test_result/
//
//   3. 如果引用其他 HLS 的输出文件：
//        ../../../../对应的hls名字/test_result/
// ============================================================

// 使用缩放输入图片
static const std::string TB_INPUT_IMAGE =
    "../../../../Image_Binarization/test_result/test_input_resized.jpg";

// 当前 overlay_hdmi_ip 的输出目录
static const std::string TB_RESULT_DIR =
    "../../../test_result";

// projection_bbox_ip 输出目录
static const std::string TB_PROJECTION_BBOX_RESULT_DIR =
    "../../../../projection_bbox/test_result";

// nn_inference 输出目录
static const std::string TB_NN_INFERENCE_RESULT_DIR =
    "../../../../FCNN_MNIST/test_result";

// projection_bbox_ip 输出的 bbox 文件
static const std::string TB_BOX_FILE =
    TB_PROJECTION_BBOX_RESULT_DIR + "/projection_boxes.txt";

// nn_inference 输出的预测结果文件
static const std::string TB_RESULT_FILE =
    TB_NN_INFERENCE_RESULT_DIR + "/results.txt";

// overlay 输出图片
static const std::string OUTPUT_IMAGE =
    TB_RESULT_DIR + "/overlay_hdmi_output.png";


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
// 读取 bbox
//
// 支持格式：
//     num_boxes = 6
//     0 10 12 27 39 210
//     1 45 12 62 39 145
// ============================================================

static bool read_boxes_txt(
    const std::string &path,
    std::vector<OverlayBBox> &boxes
) {
    std::ifstream ifs(path.c_str());

    if (!ifs.is_open()) {
        std::cerr << "[WARN] Failed to open boxes file: "
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

        if (line.find("num_boxes") != std::string::npos) {
            continue;
        }

        std::stringstream ss(line);

        int idx;
        OverlayBBox b;

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
// 读取 nn_inference 生成的 results.txt
//
// 支持格式：
//     3
//     1
//     4
//
// 同时兼容带注释的格式：
//     # one prediction per line
//     3
//     1
// ============================================================

static bool read_results_txt(
    const std::string &path,
    std::vector<int> &results
) {
    std::ifstream ifs(path.c_str());

    if (!ifs.is_open()) {
        std::cerr << "[WARN] Failed to open result file: "
                  << path
                  << std::endl;
        return false;
    }

    results.clear();

    std::string line;

    while (std::getline(ifs, line)) {
        if (line.empty()) {
            continue;
        }

        // 跳过注释行
        if (line[0] == '#') {
            continue;
        }

        std::stringstream ss(line);

        int v;

        if (ss >> v) {
            results.push_back(v);
        }
    }

    ifs.close();

    return !results.empty();
}


// ============================================================
// cv::Mat BGR -> AXI Stream RGB
// ============================================================

static void mat_to_axi_stream(
    const cv::Mat &img,
    hls::stream<ap_axiu<24,1,1,1> > &stream
) {
    int rows = img.rows;
    int cols = img.cols;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            cv::Vec3b pix = img.at<cv::Vec3b>(y, x);

            unsigned char b = pix[0];
            unsigned char g = pix[1];
            unsigned char r = pix[2];

            ap_axiu<24,1,1,1> axi_pix;

            axi_pix.data.range(23, 16) = r;
            axi_pix.data.range(15, 8)  = g;
            axi_pix.data.range(7, 0)   = b;

            axi_pix.keep = -1;
            axi_pix.strb = -1;
            axi_pix.user = (x == 0 && y == 0) ? 1 : 0;
            axi_pix.last = (x == cols - 1) ? 1 : 0;
            axi_pix.id   = 0;
            axi_pix.dest = 0;

            stream.write(axi_pix);
        }
    }
}


// ============================================================
// AXI Stream RGB -> cv::Mat BGR
// ============================================================

static void axi_stream_to_mat(
    hls::stream<ap_axiu<24,1,1,1> > &stream,
    cv::Mat &img
) {
    int rows = img.rows;
    int cols = img.cols;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            ap_axiu<24,1,1,1> axi_pix = stream.read();

            unsigned char r = axi_pix.data.range(23, 16);
            unsigned char g = axi_pix.data.range(15, 8);
            unsigned char b = axi_pix.data.range(7, 0);

            img.at<cv::Vec3b>(y, x) = cv::Vec3b(b, g, r);
        }
    }
}


// ============================================================
// main
// ============================================================

int main()
{
    create_result_dir();

    std::cout << "============================================" << std::endl;
    std::cout << " Overlay HDMI IP Simulation" << std::endl;
    std::cout << "============================================" << std::endl;

    std::cout << "Input image                 : "
              << TB_INPUT_IMAGE
              << std::endl;

    std::cout << "Projection bbox result dir  : "
              << TB_PROJECTION_BBOX_RESULT_DIR
              << std::endl;

    std::cout << "NN inference result dir     : "
              << TB_NN_INFERENCE_RESULT_DIR
              << std::endl;

    std::cout << "Boxes file                  : "
              << TB_BOX_FILE
              << std::endl;

    std::cout << "Results file                : "
              << TB_RESULT_FILE
              << std::endl;

    std::cout << "Output image                : "
              << OUTPUT_IMAGE
              << std::endl;

    // ------------------------------------------------------------
    // 读取原始输入图片
    // ------------------------------------------------------------

    cv::Mat img = cv::imread(TB_INPUT_IMAGE, cv::IMREAD_COLOR);

    if (img.empty()) {
        std::cerr << "[ERROR] Failed to load image: "
                  << TB_INPUT_IMAGE
                  << std::endl;
        return 1;
    }

    int rows = img.rows;
    int cols = img.cols;

    std::cout << "Image size                  : "
              << cols
              << " x "
              << rows
              << std::endl;

    if (rows > MAX_OVERLAY_HEIGHT || cols > MAX_OVERLAY_WIDTH) {
        std::cerr << "[ERROR] Image too large." << std::endl;
        std::cerr << "MAX_OVERLAY_WIDTH  = "
                  << MAX_OVERLAY_WIDTH
                  << std::endl;
        std::cerr << "MAX_OVERLAY_HEIGHT = "
                  << MAX_OVERLAY_HEIGHT
                  << std::endl;
        return 1;
    }

    // ------------------------------------------------------------
    // 读取 projection_bbox_ip 输出的 bbox
    // ------------------------------------------------------------

    std::vector<OverlayBBox> box_vec;

    if (!read_boxes_txt(TB_BOX_FILE, box_vec)) {
        std::cerr << "[WARN] No boxes loaded." << std::endl;
        std::cerr << "[WARN] Please run projection_bbox_ip test first." << std::endl;
        std::cerr << "[WARN] Use one fallback test box." << std::endl;

        OverlayBBox b;

        b.x0 = cols / 4;
        b.y0 = rows / 4;
        b.x1 = cols / 2;
        b.y1 = rows / 2;
        b.area = 0;

        box_vec.push_back(b);
    }

    if (box_vec.size() > MAX_OVERLAY_BOXES) {
        std::cout << "[WARN] Too many boxes. Truncate to "
                  << MAX_OVERLAY_BOXES
                  << std::endl;

        box_vec.resize(MAX_OVERLAY_BOXES);
    }

    // ------------------------------------------------------------
    // 读取 nn_inference 输出的预测结果
    // ------------------------------------------------------------

    std::vector<int> result_vec;

    if (!read_results_txt(TB_RESULT_FILE, result_vec)) {
        std::cerr << "[WARN] No NN results loaded." << std::endl;
        std::cerr << "[WARN] Please run nn_inference test first." << std::endl;
        std::cerr << "[WARN] Use fallback result i % 10." << std::endl;

        result_vec.resize(box_vec.size());

        for (size_t i = 0; i < box_vec.size(); i++) {
            result_vec[i] = static_cast<int>(i % 10);
        }
    }

    if (result_vec.size() < box_vec.size()) {
        size_t old_size = result_vec.size();

        std::cerr << "[WARN] Results less than boxes. Fill remaining with -1."
                  << std::endl;

        result_vec.resize(box_vec.size());

        for (size_t i = old_size; i < result_vec.size(); i++) {
            result_vec[i] = -1;
        }
    }

    // ------------------------------------------------------------
    // 准备 IP 输入数组
    // ------------------------------------------------------------

    OverlayBBox boxes[MAX_OVERLAY_BOXES];
    int results[MAX_OVERLAY_BOXES];

    std::memset(boxes, 0, sizeof(boxes));

    for (int i = 0; i < MAX_OVERLAY_BOXES; i++) {
        results[i] = -1;
    }

    int num_boxes = static_cast<int>(box_vec.size());

    for (int i = 0; i < num_boxes; i++) {
        boxes[i] = box_vec[i];
        results[i] = result_vec[i];

        std::cout
            << "Box " << i
            << ": x0=" << boxes[i].x0
            << ", y0=" << boxes[i].y0
            << ", x1=" << boxes[i].x1
            << ", y1=" << boxes[i].y1
            << ", area=" << boxes[i].area
            << ", result=" << results[i]
            << std::endl;
    }

    // ------------------------------------------------------------
    // 调用 overlay_hdmi_ip
    // ------------------------------------------------------------

    hls::stream<ap_axiu<24,1,1,1> > src_axi;
    hls::stream<ap_axiu<24,1,1,1> > dst_axi;

    mat_to_axi_stream(img, src_axi);

    int enable = 1;
    int box_thickness = 2;
    int digit_scale = 3;

    std::cout << "Overlay parameters:" << std::endl;
    std::cout << "  enable        = " << enable << std::endl;
    std::cout << "  box_thickness = " << box_thickness << std::endl;
    std::cout << "  digit_scale   = " << digit_scale << std::endl;

    overlay_hdmi_ip(
        src_axi,
        dst_axi,

        reinterpret_cast<volatile OverlayBBox *>(boxes),
        reinterpret_cast<volatile int *>(results),

        num_boxes,
        rows,
        cols,

        enable,
        box_thickness,
        digit_scale
    );

    cv::Mat out_img(rows, cols, CV_8UC3);

    axi_stream_to_mat(dst_axi, out_img);

    if (!cv::imwrite(OUTPUT_IMAGE, out_img)) {
        std::cerr << "[ERROR] Failed to save image: "
                  << OUTPUT_IMAGE
                  << std::endl;
        return 1;
    }

    std::cout << "[OK] Saved overlay image: "
              << OUTPUT_IMAGE
              << std::endl;

    std::cout << "============================================" << std::endl;
    std::cout << " Overlay HDMI IP Simulation Finished" << std::endl;
    std::cout << "============================================" << std::endl;

    return 0;
}
