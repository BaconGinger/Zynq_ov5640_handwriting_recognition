#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <string>

#include "nn_inference.h"

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
//   1. 当前 HLS 的输出目录：
//        ../../../test_result/
//
//   2. 如果引用其他 HLS 的输出文件：
//        ../../../../对应的hls名字/test_result/
// ============================================================

// 当前 nn_inference 的输出目录
static const std::string TB_RESULT_DIR =
    "../../../test_result";

// 引用 roi_resize_mnist_ip 的输出目录
static const std::string TB_ROI_RESIZE_RESULT_DIR =
    "../../../../roi_resize_mnist/test_result";

// 引用 projection_bbox_ip 的输出目录
static const std::string TB_PROJECTION_BBOX_RESULT_DIR =
    "../../../../projection_bbox/test_result";

// ROI float 文件格式
// 文件来自 roi_resize_mnist_ip/test_result/
static const std::string TB_ROI_FLOAT_FILE_FMT =
    TB_ROI_RESIZE_RESULT_DIR + "/roi_mnist_%02d_float.txt";

// bbox 文件来自 projection_bbox_ip/test_result/
// 用于确定 ROI 个数
static const std::string TB_BOX_FILE =
    TB_PROJECTION_BBOX_RESULT_DIR + "/projection_boxes.txt";

// NN 输出文件
static const std::string OUT_RESULTS_TXT =
    TB_RESULT_DIR + "/results.txt";

static const std::string OUT_DETAIL_TXT =
    TB_RESULT_DIR + "/nn_detail.txt";


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
// 根据 printf 风格格式生成文件路径
// ============================================================

static std::string make_roi_float_path(int index)
{
    char buf[512];

    std::snprintf(
        buf,
        sizeof(buf),
        TB_ROI_FLOAT_FILE_FMT.c_str(),
        index
    );

    return std::string(buf);
}


// ============================================================
// 读取 bbox 数量
//
// 支持格式：
//     num_boxes = 6
//     0 10 12 27 39 210
//     1 45 12 62 39 145
//
// 返回：
//   >=0 : 读取到的 bbox 数量
//   -1  : 文件打不开
// ============================================================

static int read_num_boxes_from_txt(const std::string &path)
{
    std::ifstream ifs(path.c_str());

    if (!ifs.is_open()) {
        std::cerr << "[WARN] Failed to open boxes file: "
                  << path
                  << std::endl;
        return -1;
    }

    int count = 0;
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
        int x0;
        int y0;
        int x1;
        int y1;
        int area;

        if (ss >> idx >> x0 >> y0 >> x1 >> y1) {
            count++;
        }
    }

    ifs.close();

    return count;
}


// ============================================================
// 读取 roi_mnist_xx_float.txt
//
// 文件格式：
//   28 行，每行 28 个 float
//
// 总计读取 784 个 float。
// ============================================================

static bool read_roi_float_txt(
    const std::string &path,
    float input_pix[784]
) {
    std::ifstream ifs(path.c_str());

    if (!ifs.is_open()) {
        return false;
    }

    for (int i = 0; i < 784; i++) {
        if (!(ifs >> input_pix[i])) {
            std::cerr << "[ERROR] Invalid ROI float file: "
                      << path
                      << std::endl;
            ifs.close();
            return false;
        }
    }

    ifs.close();

    return true;
}


// ============================================================
// 保存输入向量摘要，便于调试
// ============================================================

static void write_input_summary(
    std::ofstream &ofs,
    const float input_pix[784]
) {
    float min_v = input_pix[0];
    float max_v = input_pix[0];
    double sum = 0.0;

    for (int i = 0; i < 784; i++) {
        float v = input_pix[i];

        if (v < min_v) {
            min_v = v;
        }

        if (v > max_v) {
            max_v = v;
        }

        sum += v;
    }

    double mean = sum / 784.0;

    ofs << " input_min=" << min_v
        << " input_max=" << max_v
        << " input_mean=" << mean;
}


// ============================================================
// main
// ============================================================

int main()
{
    create_result_dir();

    std::cout << "============================================" << std::endl;
    std::cout << " NN Inference Simulation From ROI Output" << std::endl;
    std::cout << "============================================" << std::endl;

    std::cout << "Current HLS result dir      : "
              << TB_RESULT_DIR
              << std::endl;

    std::cout << "ROI resize result dir       : "
              << TB_ROI_RESIZE_RESULT_DIR
              << std::endl;

    std::cout << "Projection bbox result dir  : "
              << TB_PROJECTION_BBOX_RESULT_DIR
              << std::endl;

    std::cout << "Boxes file                  : "
              << TB_BOX_FILE
              << std::endl;

    std::cout << "ROI float file format       : "
              << TB_ROI_FLOAT_FILE_FMT
              << std::endl;

    // ------------------------------------------------------------
    // 读取 bbox 数量
    // ------------------------------------------------------------

    int num_boxes = read_num_boxes_from_txt(TB_BOX_FILE);

    if (num_boxes <= 0) {
        std::cerr << "[WARN] Could not get num_boxes from boxes file."
                  << std::endl;

        std::cerr << "[WARN] Will read ROI files sequentially until missing."
                  << std::endl;

        // 如果无法读取 projection_boxes.txt，则顺序尝试：
        // roi_mnist_00_float.txt, roi_mnist_01_float.txt, ...
        num_boxes = 9999;
    } else {
        std::cout << "Number of boxes from txt     : "
                  << num_boxes
                  << std::endl;
    }

    // ------------------------------------------------------------
    // 打开输出文件
    // ------------------------------------------------------------

    std::ofstream result_ofs(OUT_RESULTS_TXT.c_str());
    std::ofstream detail_ofs(OUT_DETAIL_TXT.c_str());

    if (!result_ofs.is_open()) {
        std::cerr << "[ERROR] Failed to open: "
                  << OUT_RESULTS_TXT
                  << std::endl;
        return 1;
    }

    if (!detail_ofs.is_open()) {
        std::cerr << "[ERROR] Failed to open: "
                  << OUT_DETAIL_TXT
                  << std::endl;
        result_ofs.close();
        return 1;
    }

    result_ofs << "# one prediction per line\n";

    detail_ofs << "NN inference detail\n";
    detail_ofs << "roi_float_dir = "
               << TB_ROI_RESIZE_RESULT_DIR
               << "\n";

    detail_ofs << "boxes_file = "
               << TB_BOX_FILE
               << "\n\n";

    // ------------------------------------------------------------
    // 逐个 ROI 调用 nn_inference_hw
    // ------------------------------------------------------------

    int processed = 0;

    for (int i = 0; i < num_boxes; i++) {
        std::string roi_path = make_roi_float_path(i);

        float input_pix[784];
        int result_mem[1];

        std::memset(input_pix, 0, sizeof(input_pix));

        result_mem[0] = -1;

        if (!read_roi_float_txt(roi_path, input_pix)) {
            if (num_boxes == 9999) {
                // 顺序读取模式下，遇到第一个不存在的 ROI 文件就结束
                std::cout << "Stop reading ROI files at index "
                          << i
                          << ", missing file: "
                          << roi_path
                          << std::endl;
                break;
            }

            std::cerr << "[ERROR] Failed to read ROI file: "
                      << roi_path
                      << std::endl;

            result_ofs.close();
            detail_ofs.close();

            return 1;
        }

        std::cout << "Read ROI "
                  << i
                  << ": "
                  << roi_path
                  << std::endl;

        // ========================================================
        // nn_inference_hw 直接接收 roi_resize_mnist_ip 输出的
        // float[784]。
        //
        // 当前 HLS 接口：
        //   nn_inference_hw(input_pix, result_mem)
        //
        // 该接口只输出分类结果 result，不输出 logits。
        // ========================================================

        nn_inference_hw(
            input_pix,
            result_mem
        );

        int pred = result_mem[0];

        result_ofs << pred << "\n";

        detail_ofs << "roi "
                   << i
                   << " pred "
                   << pred;

        write_input_summary(
            detail_ofs,
            input_pix
        );

        detail_ofs << " file "
                   << roi_path
                   << "\n";

        std::cout << "ROI "
                  << i
                  << " prediction = "
                  << pred
                  << std::endl;

        processed++;
    }

    result_ofs.close();
    detail_ofs.close();

    // ------------------------------------------------------------
    // 检查是否成功处理
    // ------------------------------------------------------------

    if (processed <= 0) {
        std::cerr << "[ERROR] No ROI float files processed."
                  << std::endl;

        std::cerr << "[ERROR] Please run roi_resize_mnist_ip test first."
                  << std::endl;

        std::cerr << "[ERROR] Expected ROI files under:"
                  << std::endl;

        std::cerr << "        "
                  << TB_ROI_RESIZE_RESULT_DIR
                  << std::endl;

        return 1;
    }

    std::cout << std::endl;
    std::cout << "[OK] Processed ROI count: "
              << processed
              << std::endl;

    std::cout << "[OK] Saved results      : "
              << OUT_RESULTS_TXT
              << std::endl;

    std::cout << "[OK] Saved detail       : "
              << OUT_DETAIL_TXT
              << std::endl;

    std::cout << "============================================" << std::endl;
    std::cout << " NN Inference Simulation Finished" << std::endl;
    std::cout << "============================================" << std::endl;

    return 0;
}
