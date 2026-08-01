#include "roi_resize_mnist_ip.h"

static int clamp_int(int v, int lo, int hi) {
#pragma HLS INLINE
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static void swap_int(int &a, int &b) {
#pragma HLS INLINE
    int t = a;
    a = b;
    b = t;
}

void roi_resize_mnist_ip(
    volatile unsigned char *bin_img,
    volatile float *out_pix,

    int rows,
    int cols,

    int x0,
    int y0,
    int x1,
    int y1,

    int fit_size,
    int pad,
    int invert
) {
#pragma HLS INTERFACE m_axi port=bin_img offset=slave bundle=gmem depth=4000000
#pragma HLS INTERFACE m_axi port=out_pix offset=slave bundle=gmem depth=784

#pragma HLS INTERFACE s_axilite port=bin_img bundle=control
#pragma HLS INTERFACE s_axilite port=out_pix bundle=control
#pragma HLS INTERFACE s_axilite port=rows     bundle=control
#pragma HLS INTERFACE s_axilite port=cols     bundle=control
#pragma HLS INTERFACE s_axilite port=x0       bundle=control
#pragma HLS INTERFACE s_axilite port=y0       bundle=control
#pragma HLS INTERFACE s_axilite port=x1       bundle=control
#pragma HLS INTERFACE s_axilite port=y1       bundle=control
#pragma HLS INTERFACE s_axilite port=fit_size bundle=control
#pragma HLS INTERFACE s_axilite port=pad      bundle=control
#pragma HLS INTERFACE s_axilite port=invert   bundle=control
#pragma HLS INTERFACE s_axilite port=return   bundle=control

    // ============================================================
    // 0. 清空输出 28x28
    // ============================================================
Clear_Output:
    for (int i = 0; i < MNIST_PIXELS; i++) {
#pragma HLS PIPELINE II=1
        out_pix[i] = invert ? 1.0f : 0.0f;
    }

    if (rows <= 0 || cols <= 0) {
        return;
    }

    if (rows > ROI_MAX_HEIGHT) {
        rows = ROI_MAX_HEIGHT;
    }

    if (cols > ROI_MAX_WIDTH) {
        cols = ROI_MAX_WIDTH;
    }

    if (x0 > x1) {
        swap_int(x0, x1);
    }

    if (y0 > y1) {
        swap_int(y0, y1);
    }

    x0 = clamp_int(x0, 0, cols - 1);
    x1 = clamp_int(x1, 0, cols - 1);
    y0 = clamp_int(y0, 0, rows - 1);
    y1 = clamp_int(y1, 0, rows - 1);

    if (x1 < x0 || y1 < y0) {
        return;
    }

    if (fit_size < 1) {
        fit_size = 1;
    }

    if (fit_size > MNIST_SIZE) {
        fit_size = MNIST_SIZE;
    }

    if (pad < 0) {
        pad = 0;
    }

    // ============================================================
    // 1. 在输入 bbox 内重新扫描真实 ink，精修 bbox
    //    好处：
    //    1. projection_bbox_ip 框稍微松一点也没关系
    //    2. 手动给大框时，也可以自动收紧到数字
    // ============================================================

    int real_x0 = ROI_MAX_WIDTH;
    int real_y0 = ROI_MAX_HEIGHT;
    int real_x1 = 0;
    int real_y1 = 0;
    int area = 0;

Refine_Y:
    for (int y = y0; y <= y1; y++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=2000

    Refine_X:
        for (int x = x0; x <= x1; x++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=2000
#pragma HLS PIPELINE II=1

            unsigned char pix = bin_img[y * cols + x];

            if (pix != 0) {
                area++;

                if (x < real_x0) real_x0 = x;
                if (x > real_x1) real_x1 = x;
                if (y < real_y0) real_y0 = y;
                if (y > real_y1) real_y1 = y;
            }
        }
    }

    if (area <= 0) {
        return;
    }

    // ============================================================
    // 2. 外扩 pad，并裁剪到图像范围内
    // ============================================================

    int crop_x0 = real_x0 - pad;
    int crop_y0 = real_y0 - pad;
    int crop_x1 = real_x1 + pad;
    int crop_y1 = real_y1 + pad;

    crop_x0 = clamp_int(crop_x0, 0, cols - 1);
    crop_x1 = clamp_int(crop_x1, 0, cols - 1);
    crop_y0 = clamp_int(crop_y0, 0, rows - 1);
    crop_y1 = clamp_int(crop_y1, 0, rows - 1);

    int crop_w = crop_x1 - crop_x0 + 1;
    int crop_h = crop_y1 - crop_y0 + 1;

    if (crop_w <= 0 || crop_h <= 0) {
        return;
    }

    // ============================================================
    // 3. 保持宽高比缩放
    //
    //    最大边缩放到 fit_size。
    //    推荐 fit_size = 18 / 20 / 22。
    // ============================================================

    int new_w;
    int new_h;

    if (crop_w >= crop_h) {
        new_w = fit_size;
        new_h = (crop_h * fit_size + crop_w / 2) / crop_w;
    } else {
        new_h = fit_size;
        new_w = (crop_w * fit_size + crop_h / 2) / crop_h;
    }

    if (new_w < 1) new_w = 1;
    if (new_h < 1) new_h = 1;

    if (new_w > MNIST_SIZE) new_w = MNIST_SIZE;
    if (new_h > MNIST_SIZE) new_h = MNIST_SIZE;

    int off_x = (MNIST_SIZE - new_w) / 2;
    int off_y = (MNIST_SIZE - new_h) / 2;

    // ============================================================
    // 4. 最近邻缩放并写入 28x28
    //
    //    out_pix[dy * 28 + dx]
    //
    //    invert = 0:
    //        ink = 1.0, background = 0.0
    //
    //    invert = 1:
    //        ink = 0.0, background = 1.0
    // ============================================================

Resize_Y:
    for (int oy = 0; oy < MNIST_SIZE; oy++) {
#pragma HLS LOOP_TRIPCOUNT min=28 max=28

    Resize_X:
        for (int ox = 0; ox < MNIST_SIZE; ox++) {
#pragma HLS LOOP_TRIPCOUNT min=28 max=28
#pragma HLS PIPELINE II=1

            float val = invert ? 1.0f : 0.0f;

            if (
                ox >= off_x &&
                ox < off_x + new_w &&
                oy >= off_y &&
                oy < off_y + new_h
            ) {
                int local_x = ox - off_x;
                int local_y = oy - off_y;

                int src_x = crop_x0 + (local_x * crop_w) / new_w;
                int src_y = crop_y0 + (local_y * crop_h) / new_h;

                src_x = clamp_int(src_x, crop_x0, crop_x1);
                src_y = clamp_int(src_y, crop_y0, crop_y1);

                unsigned char pix = bin_img[src_y * cols + src_x];

                bool ink = pix != 0;

                if (invert) {
                    val = ink ? 0.0f : 1.0f;
                } else {
                    val = ink ? 1.0f : 0.0f;
                }
            }

            out_pix[oy * MNIST_SIZE + ox] = val;
        }
    }
}
