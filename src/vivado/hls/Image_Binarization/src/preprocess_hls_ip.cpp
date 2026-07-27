#include "preprocess_hls_ip.h"

#define PREPROCESS_MAX_WIN 51

static int clamp_int(int v, int lo, int hi) {
#pragma HLS INLINE

    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

// 从 DDR 读取 RGB888 并转灰度
static unsigned char read_rgb_to_gray(
    volatile unsigned char *src_img,
    int y,
    int x,
    int cols
) {
#pragma HLS INLINE

    int idx = (y * cols + x) * 3;

    /*
     * 这里保持你原来的读取顺序：
     * DDR 中实际顺序如果是 B,G,R，就没问题。
     * 如果你确认是 R,G,B，则这里需要调换。
     */
    unsigned char b = src_img[idx];
    unsigned char g = src_img[idx + 1];
    unsigned char r = src_img[idx + 2];

    int gray = 77 * (int)r + 150 * (int)g + 29 * (int)b;
    gray = (gray + 128) >> 8;

    if (gray < 0) gray = 0;
    if (gray > 255) gray = 255;

    return (unsigned char)gray;
}

void preprocess_hls_ip(
    volatile unsigned char *src_img,
    volatile unsigned char *bin_img,

    int rows,
    int cols,

    int win_size,
    int mean_c
) {
#pragma HLS INTERFACE m_axi port=src_img offset=slave bundle=gmem depth=2764800
#pragma HLS INTERFACE m_axi port=bin_img offset=slave bundle=gmem depth=921600

#pragma HLS INTERFACE s_axilite port=src_img  bundle=control
#pragma HLS INTERFACE s_axilite port=bin_img  bundle=control
#pragma HLS INTERFACE s_axilite port=rows     bundle=control
#pragma HLS INTERFACE s_axilite port=cols     bundle=control
#pragma HLS INTERFACE s_axilite port=win_size bundle=control
#pragma HLS INTERFACE s_axilite port=mean_c   bundle=control
#pragma HLS INTERFACE s_axilite port=return   bundle=control

    static unsigned char line_ring[PREPROCESS_MAX_WIN][PREPROCESS_MAX_WIDTH];
#pragma HLS RESOURCE variable=line_ring core=RAM_S2P_BRAM

    static int col_sum[PREPROCESS_MAX_WIDTH];
#pragma HLS RESOURCE variable=col_sum core=RAM_S2P_BRAM

    if (rows <= 0 || cols <= 0) {
        return;
    }

    if (rows > PREPROCESS_MAX_HEIGHT) {
        rows = PREPROCESS_MAX_HEIGHT;
    }

    if (cols > PREPROCESS_MAX_WIDTH) {
        cols = PREPROCESS_MAX_WIDTH;
    }

    /*
     * 运行时窗口大小限制。
     *
     * 原来 W = 25。
     * 现在 main.c 可以传 win_size。
     *
     * 建议 win_size 用奇数，例如：
     * 15, 21, 25, 31, 41, 51
     */
    int win = win_size;

    if (win < 3) {
        win = 3;
    }

    if (win > PREPROCESS_MAX_WIN) {
        win = PREPROCESS_MAX_WIN;
    }

    /*
     * 强制转成奇数。
     * 例如 main 传 24，这里会变成 25。
     */
    if ((win & 1) == 0) {
        win++;
        if (win > PREPROCESS_MAX_WIN) {
            win = PREPROCESS_MAX_WIN;
        }
    }

    int c_val = mean_c;

    if (c_val < 0) {
        c_val = 0;
    }

    if (c_val > 255) {
        c_val = 255;
    }

    // ============================================================
    // 初始化 col_sum
    // ============================================================
Init_Col_Sum:
    for (int c = 0; c < PREPROCESS_MAX_WIDTH; c++) {
#pragma HLS LOOP_TRIPCOUNT min=1280 max=1280
#pragma HLS PIPELINE II=1

        col_sum[c] = 0;
    }

    // ============================================================
    // 预加载前 win 行
    // ============================================================
    int preload_rows = rows < win ? rows : win;
    int v_count = preload_rows;

Preload_Rows:
    for (int r = 0; r < PREPROCESS_MAX_WIN; r++) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=51

        if (r < preload_rows) {

        Preload_Cols:
            for (int c = 0; c < PREPROCESS_MAX_WIDTH; c++) {
#pragma HLS LOOP_TRIPCOUNT min=1280 max=1280
#pragma HLS PIPELINE II=1

                if (c < cols) {
                    unsigned char gray = read_rgb_to_gray(src_img, r, c, cols);
                    line_ring[r][c] = gray;
                    col_sum[c] += (int)gray;
                }
            }
        }
    }

    /*
     * ring_idx 表示当前输出行 r_out 对应 line_ring 中的哪一行。
     *
     * 原来写法是：
     *     ring_idx = r_out % W;
     *
     * 现在 win 是运行时参数，避免每行做除法取模，
     * 改成手动循环递增。
     */
    int ring_idx = 0;

    // ============================================================
    // 主处理循环
    // ============================================================
Process_Rows:
    for (int r_out = 0; r_out < PREPROCESS_MAX_HEIGHT; r_out++) {
#pragma HLS LOOP_TRIPCOUNT min=720 max=720

        if (r_out < rows) {

            // ----------------------------------------------------
            // 初始化当前行的水平窗口
            // ----------------------------------------------------
            int h_win_sum = 0;
            int h_count = 0;

        Init_HWin:
            for (int k = 0; k < PREPROCESS_MAX_WIN; k++) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=51
#pragma HLS PIPELINE II=1

                if (k < win && k < cols) {
                    h_win_sum += col_sum[k];
                    h_count++;
                }
            }

            // ----------------------------------------------------
            // 当前输出行二值化
            // ----------------------------------------------------
        Process_Cols:
            for (int c_out = 0; c_out < PREPROCESS_MAX_WIDTH; c_out++) {
#pragma HLS LOOP_TRIPCOUNT min=1280 max=1280
#pragma HLS PIPELINE II=1

                if (c_out < cols) {
                    unsigned char gray = line_ring[ring_idx][c_out];

                    int win_count = v_count * h_count;
                    unsigned char bin = 0;

                    if (win_count > 0) {
                        int lhs = ((int)gray) * win_count;
                        int rhs = h_win_sum - c_val * win_count;

                        /*
                         * 如果当前像素比局部均值暗 mean_c 以上，则认为是前景。
                         */
                        if (rhs > 0 && lhs < rhs) {
                            bin = 255;
                        } else {
                            bin = 0;
                        }
                    }

                    bin_img[r_out * cols + c_out] = bin;

                    // --------------------------------------------
                    // 水平窗口右移
                    // --------------------------------------------
                    h_win_sum -= col_sum[c_out];
                    h_count--;

                    int c_new = c_out + win;

                    if (c_new < cols) {
                        h_win_sum += col_sum[c_new];
                        h_count++;
                    }
                }
            }

            // ----------------------------------------------------
            // 更新垂直窗口
            // 当前 ring_idx 对应的旧行移出，
            // 新行 r_out + win 放入。
            // ----------------------------------------------------
            int r_new = r_out + win;

        Update_Cols:
            for (int c = 0; c < PREPROCESS_MAX_WIDTH; c++) {
#pragma HLS LOOP_TRIPCOUNT min=1280 max=1280
#pragma HLS PIPELINE II=1

                if (c < cols) {
                    unsigned char old_gray = line_ring[ring_idx][c];
                    col_sum[c] -= (int)old_gray;

                    if (r_new < rows) {
                        unsigned char new_gray = read_rgb_to_gray(src_img, r_new, c, cols);
                        line_ring[ring_idx][c] = new_gray;
                        col_sum[c] += (int)new_gray;
                    }
                }
            }

            if (r_new >= rows) {
                v_count--;
                if (v_count < 0) {
                    v_count = 0;
                }
            }

            // ----------------------------------------------------
            // ring_idx 循环递增
            // ----------------------------------------------------
            ring_idx++;

            if (ring_idx >= win) {
                ring_idx = 0;
            }
        }
    }
}
