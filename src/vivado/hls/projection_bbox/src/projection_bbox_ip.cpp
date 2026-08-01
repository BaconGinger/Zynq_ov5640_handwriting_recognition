#include "projection_bbox_ip.h"

static void refine_and_add_box(
    volatile unsigned char *bin_img,
    BBox boxes[MAX_DIGITS],
    int *box_count,

    int rows,
    int cols,

    int row_y0,
    int row_y1,
    int seg_x0,
    int seg_x1,

    int min_width,
    int min_height,
    int min_area
) {
#pragma HLS INLINE off

    if (*box_count >= MAX_DIGITS) {
        return;
    }

    if (row_y0 < 0) row_y0 = 0;
    if (row_y1 >= rows) row_y1 = rows - 1;
    if (seg_x0 < 0) seg_x0 = 0;
    if (seg_x1 >= cols) seg_x1 = cols - 1;

    if (row_y1 < row_y0 || seg_x1 < seg_x0) {
        return;
    }

    int real_x0 = MAX_WIDTH;
    int real_y0 = MAX_HEIGHT;
    int real_x1 = 0;
    int real_y1 = 0;
    int area = 0;

Refine_Y_Loop:
    for (int y = row_y0; y <= row_y1; y++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=2000

    Refine_X_Loop:
        for (int x = seg_x0; x <= seg_x1; x++) {
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

    int width  = real_x1 - real_x0 + 1;
    int height = real_y1 - real_y0 + 1;

    if (width < min_width) {
        return;
    }

    if (height < min_height) {
        return;
    }

    if (area < min_area) {
        return;
    }

    BBox b;
    b.x0 = real_x0;
    b.y0 = real_y0;
    b.x1 = real_x1;
    b.y1 = real_y1;
    b.area = area;

    boxes[*box_count] = b;
    (*box_count)++;
}

void projection_bbox_ip(
    volatile unsigned char *bin_img,
    BBox boxes[MAX_DIGITS],
    volatile int *num_boxes,

    int rows,
    int cols,

    int row_threshold,
    int col_threshold,
    int row_gap,
    int col_gap,

    int min_width,
    int min_height,
    int min_area
) {
#pragma HLS INTERFACE m_axi     port=bin_img   offset=slave bundle=gmem depth=4000000
#pragma HLS INTERFACE m_axi     port=boxes     offset=slave bundle=gmem depth=64
#pragma HLS INTERFACE m_axi     port=num_boxes offset=slave bundle=gmem depth=1
#pragma HLS INTERFACE s_axilite port=bin_img   bundle=control
#pragma HLS INTERFACE s_axilite port=boxes     bundle=control
#pragma HLS INTERFACE s_axilite port=num_boxes bundle=control
#pragma HLS INTERFACE s_axilite port=rows          bundle=control
#pragma HLS INTERFACE s_axilite port=cols          bundle=control
#pragma HLS INTERFACE s_axilite port=row_threshold bundle=control
#pragma HLS INTERFACE s_axilite port=col_threshold bundle=control
#pragma HLS INTERFACE s_axilite port=row_gap       bundle=control
#pragma HLS INTERFACE s_axilite port=col_gap       bundle=control
#pragma HLS INTERFACE s_axilite port=min_width     bundle=control
#pragma HLS INTERFACE s_axilite port=min_height    bundle=control
#pragma HLS INTERFACE s_axilite port=min_area      bundle=control
#pragma HLS INTERFACE s_axilite port=return        bundle=control

    static int row_sum[MAX_HEIGHT];
#pragma HLS RESOURCE variable=row_sum core=RAM_1P_BRAM

    static int col_sum[MAX_WIDTH];
#pragma HLS RESOURCE variable=col_sum core=RAM_1P_BRAM

    int row_start[MAX_ROW_BANDS];
    int row_end[MAX_ROW_BANDS];

#pragma HLS ARRAY_PARTITION variable=row_start cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=row_end   cyclic factor=4 dim=1

    if (rows <= 0 || cols <= 0) {
        *num_boxes = 0;
        return;
    }

    if (rows > MAX_HEIGHT) {
        rows = MAX_HEIGHT;
    }

    if (cols > MAX_WIDTH) {
        cols = MAX_WIDTH;
    }

    int rth = row_threshold;
    int cth = col_threshold;

    if (rth < 1) rth = 1;
    if (cth < 1) cth = 1;

    if (row_gap < 0) row_gap = 0;
    if (col_gap < 0) col_gap = 0;

    if (min_width < 1) min_width = 1;
    if (min_height < 1) min_height = 1;
    if (min_area < 1) min_area = 1;

    int box_count = 0;

    // ============================================================
    // Pass 1: 计算每一行的 ink 像素数
    // ============================================================
Row_Projection_Y:
    for (int y = 0; y < rows; y++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=2000

        int cnt = 0;

    Row_Projection_X:
        for (int x = 0; x < cols; x++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=2000
#pragma HLS PIPELINE II=1

            unsigned char pix = bin_img[y * cols + x];

            if (pix != 0) {
                cnt++;
            }
        }

        row_sum[y] = cnt;
    }

    // ============================================================
    // Pass 2: 根据 row_sum 找多行区域
    // ============================================================
    int row_region_count = 0;

    bool in_row_region = false;
    int current_row_start = 0;
    int last_active_row = 0;
    int current_row_gap = 0;

Find_Row_Bands:
    for (int y = 0; y < rows; y++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=2000
#pragma HLS PIPELINE II=1

        bool active = row_sum[y] >= rth;

        if (active) {
            if (!in_row_region) {
                in_row_region = true;
                current_row_start = y;
            }

            last_active_row = y;
            current_row_gap = 0;
        } else {
            if (in_row_region) {
                current_row_gap++;

                if (current_row_gap > row_gap) {
                    int y0 = current_row_start;
                    int y1 = last_active_row;
                    int height = y1 - y0 + 1;

                    if (height >= min_height && row_region_count < MAX_ROW_BANDS) {
                        row_start[row_region_count] = y0;
                        row_end[row_region_count] = y1;
                        row_region_count++;
                    }

                    in_row_region = false;
                    current_row_gap = 0;
                }
            }
        }
    }

    if (in_row_region) {
        int y0 = current_row_start;
        int y1 = last_active_row;
        int height = y1 - y0 + 1;

        if (height >= min_height && row_region_count < MAX_ROW_BANDS) {
            row_start[row_region_count] = y0;
            row_end[row_region_count] = y1;
            row_region_count++;
        }
    }

    // ============================================================
    // Pass 3: 对每一个行带做列投影，找每个数字
    // ============================================================
Row_Band_Loop:
    for (int rb = 0; rb < row_region_count; rb++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=32

        if (box_count >= MAX_DIGITS) {
            break;
        }

        int y0 = row_start[rb];
        int y1 = row_end[rb];

        // --------------------------------------------------------
        // 清空 col_sum
        // --------------------------------------------------------
    Clear_Col_Sum:
        for (int x = 0; x < cols; x++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=2000
#pragma HLS PIPELINE II=1

            col_sum[x] = 0;
        }

        // --------------------------------------------------------
        // 在当前行带内计算列投影
        // --------------------------------------------------------
    Col_Projection_Y:
        for (int y = y0; y <= y1; y++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=2000

        Col_Projection_X:
            for (int x = 0; x < cols; x++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=2000
#pragma HLS PIPELINE II=1

                unsigned char pix = bin_img[y * cols + x];

                if (pix != 0) {
                    col_sum[x]++;
                }
            }
        }

        // --------------------------------------------------------
        // 根据 col_sum 找当前行带中的数字区间
        // --------------------------------------------------------
        bool in_col_region = false;
        int current_col_start = 0;
        int last_active_col = 0;
        int current_col_gap = 0;

    Find_Col_Bands:
        for (int x = 0; x < cols; x++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=2000

            bool active = col_sum[x] >= cth;

            if (active) {
                if (!in_col_region) {
                    in_col_region = true;
                    current_col_start = x;
                }

                last_active_col = x;
                current_col_gap = 0;
            } else {
                if (in_col_region) {
                    current_col_gap++;

                    if (current_col_gap > col_gap) {
                        int sx0 = current_col_start;
                        int sx1 = last_active_col;

                        refine_and_add_box(
                            bin_img,
                            boxes,
                            &box_count,
                            rows,
                            cols,
                            y0,
                            y1,
                            sx0,
                            sx1,
                            min_width,
                            min_height,
                            min_area
                        );

                        if (box_count >= MAX_DIGITS) {
                            break;
                        }

                        in_col_region = false;
                        current_col_gap = 0;
                    }
                }
            }
        }

        if (in_col_region && box_count < MAX_DIGITS) {
            int sx0 = current_col_start;
            int sx1 = last_active_col;

            refine_and_add_box(
                bin_img,
                boxes,
                &box_count,
                rows,
                cols,
                y0,
                y1,
                sx0,
                sx1,
                min_width,
                min_height,
                min_area
            );
        }
    }

    *num_boxes = box_count;
}
