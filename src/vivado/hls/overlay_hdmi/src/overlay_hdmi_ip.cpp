#include "overlay_hdmi_ip.h"

// ============================================================
// Local structure
// ============================================================
typedef struct {
    ap_uint<1> valid;
    ap_uint<1> digit_valid;

    int x0;
    int y0;
    int x1;
    int y1;

    int left_end;
    int right_start;
    int top_end;
    int bottom_start;

    int bg_x0;
    int bg_x1;
    int bg_y0;
    int bg_y1;

    int xl0;
    int xl1;
    int xm0;
    int xm1;
    int xr0;
    int xr1;

    int yt0;
    int yt1;
    int yu0;
    int yu1;
    int ym0;
    int ym1;
    int yl0;
    int yl1;
    int yb0;
    int yb1;

    ap_uint<7> seg;
} LocalOverlayBox;


// ============================================================
// Helper functions
// ============================================================
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


static bool in_range_i(int v, int a, int b) {
#pragma HLS INLINE

    return v >= a && v <= b;
}


static OverlayBBox read_volatile_bbox(volatile OverlayBBox *boxes, int idx) {
#pragma HLS INLINE

    OverlayBBox b;

    b.x0   = boxes[idx].x0;
    b.y0   = boxes[idx].y0;
    b.x1   = boxes[idx].x1;
    b.y1   = boxes[idx].y1;
    b.area = boxes[idx].area;

    return b;
}


static ap_uint<7> digit_to_segments(int digit) {
#pragma HLS INLINE

    switch (digit) {
        case 0: return 0x3F;
        case 1: return 0x06;
        case 2: return 0x5B;
        case 3: return 0x4F;
        case 4: return 0x66;
        case 5: return 0x6D;
        case 6: return 0x7D;
        case 7: return 0x07;
        case 8: return 0x7F;
        case 9: return 0x6F;
        default: return 0x00;
    }
}


static void clear_local_box(LocalOverlayBox &lb) {
#pragma HLS INLINE

    lb.valid       = 0;
    lb.digit_valid = 0;

    lb.x0 = 0;
    lb.y0 = 0;
    lb.x1 = 0;
    lb.y1 = 0;

    lb.left_end     = 0;
    lb.right_start  = 0;
    lb.top_end      = 0;
    lb.bottom_start = 0;

    lb.bg_x0 = 0;
    lb.bg_x1 = 0;
    lb.bg_y0 = 0;
    lb.bg_y1 = 0;

    lb.xl0 = 0;
    lb.xl1 = 0;
    lb.xm0 = 0;
    lb.xm1 = 0;
    lb.xr0 = 0;
    lb.xr1 = 0;

    lb.yt0 = 0;
    lb.yt1 = 0;
    lb.yu0 = 0;
    lb.yu1 = 0;
    lb.ym0 = 0;
    lb.ym1 = 0;
    lb.yl0 = 0;
    lb.yl1 = 0;
    lb.yb0 = 0;
    lb.yb1 = 0;

    lb.seg = 0;
}


static bool is_on_box_border(int x, int y, const LocalOverlayBox &lb) {
#pragma HLS INLINE

    if (!lb.valid) return false;

    bool in_box =
        x >= lb.x0 && x <= lb.x1 &&
        y >= lb.y0 && y <= lb.y1;

    if (!in_box) return false;

    bool on_left =
        x >= lb.x0 && x <= lb.left_end;

    bool on_right =
        x >= lb.right_start && x <= lb.x1;

    bool on_top =
        y >= lb.y0 && y <= lb.top_end;

    bool on_bottom =
        y >= lb.bottom_start && y <= lb.y1;

    return on_left || on_right || on_top || on_bottom;
}


static bool is_digit_bg_pixel(int x, int y, const LocalOverlayBox &lb) {
#pragma HLS INLINE

    if (!lb.valid || !lb.digit_valid) return false;

    return x >= lb.bg_x0 && x <= lb.bg_x1 &&
           y >= lb.bg_y0 && y <= lb.bg_y1;
}


static bool is_digit_pixel(int x, int y, const LocalOverlayBox &lb) {
#pragma HLS INLINE

    if (!lb.valid || !lb.digit_valid) return false;

    ap_uint<7> seg = lb.seg;

    bool seg_a =
        seg[0] &&
        in_range_i(y, lb.yt0, lb.yt1) &&
        in_range_i(x, lb.xm0, lb.xm1);

    bool seg_b =
        seg[1] &&
        in_range_i(y, lb.yu0, lb.yu1) &&
        in_range_i(x, lb.xr0, lb.xr1);

    bool seg_c =
        seg[2] &&
        in_range_i(y, lb.yl0, lb.yl1) &&
        in_range_i(x, lb.xr0, lb.xr1);

    bool seg_d =
        seg[3] &&
        in_range_i(y, lb.yb0, lb.yb1) &&
        in_range_i(x, lb.xm0, lb.xm1);

    bool seg_e =
        seg[4] &&
        in_range_i(y, lb.yl0, lb.yl1) &&
        in_range_i(x, lb.xl0, lb.xl1);

    bool seg_f =
        seg[5] &&
        in_range_i(y, lb.yu0, lb.yu1) &&
        in_range_i(x, lb.xl0, lb.xl1);

    bool seg_g =
        seg[6] &&
        in_range_i(y, lb.ym0, lb.ym1) &&
        in_range_i(x, lb.xm0, lb.xm1);

    return seg_a || seg_b || seg_c || seg_d || seg_e || seg_f || seg_g;
}


// ============================================================
// Top Function
// ============================================================
void overlay_hdmi_ip(
    hls::stream<ap_axiu<24,1,1,1> > &src_axi,
    hls::stream<ap_axiu<24,1,1,1> > &dst_axi,

    volatile OverlayBBox *boxes,
    volatile int *results,

    int num_boxes,
    int rows,
    int cols,
    int enable,
    int box_thickness,
    int digit_scale
) {
#pragma HLS INTERFACE axis port=src_axi
#pragma HLS INTERFACE axis port=dst_axi

#pragma HLS INTERFACE m_axi port=boxes   offset=slave bundle=gmem depth=11
#pragma HLS INTERFACE m_axi port=results offset=slave bundle=gmem depth=11

#pragma HLS INTERFACE s_axilite port=boxes         bundle=control
#pragma HLS INTERFACE s_axilite port=results       bundle=control
#pragma HLS INTERFACE s_axilite port=num_boxes     bundle=control
#pragma HLS INTERFACE s_axilite port=rows          bundle=control
#pragma HLS INTERFACE s_axilite port=cols          bundle=control
#pragma HLS INTERFACE s_axilite port=enable        bundle=control
#pragma HLS INTERFACE s_axilite port=box_thickness bundle=control
#pragma HLS INTERFACE s_axilite port=digit_scale   bundle=control
#pragma HLS INTERFACE s_axilite port=return        bundle=control

    LocalOverlayBox local_boxes[MAX_OVERLAY_BOXES];
#pragma HLS ARRAY_PARTITION variable=local_boxes complete dim=1

    int _rows = (rows > 0 && rows <= MAX_OVERLAY_HEIGHT) ? rows : 720;
    int _cols = (cols > 0 && cols <= MAX_OVERLAY_WIDTH)  ? cols : 1280;

    // ========================================================
    // enable = 0: bypass
    // ========================================================
    if (enable == 0) {
    Bypass_Y:
        for (int y = 0; y < _rows; y++) {
#pragma HLS LOOP_TRIPCOUNT min=720 max=720

        Bypass_X:
            for (int x = 0; x < _cols; x++) {
#pragma HLS LOOP_TRIPCOUNT min=1280 max=1280
#pragma HLS PIPELINE II=1

                ap_axiu<24,1,1,1> pix = src_axi.read();
                dst_axi.write(pix);
            }
        }

        return;
    }

    // ========================================================
    // enable = 2: hard-coded red box test, no DDR read
    // ========================================================
    if (enable == 2) {
    HardBox_Y:
        for (int y = 0; y < _rows; y++) {
#pragma HLS LOOP_TRIPCOUNT min=720 max=720

        HardBox_X:
            for (int x = 0; x < _cols; x++) {
#pragma HLS LOOP_TRIPCOUNT min=1280 max=1280
#pragma HLS PIPELINE II=1

                ap_axiu<24,1,1,1> in_pixel = src_axi.read();
                ap_axiu<24,1,1,1> out_pixel = in_pixel;

                bool border =
                    ((x >= 100 && x <= 400 && y >= 100 && y <= 103) ||
                     (x >= 100 && x <= 400 && y >= 297 && y <= 300) ||
                     (x >= 100 && x <= 103 && y >= 100 && y <= 300) ||
                     (x >= 397 && x <= 400 && y >= 100 && y <= 300));

                if (border) {
                    out_pixel.data = 0xFF0000;
                }

                dst_axi.write(out_pixel);
            }
        }

        return;
    }

    // ========================================================
    // Common parameter clamp
    // ========================================================
    int n = num_boxes;

    if (n < 0) n = 0;
    if (n > MAX_OVERLAY_BOXES) n = MAX_OVERLAY_BOXES;

    if (box_thickness < 1)  box_thickness = 1;
    if (box_thickness > 10) box_thickness = 10;

    if (digit_scale < 1) digit_scale = 1;
    if (digit_scale > 8) digit_scale = 8;

    const int s  = digit_scale;
    const int s4 = digit_scale << 2;
    const int s5 = digit_scale * 5;
    const int s8 = digit_scale << 3;
    const int s9 = digit_scale * 9;

    const int digit_w = s5;
    const int digit_h = s9;

    // ========================================================
    // Load boxes/results from DDR
    // ========================================================
Load_Boxes:
    for (int i = 0; i < MAX_OVERLAY_BOXES; i++) {
#pragma HLS LOOP_TRIPCOUNT min=11 max=11
#pragma HLS PIPELINE

        LocalOverlayBox lb;
        clear_local_box(lb);

        if (i < n) {
            OverlayBBox b = read_volatile_bbox(boxes, i);
            int digit = results[i];

            if (b.x0 > b.x1) swap_int(b.x0, b.x1);
            if (b.y0 > b.y1) swap_int(b.y0, b.y1);

            b.x0 = clamp_int(b.x0, 0, _cols - 1);
            b.x1 = clamp_int(b.x1, 0, _cols - 1);
            b.y0 = clamp_int(b.y0, 0, _rows - 1);
            b.y1 = clamp_int(b.y1, 0, _rows - 1);

            lb.valid = 1;

            lb.x0 = b.x0;
            lb.y0 = b.y0;
            lb.x1 = b.x1;
            lb.y1 = b.y1;

            int left_end_i = b.x0 + box_thickness - 1;
            if (left_end_i > b.x1) left_end_i = b.x1;

            int right_start_i = b.x1 - box_thickness + 1;
            if (right_start_i < b.x0) right_start_i = b.x0;

            int top_end_i = b.y0 + box_thickness - 1;
            if (top_end_i > b.y1) top_end_i = b.y1;

            int bottom_start_i = b.y1 - box_thickness + 1;
            if (bottom_start_i < b.y0) bottom_start_i = b.y0;

            lb.left_end     = left_end_i;
            lb.right_start  = right_start_i;
            lb.top_end      = top_end_i;
            lb.bottom_start = bottom_start_i;

            if (digit >= 0 && digit <= 9) {
                lb.digit_valid = 1;
                lb.seg = digit_to_segments(digit);

                int tx = b.x0;
                int ty = b.y0 - digit_h - 3;

                if (ty < 0) {
                    ty = b.y0 + 2;
                }

                if (tx + digit_w >= _cols) {
                    tx = _cols - digit_w - 1;
                }

                if (tx < 0) {
                    tx = 0;
                }

                if (ty + digit_h >= _rows) {
                    ty = _rows - digit_h - 1;
                }

                if (ty < 0) {
                    ty = 0;
                }

                int bg_x0 = tx - 1;
                int bg_y0 = ty - 1;
                int bg_x1 = tx + digit_w;
                int bg_y1 = ty + digit_h;

                if (bg_x0 < 0) bg_x0 = 0;
                if (bg_y0 < 0) bg_y0 = 0;
                if (bg_x1 > _cols - 1) bg_x1 = _cols - 1;
                if (bg_y1 > _rows - 1) bg_y1 = _rows - 1;

                lb.bg_x0 = bg_x0;
                lb.bg_x1 = bg_x1;
                lb.bg_y0 = bg_y0;
                lb.bg_y1 = bg_y1;

                lb.xl0 = tx;
                lb.xl1 = tx + s - 1;

                lb.xm0 = tx + s;
                lb.xm1 = tx + s4 - 1;

                lb.xr0 = tx + s4;
                lb.xr1 = tx + s5 - 1;

                lb.yt0 = ty;
                lb.yt1 = ty + s - 1;

                lb.yu0 = ty + s;
                lb.yu1 = ty + s4 - 1;

                lb.ym0 = ty + s4;
                lb.ym1 = ty + s5 - 1;

                lb.yl0 = ty + s5;
                lb.yl1 = ty + s8 - 1;

                lb.yb0 = ty + s8;
                lb.yb1 = ty + s9 - 1;
            }
        }

        local_boxes[i] = lb;
    }

    // ========================================================
    // enable = 3: DDR box test, only draw red boxes
    // ========================================================
    if (enable == 3) {
    DDRBox_Y:
        for (int y = 0; y < _rows; y++) {
#pragma HLS LOOP_TRIPCOUNT min=720 max=720

        DDRBox_X:
            for (int x = 0; x < _cols; x++) {
#pragma HLS LOOP_TRIPCOUNT min=1280 max=1280
#pragma HLS PIPELINE II=1

                ap_axiu<24,1,1,1> in_pixel = src_axi.read();
                ap_axiu<24,1,1,1> out_pixel = in_pixel;

                bool draw_box = false;

            DDRBox_Check:
                for (int i = 0; i < MAX_OVERLAY_BOXES; i++) {
#pragma HLS UNROLL

                    if (is_on_box_border(x, y, local_boxes[i])) {
                        draw_box = true;
                    }
                }

                if (draw_box) {
                    out_pixel.data = 0xFF0000;
                }

                dst_axi.write(out_pixel);
            }
        }

        return;
    }

    // ========================================================
    // enable = 1: normal overlay
    // red box + black digit background + green digit
    // ========================================================
Overlay_Y:
    for (int y = 0; y < _rows; y++) {
#pragma HLS LOOP_TRIPCOUNT min=720 max=720

    Overlay_X:
        for (int x = 0; x < _cols; x++) {
#pragma HLS LOOP_TRIPCOUNT min=1280 max=1280
#pragma HLS PIPELINE II=1

            ap_axiu<24,1,1,1> in_pixel = src_axi.read();
            ap_axiu<24,1,1,1> out_pixel = in_pixel;

            bool draw_box   = false;
            bool draw_bg    = false;
            bool draw_digit = false;

        Overlay_Check:
            for (int i = 0; i < MAX_OVERLAY_BOXES; i++) {
#pragma HLS UNROLL

                LocalOverlayBox lb = local_boxes[i];

                if (is_on_box_border(x, y, lb)) {
                    draw_box = true;
                }

                if (is_digit_bg_pixel(x, y, lb)) {
                    draw_bg = true;
                }

                if (is_digit_pixel(x, y, lb)) {
                    draw_digit = true;
                }
            }

            if (draw_digit) {
                out_pixel.data = 0x00FF00;
            } else if (draw_bg) {
                out_pixel.data = 0x000000;
            } else if (draw_box) {
                out_pixel.data = 0xFF0000;
            }

            dst_axi.write(out_pixel);
        }
    }

    return;
}
