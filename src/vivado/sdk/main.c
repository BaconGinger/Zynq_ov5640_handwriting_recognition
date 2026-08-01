#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "xil_types.h"
#include "xil_cache.h"
#include "xparameters.h"
#include "xaxivdma.h"
#include "xaxivdma_i.h"
#include "xil_io.h"
#include "sleep.h"

#include "display_ctrl_hdmi/display_ctrl.h"
#include "vdma_api/vdma_api.h"
#include "emio_sccb_cfg/emio_sccb_cfg.h"
#include "ov5640/ov5640_init.h"

/*
 * HLS IP register headers
 * 如果你的文件名不同，按实际工程修改。
 */
#include "xpreprocess_hls_ip_hw.h"
#include "xprojection_bbox_ip_hw.h"
#include "xroi_resize_mnist_ip_hw.h"
#include "xnn_inference_hw_hw.h"
#include "xoverlay_hdmi_ip_hw.h"


// ================================================================
// Video / Display
// ================================================================

#define DYNCLK_BASEADDR  XPAR_AXI_DYNCLK_0_BASEADDR
#define VDMA_ID          XPAR_AXIVDMA_0_DEVICE_ID
#define DISP_VTC_ID      XPAR_VTC_0_DEVICE_ID

#define VIDEO_WIDTH      1280
#define VIDEO_HEIGHT     720


// ================================================================
// IP Base Address
//
// 如果编译报 XPAR_xxx 未定义，打开 xparameters.h 搜索实际名字。
// ================================================================

#define PREPROCESS_BASE       XPAR_PREPROCESS_HLS_IP_0_S_AXI_CONTROL_BASEADDR
#define PROJECTION_BBOX_BASE  XPAR_PROJECTION_BBOX_IP_0_S_AXI_CONTROL_BASEADDR
#define ROI_RESIZE_BASE       XPAR_ROI_RESIZE_MNIST_IP_0_S_AXI_CONTROL_BASEADDR
#define NN_INFERENCE_BASE     XPAR_NN_INFERENCE_HW_0_S_AXI_CONTROL_BASEADDR
#define OVERLAY_BASE          XPAR_OVERLAY_HDMI_IP_0_S_AXI_CONTROL_BASEADDR


// ================================================================
// DDR Buffer Address
// ================================================================

#define DDR_BASE_ADDR         XPAR_PS7_DDR_0_S_AXI_BASEADDR

#define FRAME_BUFFER_ADDR     (DDR_BASE_ADDR + 0x01000000)

/*
 * preprocess 输出二值图
 * 1280 * 720 = 921600 bytes
 */
#define BIN_IMG_ADDR          (DDR_BASE_ADDR + 0x03000000)

/*
 * projection_bbox_ip 原始输出框
 */
#define RAW_BOXES_X0_ADDR     (DDR_BASE_ADDR + 0x04000000)
#define RAW_BOXES_Y0_ADDR     (DDR_BASE_ADDR + 0x04000100)
#define RAW_BOXES_X1_ADDR     (DDR_BASE_ADDR + 0x04000200)
#define RAW_BOXES_Y1_ADDR     (DDR_BASE_ADDR + 0x04000300)
#define RAW_BOXES_AREA_ADDR   (DDR_BASE_ADDR + 0x04000400)
#define RAW_NUM_BOXES_ADDR    (DDR_BASE_ADDR + 0x04000500)

/*
 * ROI resize 输出 28x28 float
 * 784 * 4 = 3136 bytes
 */
#define OUT_PIX_ADDR          (DDR_BASE_ADDR + 0x04001000)

/*
 * NN result
 */
#define RESULT_ADDR           (DDR_BASE_ADDR + 0x04002000)

/*
 * overlay 使用的稳定框和稳定识别结果
 * 注意：overlay 不直接读 RAW_BOXES，而读投票后的 STABLE_BOXES。
 */
#define STABLE_BOXES_X0_ADDR   (DDR_BASE_ADDR + 0x04004000)
#define STABLE_BOXES_Y0_ADDR   (DDR_BASE_ADDR + 0x04004100)
#define STABLE_BOXES_X1_ADDR   (DDR_BASE_ADDR + 0x04004200)
#define STABLE_BOXES_Y1_ADDR   (DDR_BASE_ADDR + 0x04004300)
#define STABLE_BOXES_AREA_ADDR (DDR_BASE_ADDR + 0x04004400)
#define STABLE_RESULTS_ADDR    (DDR_BASE_ADDR + 0x04004500)

#define MAX_DIGITS             10


// ================================================================
// preprocess 参数，可运行时修改
// ================================================================

static int g_win_size = 25;
static int g_mean_c   = 13;


// ================================================================
// projection 参数
// ================================================================

static int g_row_threshold = 15;
static int g_col_threshold = 8;
static int g_row_gap       = 4;
static int g_col_gap       = 14;
static int g_min_width     = 15;
static int g_min_height    = 20;
static int g_min_area      = 150;

/*
 * ROI / MNIST 参数
 */
static int g_fit_size = 18;
static int g_pad      = 4;
static int g_invert   = 0;


// ================================================================
// 投票参数
// ================================================================

/*
 * 每次识别到某个数字，给对应数字累加多少票
 */
#define VOTE_ADD_SCORE        4

/*
 * 每帧对历史票数做衰减：hist = hist * 3 / 4
 */
#define VOTE_DECAY_NUM        3
#define VOTE_DECAY_DEN        4

/*
 * 稳定识别所需最小票数
 */
#define VOTE_MIN_SCORE        8

/*
 * 某个 track 连续丢失多少帧后删除
 */
#define VOTE_MAX_MISS         5

/*
 * bbox 平滑比例：
 * 新框 = old * 3/4 + current * 1/4
 */
#define BBOX_SMOOTH_OLD       3
#define BBOX_SMOOTH_NEW       1
#define BBOX_SMOOTH_DEN       4

/*
 * 匹配阈值：
 * IoU >= 10% 认为是同一个目标。
 */
#define MATCH_IOU_THRESHOLD   10


// ================================================================
// Global variables
// ================================================================

XAxiVdma     vdma;
DisplayCtrl  dispCtrl;
VideoMode    vd_mode;


// ================================================================
// Detection / Track structures
// ================================================================

typedef struct {
    int x0;
    int y0;
    int x1;
    int y1;
    int area;
    int result;
} DetBox;

typedef struct {
    int valid;

    int x0;
    int y0;
    int x1;
    int y1;
    int area;

    int vote_hist[10];

    int stable_result;

    int age;
    int miss;
} VoteTrack;

static VoteTrack g_tracks[MAX_DIGITS];


// ================================================================
// Utility
// ================================================================

static int min_i(int a, int b)
{
    return a < b ? a : b;
}

static int max_i(int a, int b)
{
    return a > b ? a : b;
}

static int abs_i(int a)
{
    return a < 0 ? -a : a;
}

static int clamp_i(int v, int lo, int hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static int rect_area_xy(int x0, int y0, int x1, int y1)
{
    int w = x1 - x0 + 1;
    int h = y1 - y0 + 1;

    if (w <= 0 || h <= 0) {
        return 0;
    }

    return w * h;
}

static int inter_area_det_track(const DetBox *d, const VoteTrack *t)
{
    int ix0 = max_i(d->x0, t->x0);
    int iy0 = max_i(d->y0, t->y0);
    int ix1 = min_i(d->x1, t->x1);
    int iy1 = min_i(d->y1, t->y1);

    return rect_area_xy(ix0, iy0, ix1, iy1);
}

static int iou_percent_det_track(const DetBox *d, const VoteTrack *t)
{
    int inter = inter_area_det_track(d, t);

    if (inter <= 0) {
        return 0;
    }

    int area_d = rect_area_xy(d->x0, d->y0, d->x1, d->y1);
    int area_t = rect_area_xy(t->x0, t->y0, t->x1, t->y1);

    int uni = area_d + area_t - inter;

    if (uni <= 0) {
        return 0;
    }

    return inter * 100 / uni;
}

static int center_match_score(const DetBox *d, const VoteTrack *t)
{
    int dcx = (d->x0 + d->x1) / 2;
    int dcy = (d->y0 + d->y1) / 2;

    int tcx = (t->x0 + t->x1) / 2;
    int tcy = (t->y0 + t->y1) / 2;

    int dx = abs_i(dcx - tcx);
    int dy = abs_i(dcy - tcy);

    int dw = d->x1 - d->x0 + 1;
    int dh = d->y1 - d->y0 + 1;
    int tw = t->x1 - t->x0 + 1;
    int th = t->y1 - t->y0 + 1;

    int ref_w = max_i(dw, tw);
    int ref_h = max_i(dh, th);

    /*
     * 中心点距离限制。
     * 允许目标轻微移动。
     */
    int lim_x = ref_w + 30;
    int lim_y = ref_h + 30;

    if (dx <= lim_x && dy <= lim_y) {
        int score = 500 - dx - dy;
        if (score < 1) score = 1;
        return score;
    }

    return -1;
}

static int match_score_det_track(const DetBox *d, const VoteTrack *t)
{
    if (!t->valid) {
        return -1;
    }

    int iou = iou_percent_det_track(d, t);

    if (iou >= MATCH_IOU_THRESHOLD) {
        return 1000 + iou;
    }

    return center_match_score(d, t);
}


// ================================================================
// HLS IP start/wait
// ================================================================

static int ip_start_and_wait(UINTPTR baseaddr)
{
    const int timeout_max = 200000000;
    int timeout = 0;

    Xil_Out32(baseaddr + 0x00, 0x01);

    while (1) {
        u32 ctrl = Xil_In32(baseaddr + 0x00);

        if (ctrl & 0x2) {
            return 0;
        }

        timeout++;

        if (timeout > timeout_max) {
            xil_printf("ERROR: IP timeout at 0x%08lx, ctrl=0x%08lx\r\n",
                       (unsigned long)baseaddr,
                       (unsigned long)ctrl);
            return -1;
        }
    }
}


// ================================================================
// Voting
// ================================================================

static void voting_init(void)
{
    for (int i = 0; i < MAX_DIGITS; i++) {
        g_tracks[i].valid = 0;
        g_tracks[i].x0 = 0;
        g_tracks[i].y0 = 0;
        g_tracks[i].x1 = 0;
        g_tracks[i].y1 = 0;
        g_tracks[i].area = 0;
        g_tracks[i].stable_result = -1;
        g_tracks[i].age = 0;
        g_tracks[i].miss = 0;

        for (int k = 0; k < 10; k++) {
            g_tracks[i].vote_hist[k] = 0;
        }
    }
}

static void voting_decay_track(VoteTrack *t)
{
    for (int k = 0; k < 10; k++) {
        t->vote_hist[k] = t->vote_hist[k] * VOTE_DECAY_NUM / VOTE_DECAY_DEN;
    }
}

static void voting_update_stable_result(VoteTrack *t, int current_result)
{
    int best_digit = 0;
    int best_score = t->vote_hist[0];

    for (int k = 1; k < 10; k++) {
        if (t->vote_hist[k] > best_score) {
            best_score = t->vote_hist[k];
            best_digit = k;
        }
    }

    if (best_score >= VOTE_MIN_SCORE) {
        t->stable_result = best_digit;
    } else {
        /*
         * 如果票数还不够，优先保持旧稳定值。
         * 新 track 没旧值时，用当前识别结果。
         */
        if (t->stable_result < 0 && current_result >= 0 && current_result <= 9) {
            t->stable_result = current_result;
        }
    }
}

static void voting_create_track(int idx, const DetBox *d)
{
    g_tracks[idx].valid = 1;

    g_tracks[idx].x0 = d->x0;
    g_tracks[idx].y0 = d->y0;
    g_tracks[idx].x1 = d->x1;
    g_tracks[idx].y1 = d->y1;
    g_tracks[idx].area = d->area;

    for (int k = 0; k < 10; k++) {
        g_tracks[idx].vote_hist[k] = 0;
    }

    if (d->result >= 0 && d->result <= 9) {
        g_tracks[idx].vote_hist[d->result] = VOTE_ADD_SCORE;
        g_tracks[idx].stable_result = d->result;
    } else {
        g_tracks[idx].stable_result = -1;
    }

    g_tracks[idx].age = 1;
    g_tracks[idx].miss = 0;
}

static void voting_update_track(int idx, const DetBox *d)
{
    VoteTrack *t = &g_tracks[idx];

    /*
     * 平滑 bbox，减少抖动。
     */
    t->x0 = (t->x0 * BBOX_SMOOTH_OLD + d->x0 * BBOX_SMOOTH_NEW) / BBOX_SMOOTH_DEN;
    t->y0 = (t->y0 * BBOX_SMOOTH_OLD + d->y0 * BBOX_SMOOTH_NEW) / BBOX_SMOOTH_DEN;
    t->x1 = (t->x1 * BBOX_SMOOTH_OLD + d->x1 * BBOX_SMOOTH_NEW) / BBOX_SMOOTH_DEN;
    t->y1 = (t->y1 * BBOX_SMOOTH_OLD + d->y1 * BBOX_SMOOTH_NEW) / BBOX_SMOOTH_DEN;
    t->area = (t->area * BBOX_SMOOTH_OLD + d->area * BBOX_SMOOTH_NEW) / BBOX_SMOOTH_DEN;

    voting_decay_track(t);

    if (d->result >= 0 && d->result <= 9) {
        t->vote_hist[d->result] += VOTE_ADD_SCORE;

        /*
         * 防止票数无限增大。
         */
        if (t->vote_hist[d->result] > 1000) {
            t->vote_hist[d->result] = 1000;
        }
    }

    t->age++;
    t->miss = 0;

    voting_update_stable_result(t, d->result);
}

static int voting_find_free_track(void)
{
    for (int i = 0; i < MAX_DIGITS; i++) {
        if (!g_tracks[i].valid) {
            return i;
        }
    }

    /*
     * 没有空 track，则找 miss 最大的替换。
     */
    int best_idx = 0;
    int best_miss = g_tracks[0].miss;

    for (int i = 1; i < MAX_DIGITS; i++) {
        if (g_tracks[i].miss > best_miss) {
            best_miss = g_tracks[i].miss;
            best_idx = i;
        }
    }

    return best_idx;
}

static void voting_update(const DetBox dets[MAX_DIGITS], int det_count)
{
    int track_used[MAX_DIGITS];

    for (int i = 0; i < MAX_DIGITS; i++) {
        track_used[i] = 0;
    }

    /*
     * 对每个当前检测框，匹配历史 track。
     */
    for (int d = 0; d < det_count; d++) {
        int best_track = -1;
        int best_score = -1;

        for (int t = 0; t < MAX_DIGITS; t++) {
            if (track_used[t]) {
                continue;
            }

            int score = match_score_det_track(&dets[d], &g_tracks[t]);

            if (score > best_score) {
                best_score = score;
                best_track = t;
            }
        }

        if (best_track >= 0 && best_score > 0) {
            voting_update_track(best_track, &dets[d]);
            track_used[best_track] = 1;
        } else {
            int free_idx = voting_find_free_track();
            voting_create_track(free_idx, &dets[d]);
            track_used[free_idx] = 1;
        }
    }

    /*
     * 没匹配到当前检测的 track，做 miss 和衰减。
     */
    for (int t = 0; t < MAX_DIGITS; t++) {
        if (g_tracks[t].valid && !track_used[t]) {
            g_tracks[t].miss++;
            voting_decay_track(&g_tracks[t]);

            if (g_tracks[t].miss > VOTE_MAX_MISS) {
                g_tracks[t].valid = 0;
            }
        }
    }
}

static int should_swap_det(const DetBox *a, const DetBox *b)
{
    int y_tol = 20;

    if (a->y0 > b->y0 + y_tol) {
        return 1;
    }

    if (abs_i(a->y0 - b->y0) <= y_tol && a->x0 > b->x0) {
        return 1;
    }

    return 0;
}

static void sort_det_boxes(DetBox arr[MAX_DIGITS], int n)
{
    for (int i = 0; i < MAX_DIGITS - 1; i++) {
        for (int j = 0; j < MAX_DIGITS - 1; j++) {
            if (j + 1 < n) {
                if (should_swap_det(&arr[j], &arr[j + 1])) {
                    DetBox tmp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = tmp;
                }
            }
        }
    }
}

static int voting_collect_stable(DetBox out[MAX_DIGITS])
{
    int n = 0;

    for (int i = 0; i < MAX_DIGITS; i++) {
        if (g_tracks[i].valid) {
            if (n < MAX_DIGITS) {
                out[n].x0 = clamp_i(g_tracks[i].x0, 0, VIDEO_WIDTH - 1);
                out[n].y0 = clamp_i(g_tracks[i].y0, 0, VIDEO_HEIGHT - 1);
                out[n].x1 = clamp_i(g_tracks[i].x1, 0, VIDEO_WIDTH - 1);
                out[n].y1 = clamp_i(g_tracks[i].y1, 0, VIDEO_HEIGHT - 1);
                out[n].area = g_tracks[i].area;
                out[n].result = g_tracks[i].stable_result;

                if (out[n].x1 > out[n].x0 && out[n].y1 > out[n].y0) {
                    n++;
                }
            }
        }
    }

    sort_det_boxes(out, n);

    return n;
}


// ================================================================
// Overlay
// ================================================================

static void overlay_ip_configure(void)
{
    /*
     * overlay 使用投票后的稳定框，不直接使用 projection 的 raw boxes。
     */
    Xil_Out32(OVERLAY_BASE + XOVERLAY_HDMI_IP_CONTROL_ADDR_BOXES_X0_DATA,
              STABLE_BOXES_X0_ADDR);

    Xil_Out32(OVERLAY_BASE + XOVERLAY_HDMI_IP_CONTROL_ADDR_BOXES_Y0_DATA,
              STABLE_BOXES_Y0_ADDR);

    Xil_Out32(OVERLAY_BASE + XOVERLAY_HDMI_IP_CONTROL_ADDR_BOXES_X1_DATA,
              STABLE_BOXES_X1_ADDR);

    Xil_Out32(OVERLAY_BASE + XOVERLAY_HDMI_IP_CONTROL_ADDR_BOXES_Y1_DATA,
              STABLE_BOXES_Y1_ADDR);

    Xil_Out32(OVERLAY_BASE + XOVERLAY_HDMI_IP_CONTROL_ADDR_BOXES_AREA_DATA,
              STABLE_BOXES_AREA_ADDR);

    Xil_Out32(OVERLAY_BASE + XOVERLAY_HDMI_IP_CONTROL_ADDR_RESULTS_DATA,
              STABLE_RESULTS_ADDR);

#ifdef XOVERLAY_HDMI_IP_CONTROL_ADDR_ROWS_DATA
    Xil_Out32(OVERLAY_BASE + XOVERLAY_HDMI_IP_CONTROL_ADDR_ROWS_DATA,
              VIDEO_HEIGHT);
#endif

#ifdef XOVERLAY_HDMI_IP_CONTROL_ADDR_COLS_DATA
    Xil_Out32(OVERLAY_BASE + XOVERLAY_HDMI_IP_CONTROL_ADDR_COLS_DATA,
              VIDEO_WIDTH);
#endif

    Xil_Out32(OVERLAY_BASE + XOVERLAY_HDMI_IP_CONTROL_ADDR_NUM_BOXES_DATA, 0);
    Xil_Out32(OVERLAY_BASE + XOVERLAY_HDMI_IP_CONTROL_ADDR_ENABLE_DATA, 0);

    /*
     * bit0 = ap_start
     * bit7 = auto_restart
     */
    Xil_Out32(OVERLAY_BASE + 0x00, 0x81);
}

static void overlay_write_stable_boxes(const DetBox stable[MAX_DIGITS], int num)
{
    if (num < 0) {
        num = 0;
    }

    if (num > MAX_DIGITS) {
        num = MAX_DIGITS;
    }

    volatile int *sx0  = (volatile int *)STABLE_BOXES_X0_ADDR;
    volatile int *sy0  = (volatile int *)STABLE_BOXES_Y0_ADDR;
    volatile int *sx1  = (volatile int *)STABLE_BOXES_X1_ADDR;
    volatile int *sy1  = (volatile int *)STABLE_BOXES_Y1_ADDR;
    volatile int *sarea = (volatile int *)STABLE_BOXES_AREA_ADDR;
    volatile int *sres = (volatile int *)STABLE_RESULTS_ADDR;

    for (int i = 0; i < MAX_DIGITS; i++) {
        if (i < num) {
            sx0[i] = stable[i].x0;
            sy0[i] = stable[i].y0;
            sx1[i] = stable[i].x1;
            sy1[i] = stable[i].y1;
            sarea[i] = stable[i].area;
            sres[i] = stable[i].result;
        } else {
            sx0[i] = 0;
            sy0[i] = 0;
            sx1[i] = 0;
            sy1[i] = 0;
            sarea[i] = 0;
            sres[i] = -1;
        }
    }

    Xil_DCacheFlushRange(STABLE_BOXES_X0_ADDR,   64);
    Xil_DCacheFlushRange(STABLE_BOXES_Y0_ADDR,   64);
    Xil_DCacheFlushRange(STABLE_BOXES_X1_ADDR,   64);
    Xil_DCacheFlushRange(STABLE_BOXES_Y1_ADDR,   64);
    Xil_DCacheFlushRange(STABLE_BOXES_AREA_ADDR, 64);
    Xil_DCacheFlushRange(STABLE_RESULTS_ADDR,    64);

    Xil_Out32(OVERLAY_BASE + XOVERLAY_HDMI_IP_CONTROL_ADDR_NUM_BOXES_DATA,
              num);

    if (num > 0) {
        Xil_Out32(OVERLAY_BASE + XOVERLAY_HDMI_IP_CONTROL_ADDR_ENABLE_DATA, 1);
    } else {
        Xil_Out32(OVERLAY_BASE + XOVERLAY_HDMI_IP_CONTROL_ADDR_ENABLE_DATA, 0);
    }
}


// ================================================================
// Preprocess
// ================================================================

static int run_preprocess(void)
{
    Xil_DCacheInvalidateRange(BIN_IMG_ADDR, VIDEO_WIDTH * VIDEO_HEIGHT);

    Xil_Out32(PREPROCESS_BASE + XPREPROCESS_HLS_IP_CONTROL_ADDR_SRC_IMG_DATA,
              FRAME_BUFFER_ADDR);

    Xil_Out32(PREPROCESS_BASE + XPREPROCESS_HLS_IP_CONTROL_ADDR_BIN_IMG_DATA,
              BIN_IMG_ADDR);

    Xil_Out32(PREPROCESS_BASE + XPREPROCESS_HLS_IP_CONTROL_ADDR_ROWS_DATA,
              VIDEO_HEIGHT);

    Xil_Out32(PREPROCESS_BASE + XPREPROCESS_HLS_IP_CONTROL_ADDR_COLS_DATA,
              VIDEO_WIDTH);

    Xil_Out32(PREPROCESS_BASE + XPREPROCESS_HLS_IP_CONTROL_ADDR_WIN_SIZE_DATA,
              g_win_size);

    Xil_Out32(PREPROCESS_BASE + XPREPROCESS_HLS_IP_CONTROL_ADDR_MEAN_C_DATA,
              g_mean_c);

    return ip_start_and_wait(PREPROCESS_BASE);
}


// ================================================================
// Projection bbox
// ================================================================

static int run_projection(void)
{
    Xil_DCacheInvalidateRange(RAW_BOXES_X0_ADDR,   64);
    Xil_DCacheInvalidateRange(RAW_BOXES_Y0_ADDR,   64);
    Xil_DCacheInvalidateRange(RAW_BOXES_X1_ADDR,   64);
    Xil_DCacheInvalidateRange(RAW_BOXES_Y1_ADDR,   64);
    Xil_DCacheInvalidateRange(RAW_BOXES_AREA_ADDR, 64);
    Xil_DCacheInvalidateRange(RAW_NUM_BOXES_ADDR,  64);

    Xil_Out32(PROJECTION_BBOX_BASE +
              XPROJECTION_BBOX_IP_CONTROL_ADDR_BIN_IMG_DATA,
              BIN_IMG_ADDR);

    Xil_Out32(PROJECTION_BBOX_BASE +
              XPROJECTION_BBOX_IP_CONTROL_ADDR_BOXES_X0_DATA,
              RAW_BOXES_X0_ADDR);

    Xil_Out32(PROJECTION_BBOX_BASE +
              XPROJECTION_BBOX_IP_CONTROL_ADDR_BOXES_Y0_DATA,
              RAW_BOXES_Y0_ADDR);

    Xil_Out32(PROJECTION_BBOX_BASE +
              XPROJECTION_BBOX_IP_CONTROL_ADDR_BOXES_X1_DATA,
              RAW_BOXES_X1_ADDR);

    Xil_Out32(PROJECTION_BBOX_BASE +
              XPROJECTION_BBOX_IP_CONTROL_ADDR_BOXES_Y1_DATA,
              RAW_BOXES_Y1_ADDR);

    Xil_Out32(PROJECTION_BBOX_BASE +
              XPROJECTION_BBOX_IP_CONTROL_ADDR_BOXES_AREA_DATA,
              RAW_BOXES_AREA_ADDR);

    Xil_Out32(PROJECTION_BBOX_BASE +
              XPROJECTION_BBOX_IP_CONTROL_ADDR_NUM_BOXES_DATA,
              RAW_NUM_BOXES_ADDR);

    Xil_Out32(PROJECTION_BBOX_BASE +
              XPROJECTION_BBOX_IP_CONTROL_ADDR_ROWS_DATA,
              VIDEO_HEIGHT);

    Xil_Out32(PROJECTION_BBOX_BASE +
              XPROJECTION_BBOX_IP_CONTROL_ADDR_COLS_DATA,
              VIDEO_WIDTH);

    Xil_Out32(PROJECTION_BBOX_BASE +
              XPROJECTION_BBOX_IP_CONTROL_ADDR_ROW_THRESHOLD_DATA,
              g_row_threshold);

    Xil_Out32(PROJECTION_BBOX_BASE +
              XPROJECTION_BBOX_IP_CONTROL_ADDR_COL_THRESHOLD_DATA,
              g_col_threshold);

    Xil_Out32(PROJECTION_BBOX_BASE +
              XPROJECTION_BBOX_IP_CONTROL_ADDR_ROW_GAP_DATA,
              g_row_gap);

    Xil_Out32(PROJECTION_BBOX_BASE +
              XPROJECTION_BBOX_IP_CONTROL_ADDR_COL_GAP_DATA,
              g_col_gap);

    Xil_Out32(PROJECTION_BBOX_BASE +
              XPROJECTION_BBOX_IP_CONTROL_ADDR_MIN_WIDTH_DATA,
              g_min_width);

    Xil_Out32(PROJECTION_BBOX_BASE +
              XPROJECTION_BBOX_IP_CONTROL_ADDR_MIN_HEIGHT_DATA,
              g_min_height);

    Xil_Out32(PROJECTION_BBOX_BASE +
              XPROJECTION_BBOX_IP_CONTROL_ADDR_MIN_AREA_DATA,
              g_min_area);

    int ret = ip_start_and_wait(PROJECTION_BBOX_BASE);

    Xil_DCacheInvalidateRange(RAW_NUM_BOXES_ADDR,  64);
    Xil_DCacheInvalidateRange(RAW_BOXES_X0_ADDR,   64);
    Xil_DCacheInvalidateRange(RAW_BOXES_Y0_ADDR,   64);
    Xil_DCacheInvalidateRange(RAW_BOXES_X1_ADDR,   64);
    Xil_DCacheInvalidateRange(RAW_BOXES_Y1_ADDR,   64);
    Xil_DCacheInvalidateRange(RAW_BOXES_AREA_ADDR, 64);

    return ret;
}


// ================================================================
// ROI resize
// ================================================================

static int run_roi_resize(int x0, int y0, int x1, int y1)
{
    Xil_DCacheInvalidateRange(OUT_PIX_ADDR, 4096);

    Xil_Out32(ROI_RESIZE_BASE +
              XROI_RESIZE_MNIST_IP_CONTROL_ADDR_BIN_IMG_DATA,
              BIN_IMG_ADDR);

    Xil_Out32(ROI_RESIZE_BASE +
              XROI_RESIZE_MNIST_IP_CONTROL_ADDR_OUT_PIX_DATA,
              OUT_PIX_ADDR);

    Xil_Out32(ROI_RESIZE_BASE +
              XROI_RESIZE_MNIST_IP_CONTROL_ADDR_ROWS_DATA,
              VIDEO_HEIGHT);

    Xil_Out32(ROI_RESIZE_BASE +
              XROI_RESIZE_MNIST_IP_CONTROL_ADDR_COLS_DATA,
              VIDEO_WIDTH);

    Xil_Out32(ROI_RESIZE_BASE +
              XROI_RESIZE_MNIST_IP_CONTROL_ADDR_X0_DATA,
              x0);

    Xil_Out32(ROI_RESIZE_BASE +
              XROI_RESIZE_MNIST_IP_CONTROL_ADDR_Y0_DATA,
              y0);

    Xil_Out32(ROI_RESIZE_BASE +
              XROI_RESIZE_MNIST_IP_CONTROL_ADDR_X1_DATA,
              x1);

    Xil_Out32(ROI_RESIZE_BASE +
              XROI_RESIZE_MNIST_IP_CONTROL_ADDR_Y1_DATA,
              y1);

    Xil_Out32(ROI_RESIZE_BASE +
              XROI_RESIZE_MNIST_IP_CONTROL_ADDR_FIT_SIZE_DATA,
              g_fit_size);

    Xil_Out32(ROI_RESIZE_BASE +
              XROI_RESIZE_MNIST_IP_CONTROL_ADDR_PAD_DATA,
              g_pad);

    Xil_Out32(ROI_RESIZE_BASE +
              XROI_RESIZE_MNIST_IP_CONTROL_ADDR_INVERT_DATA,
              g_invert);

    return ip_start_and_wait(ROI_RESIZE_BASE);
}


// ================================================================
// NN inference
// ================================================================

static int run_nn_inference(int *result)
{
    Xil_DCacheInvalidateRange(RESULT_ADDR, 64);

    Xil_Out32(NN_INFERENCE_BASE +
              XNN_INFERENCE_HW_CONTROL_ADDR_INPUT_PIX_DATA,
              OUT_PIX_ADDR);

    Xil_Out32(NN_INFERENCE_BASE +
              XNN_INFERENCE_HW_CONTROL_ADDR_RESULT_DATA,
              RESULT_ADDR);

    int ret = ip_start_and_wait(NN_INFERENCE_BASE);

    Xil_DCacheInvalidateRange(RESULT_ADDR, 64);

    *result = *(volatile int *)RESULT_ADDR;

    return ret;
}


// ================================================================
// Main
// ================================================================

int main(void)
{
    xil_printf("\r\n");
    xil_printf("========================================\r\n");
    xil_printf(" HDMI + Detection + Voting Overlay Demo\r\n");
    xil_printf("========================================\r\n");

    u16 cmos_h_pixel  = VIDEO_WIDTH;
    u16 cmos_v_pixel  = VIDEO_HEIGHT;
    u16 total_h_pixel = 2570;
    u16 total_v_pixel = 980;

    /*
     * Camera init
     */
    emio_init();

    ov5640_init(cmos_h_pixel,
                cmos_v_pixel,
                total_h_pixel,
                total_v_pixel);

    vd_mode = VMODE_1280x720;

    /*
     * VDMA
     */
    run_vdma_frame_buffer(&vdma,
                          VDMA_ID,
                          vd_mode.width,
                          vd_mode.height,
                          FRAME_BUFFER_ADDR,
                          0,
                          0,
                          BOTH);

    /*
     * HDMI display
     */
    DisplayInitialize(&dispCtrl, DISP_VTC_ID, DYNCLK_BASEADDR);
    DisplaySetMode(&dispCtrl, &vd_mode);
    DisplayStart(&dispCtrl);

    xil_printf("Video started: %dx%d\r\n", VIDEO_WIDTH, VIDEO_HEIGHT);

    usleep(500000);

    /*
     * Voting init
     */
    voting_init();

    /*
     * Overlay init
     */
    overlay_ip_configure();

    /*
     * 清空 overlay DDR 区域
     */
    {
        DetBox empty[MAX_DIGITS];

        for (int i = 0; i < MAX_DIGITS; i++) {
            empty[i].x0 = 0;
            empty[i].y0 = 0;
            empty[i].x1 = 0;
            empty[i].y1 = 0;
            empty[i].area = 0;
            empty[i].result = -1;
        }

        overlay_write_stable_boxes(empty, 0);
    }

    xil_printf("Overlay and voting started.\r\n");

    int frame_id = 0;

    while (1) {
        int ret;

        /*
         * 1. preprocess
         */
        ret = run_preprocess();

        if (ret != 0) {
            xil_printf("preprocess failed\r\n");

            voting_update(NULL, 0);

            DetBox stable[MAX_DIGITS];
            int stable_num = voting_collect_stable(stable);
            overlay_write_stable_boxes(stable, stable_num);

            continue;
        }

        /*
         * 2. projection bbox
         */
        ret = run_projection();

        if (ret != 0) {
            xil_printf("projection failed\r\n");

            voting_update(NULL, 0);

            DetBox stable[MAX_DIGITS];
            int stable_num = voting_collect_stable(stable);
            overlay_write_stable_boxes(stable, stable_num);

            continue;
        }

        /*
         * 3. 读取 projection 原始框
         */
        int raw_num = *(volatile int *)RAW_NUM_BOXES_ADDR;

        if (raw_num < 0) {
            raw_num = 0;
        }

        if (raw_num > MAX_DIGITS) {
            raw_num = MAX_DIGITS;
        }

        volatile int *rx0   = (volatile int *)RAW_BOXES_X0_ADDR;
        volatile int *ry0   = (volatile int *)RAW_BOXES_Y0_ADDR;
        volatile int *rx1   = (volatile int *)RAW_BOXES_X1_ADDR;
        volatile int *ry1   = (volatile int *)RAW_BOXES_Y1_ADDR;
        volatile int *rarea = (volatile int *)RAW_BOXES_AREA_ADDR;

        DetBox dets[MAX_DIGITS];
        int det_count = 0;

        /*
         * 4. 对每个原始框做 ROI resize + NN inference
         */
        for (int i = 0; i < raw_num; i++) {
            int x0 = rx0[i];
            int y0 = ry0[i];
            int x1 = rx1[i];
            int y1 = ry1[i];
            int area = rarea[i];

            x0 = clamp_i(x0, 0, VIDEO_WIDTH - 1);
            y0 = clamp_i(y0, 0, VIDEO_HEIGHT - 1);
            x1 = clamp_i(x1, 0, VIDEO_WIDTH - 1);
            y1 = clamp_i(y1, 0, VIDEO_HEIGHT - 1);

            if (x1 <= x0 || y1 <= y0) {
                continue;
            }

            ret = run_roi_resize(x0, y0, x1, y1);

            if (ret != 0) {
                xil_printf("roi failed i=%d\r\n", i);
                continue;
            }

            int result = -1;

            ret = run_nn_inference(&result);

            if (ret != 0) {
                xil_printf("nn failed i=%d\r\n", i);
                result = -1;
            }

            if (det_count < MAX_DIGITS) {
                dets[det_count].x0 = x0;
                dets[det_count].y0 = y0;
                dets[det_count].x1 = x1;
                dets[det_count].y1 = y1;
                dets[det_count].area = area;
                dets[det_count].result = result;
                det_count++;
            }
        }

        /*
         * 5. 投票更新
         */
        voting_update(dets, det_count);

        /*
         * 6. 收集稳定框和稳定识别结果
         */
        DetBox stable[MAX_DIGITS];
        int stable_num = voting_collect_stable(stable);

        /*
         * 7. 写入 overlay 使用的 DDR
         */
        overlay_write_stable_boxes(stable, stable_num);

        /*
         * 8. Debug print
         */
        frame_id++;

        if ((frame_id % 30) == 0) {
            xil_printf("frame=%d raw=%d det=%d stable=%d win=%d mean_c=%d\r\n",
                       frame_id,
                       raw_num,
                       det_count,
                       stable_num,
                       g_win_size,
                       g_mean_c);

            xil_printf("Projection params: rth=%d cth=%d rgap=%d cgap=%d area=%d\r\n",
                       g_row_threshold,
                       g_col_threshold,
                       g_row_gap,
                       g_col_gap,
                       g_min_area);

            for (int i = 0; i < stable_num; i++) {
                xil_printf("  stable[%d]=(%d,%d)-(%d,%d), area=%d, result=%d\r\n",
                           i,
                           stable[i].x0,
                           stable[i].y0,
                           stable[i].x1,
                           stable[i].y1,
                           stable[i].area,
                           stable[i].result);
            }
        }

        /*
         * 如果串口输出太多或者系统太忙，可以打开这个延时。
         */
        // usleep(10000);
    }

    return 0;
}
