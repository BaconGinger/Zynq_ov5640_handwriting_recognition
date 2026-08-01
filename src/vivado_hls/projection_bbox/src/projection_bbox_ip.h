#ifndef PROJECTION_BBOX_IP_H
#define PROJECTION_BBOX_IP_H

#define MAX_WIDTH       2000
#define MAX_HEIGHT      2000
#define MAX_PIXELS      (MAX_WIDTH * MAX_HEIGHT)

#define MAX_DIGITS      64
#define MAX_ROW_BANDS   32

typedef struct {
    int x0;
    int y0;
    int x1;
    int y1;
    int area;
} BBox;

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
);

#endif
