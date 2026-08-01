#ifndef ROI_RESIZE_MNIST_IP_H
#define ROI_RESIZE_MNIST_IP_H

#define ROI_MAX_WIDTH   2000
#define ROI_MAX_HEIGHT  2000
#define ROI_MAX_PIXELS  (ROI_MAX_WIDTH * ROI_MAX_HEIGHT)

#define MNIST_SIZE      28
#define MNIST_PIXELS    784

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
);

#endif
