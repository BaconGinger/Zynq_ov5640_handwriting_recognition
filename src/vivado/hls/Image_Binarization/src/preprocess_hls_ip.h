#ifndef PREPROCESS_HLS_IP_H
#define PREPROCESS_HLS_IP_H

#include <ap_int.h>
#include <hls_stream.h>

#define PREPROCESS_MAX_WIDTH   1280
#define PREPROCESS_MAX_HEIGHT  720

void preprocess_hls_ip(
    volatile unsigned char *src_img,
    volatile unsigned char *bin_img,

    int rows,
    int cols,

    int win_size,
    int mean_c
);

#endif
