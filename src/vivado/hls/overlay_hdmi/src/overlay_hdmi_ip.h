#ifndef OVERLAY_HDMI_IP_H
#define OVERLAY_HDMI_IP_H

#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <ap_int.h>

#define MAX_OVERLAY_WIDTH   2000
#define MAX_OVERLAY_HEIGHT  2000

#define MAX_OVERLAY_BOXES   11

typedef struct {
    int x0;
    int y0;
    int x1;
    int y1;
    int area;
} OverlayBBox;

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
);

#endif
//#ifndef OVERLAY_HDMI_IP_H
//#define OVERLAY_HDMI_IP_H
//
//#include <hls_stream.h>
//#include <ap_axi_sdata.h>
//#include <ap_int.h>
//
//void overlay_hdmi_ip(
//    hls::stream<ap_axiu<24,1,1,1> > &src_axi,
//    hls::stream<ap_axiu<24,1,1,1> > &dst_axi
//);
//
//#endif
