#ifndef NN_INFERENCE_H
#define NN_INFERENCE_H

// HLS Specific includes, if you plan to use hls_math functions or ap_fixed types
// #include <hls_math.h>

void nn_inference_hw(
    const float input_pix[784],
//    float output_logits[10],
    int* result
);

#endif // NN_INFERENCE_H
