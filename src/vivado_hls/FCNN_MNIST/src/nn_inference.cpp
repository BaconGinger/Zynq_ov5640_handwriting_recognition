#include "nn_inference.h"
#include <hls_math.h> // Recommended for HLS-optimized math functions like expf

// HLS directives go here
// Use pragmas for synthesis optimization

// ReLU activation function - can be inlined for better performance
static float relu_hw(float x)
{
#pragma HLS INLINE // Inlining helps avoid function call overhead in hardware
	return x > 0.0f ? x : 0.0f;
}

void nn_inference_hw(
    const float input_pix[784],
//    float output_logits[10],
    int* result
) {
#pragma HLS INTERFACE m_axi port=input_pix     offset=slave bundle=gmem depth=784
//#pragma HLS INTERFACE m_axi port=output_logits offset=slave bundle=gmem depth=10
#pragma HLS INTERFACE m_axi port=result        offset=slave bundle=gmem depth=1

#pragma HLS INTERFACE s_axilite port=input_pix     bundle=control
//#pragma HLS INTERFACE s_axilite port=output_logits bundle=control
#pragma HLS INTERFACE s_axilite port=result        bundle=control
#pragma HLS INTERFACE s_axilite port=return        bundle=control

    // 权重：用 static const，不分区，不复制
    static const float weight1[784][64] = {
        #include "./python_output/weights_export/weight1.dat"
    };
    static const float weight2[64][32] = {
        #include "./python_output/weights_export/weight2.dat"
    };
    static const float weight3[32][24] = {
        #include "./python_output/weights_export/weight3.dat"
    };
    static const float weight4[24][20] = {
        #include "./python_output/weights_export/weight4.dat"
    };
    static const float weight5[20][16] = {
        #include "./python_output/weights_export/weight5.dat"
    };
    static const float weight6[16][10] = {
        #include "./python_output/weights_export/weight6.dat"
    };

    static const float bias1[64] = {
        #include "./python_output/weights_export/bias1.dat"
    };
    static const float bias2[32] = {
        #include "./python_output/weights_export/bias2.dat"
    };
    static const float bias3[24] = {
        #include "./python_output/weights_export/bias3.dat"
    };
    static const float bias4[20] = {
        #include "./python_output/weights_export/bias4.dat"
    };
    static const float bias5[16] = {
        #include "./python_output/weights_export/bias5.dat"
    };
    static const float bias6[10] = {
        #include "./python_output/weights_export/bias6.dat"
    };

    // 中间结果用普通数组，不分区
    float a1[64];
    float a2[32];
    float a3[24];
    float a4[20];
    float a5[16];
    float a6[10];
//    float output_logits[10];
    // ============================================================
    // Layer 1: 784 -> 64 (ReLU)
    //
    // 关键修改：
    //   - 外层 k 循环不做 PIPELINE
    //   - 内层 j 循环做 PIPELINE II=1
    //   - 不要 UNROLL
    // ============================================================
    for (int k = 0; k < 64; k++) {
        float z = bias1[k];
        for (int j = 0; j < 784; j++) {
#pragma HLS PIPELINE II=1
            z += weight1[j][k] * input_pix[j];
        }
        a1[k] = relu_hw(z);
    }

    // ============================================================
    // Layer 2: 64 -> 32 (ReLU)
    // ============================================================
    for (int k = 0; k < 32; k++) {
        float z = bias2[k];
        for (int j = 0; j < 64; j++) {
#pragma HLS PIPELINE II=1
            z += weight2[j][k] * a1[j];
        }
        a2[k] = relu_hw(z);
    }

    // ============================================================
    // Layer 3: 32 -> 24 (ReLU)
    // ============================================================
    for (int k = 0; k < 24; k++) {
        float z = bias3[k];
        for (int j = 0; j < 32; j++) {
#pragma HLS PIPELINE II=1
            z += weight3[j][k] * a2[j];
        }
        a3[k] = relu_hw(z);
    }

    // ============================================================
    // Layer 4: 24 -> 20 (ReLU)
    // ============================================================
    for (int k = 0; k < 20; k++) {
        float z = bias4[k];
        for (int j = 0; j < 24; j++) {
#pragma HLS PIPELINE II=1
            z += weight4[j][k] * a3[j];
        }
        a4[k] = relu_hw(z);
    }

    // ============================================================
    // Layer 5: 20 -> 16 (ReLU)
    // ============================================================
    for (int k = 0; k < 16; k++) {
        float z = bias5[k];
        for (int j = 0; j < 20; j++) {
#pragma HLS PIPELINE II=1
            z += weight5[j][k] * a4[j];
        }
        a5[k] = relu_hw(z);
    }

    // ============================================================
    // Layer 6: 16 -> 10 (Sigmoid Output)
    // ============================================================
    for (int k = 0; k < 10; k++) {
        float z = bias6[k];
        for (int j = 0; j < 16; j++) {
    #pragma HLS PIPELINE II=1
            z += weight6[j][k] * a5[j];
        }

        // 不做 sigmoid，直接保存 logit
        a6[k] = z;
//        output_logits[k] = z;
    }

    // ============================================================
    // Find max index
    // ============================================================
    int max_idx = 0;
    float max_val = a6[0];

    for (int i = 1; i < 10; i++) {
    #pragma HLS PIPELINE II=1
        if (a6[i] > max_val) {
            max_val = a6[i];
            max_idx = i;
        }
    }

    *result = max_idx;
}
