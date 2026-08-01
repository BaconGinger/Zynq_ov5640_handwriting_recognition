# 手写数字实时识别系统

基于 ZYNQ-7020 (PYNQ-Z2) + OV5640 摄像头的手写数字多目标实时识别，PL 端 HLS 硬件加速，HDMI 720p 实时显示。

## 硬件

- 开发板：PYNQ-Z2 (`xc7z020clg400-1`)
- 摄像头：OV5640
- 显示：HDMI 720p@60Hz
- 工具：Vivado 2018.3, Vivado HLS, Xilinx SDK

## 目录结构

```
src/
├── vivado/
│   ├── hls/                          # HLS IP 核源码
│   │   ├── Image_Binarization/       # 图像灰度+自适应二值化
│   │   ├── projection_bbox/          # 行列投影分割，替代连通域标记
│   │   ├── roi_resize_mnist/         # ROI 裁剪+缩放至 28x28 浮点
│   │   ├── FCNN_MNIST/               # 5 层全连接神经网络推理 (784-64-32-24-20-16-10)
│   │   └── overlay_hdmi/             # HDMI OSD 叠加（框+数字）
│   ├── ip_core/                      # 导出的 IP 核打包
│   ├── src/constrs_1/new/top.xdc     # 管脚与时序约束
│   └── tcl/
│       ├── bd.tcl                    # Block Design 脚本
│       └── project.tcl               # Vivado 工程重建脚本
└── python/
    ├── bp_train.py                   # PyTorch 模型训练
    └── main.py                       # 预处理/检测/模型导出
test_figs/                            # 测试图片
```

## 数据流

```
OV5640 → Video In → VDMA(S2MM) → DDR3
                                    ↓
         preprocess (二值化) ←──── DDR3
                ↓
         projection_bbox (分割) → DDR3
                                    ↓
         roi_resize (归一化) ←──── DDR3
                ↓
         nn_inference (推理) → DDR3
                                    ↓
         overlay_hdmi (OSD) ←────── DDR3
                ↓
         Video Out → HDMI
```

PS 端 (ARM) 负责 SCCB 配置摄像头、控制 HLS IP 启停、IoU 多目标跟踪和时域投票滤波。

## 构建步骤

### 1. HLS IP 综合

打开 Vivado HLS，为 `src/vivado/hls/` 下每个子目录建立工程，顶层函数对应各 `*_ip.cpp`，运行 C Synthesis 后 Export RTL。

### 2. Vivado 工程

Vivado Tcl Shell 中执行：

```
cd src/vivado/tcl
source project.tcl
```

然后将导出的 HLS IP 加入 IP Repository，`source bd.tcl` 构建 Block Design，生成 Bitstream 并导出硬件描述。

### 3. PS 端程序

在 Xilinx SDK 中基于导出的硬件描述创建 BSP，编译链接 PS 端 C 代码（需自行准备），Program FPGA 后运行。

## 结果

- 板级实测：100 张真实图片（1000 个手写数字），识别率 94.51%
- 处理帧率：30 FPS（720p@60Hz 输入）
- PL 资源占用：LUT 79%, BRAM 76%, DSP 20% (xc7z020clg400-1)
- 时序：主频 75MHz，WNS 0.981ns，WHS 0.045ns

## 许可

MIT License
