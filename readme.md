# 基于 ZYNQ 与 OV5640 的手写数字多目标实时识别系统

> **Real-Time Multi-Target Handwritten Digit Recognition System based on ZYNQ-7000 & OV5640**
> 
> 哈尔滨工程大学 - 《信号与信息处理创新实践》优秀课程项目  
> 作者：龙庚培 | 专业：电子信息工程 | 学号：2023080634  

---

## 📌 项目简介 (Overview)

本项目是一套基于 **ZYNQ-7020 SoC (PYNQ-Z2)** 平台设计的端到端手写数字多目标实时识别系统。系统整合了 **OV5640 摄像头采集**、**PL 端 HLS 硬件图像预处理与神经网络加速**、**PS 端控制调度与 IoU 跟踪投票** 以及 **HDMI 720p 实时 OSD 叠加显示** 的全流程。

针对传统嵌入式识别系统中 CPU 计算资源受限、图像预处理时延高以及连通域标记算法开销大等工程痛点，本系统采用 **软硬件协同设计 (PS-PL Co-Design)** 与 **Vivado HLS 高级综合**，将高复杂度的图像处理与神经网络推理下沉至 PL 端进行流水线并行加速。关键路径延迟压至 **2.396 ns**，板级实测识别准确率达到 **94.51%**，处理帧率稳定在 **30 FPS (720p@60Hz)** 以上。

---

## ✨ 核心特性 (Key Features)

* **全链路 PL 端硬件加速**：
  * **自适应二值化 (`preprocess_hls_ip`)**：基于滑动窗口与积分图思想，利用片上 BRAM 构建 Line Ring Buffer，将局部自适应均值二值化算法的时间复杂度从 $O(W^2)$ 降低至 $O(1)$。
  * **高效投影分割 (`projection_bbox_ip`)**：设计行列投影分割算法替代高资源开销的连通域标记算法（CCL），极大地节省了 LUT 与 BRAM 资源。
  * **ROI 预处理归一化 (`roi_resize_mnist_ip`)**：硬件级完成目标区域裁剪、Padding 填充、比例缩放与浮点归一化，直接输出 $28 \times 28$ 标准 MNIST 格式矩阵。
  * **全连接神经网络推理核 (`nn_inference_hw`)**：采用 `784-64-32-24-20-16-10` 的 5 层全连接架构（FCNN），参数固化于片上 BRAM，配置 `#pragma HLS PIPELINE II=1` 实现单时钟周期吞吐。
* **PS 端智能后处理与时域平滑**：
  * **IoU 目标跟踪**：基于重叠度（Intersection over Union）算法实现跨帧多目标关联与追踪。
  * **历史帧加权投票机制**：引入置信度队列与 hysteresis 迟滞滤波机制（最少命中 8 帧生效，允许 30 帧丢失遗忘），彻底消除相机抖动与识别闪烁。
* **高带宽内存交互与实时 OSD 叠加**：
  * PL 端各个 HLS IP 核通过 **AXI4-Master (`m_axi`)** 总线与 DDR3 存储器进行高带宽数据交互。
  * 结合 **AXI VDMA (S2MM / MM2S)** 与 `overlay_hdmi_ip`，在 720p 视频流上零延迟绘制目标 Bounding Box 与识别数字字符。

---

## 📐 系统架构与内存映射 (System Architecture & Memory Map)

### 1. 软硬件协同架构 (PS-PL Co-Design)

```text
+-----------------------------------------------------------------------------------+
|                                  PL (FPGA Logic)                                  |
|                                                                                   |
|  +---------+   RGB565  +------------+ AXI4S  +----------+ DDR3   +-------------+  |
|  | OV5640  |---------->| Video In   |------->| AXI VDMA |------->| DDR3 Buffer |  |
|  | Camera  |           | to AXI4S   |        |  (S2MM)  |        | (Frame/Bin) |  |
|  +---------+           +------------+        +----------+        +-------------+  |
|                                                    ^                    |         |
|  +-------------+       AXI4-Master                 |                    v         |
|  | Preprocess  |<----------------------------------+            +---------------+ |
|  |   HLS IP    |----(Bin Image)-------------------------------->| Projection BBox| |
|  +-------------+                                                |     HLS IP    | |
|                                                                 +---------------+ |
|                                                                         |         |
|  +-------------+       AXI4-Master                              (BBox Coordinates)|
|  |NN Inference |<---(28x28 Float)--- +------------------+               v         |
|  |   HLS IP    |                     |ROI Resize HLS IP |<-----------------+        |
|  +-------------+                     +------------------+                         |
|         |                                                                         |
|  (Class Result)                                                                   |
|         |              +------------+ AXI4S  +------------+ DVI   +----------+  |
|         +------------->| Overlay    |------->| AXI4S to   |------>| HDMI     |  |
|                        | HDMI IP    |        | Video Out  | TMDS  | Display  |  |
|                        +------------+        +------------+       +----------+  |
+------------------------------^----------------------------------------------------+
                               | AXI4-Lite Register Writes & Control
+------------------------------v----------------------------------------------------+
|                                 PS (ARM Cortex-A9)                                |
|                                                                                   |
|  +-----------------------------------------------------------------------------+  |
|  |  Xilinx Standalone Driver (main.c)                                          |  |
|  |  - SCCB Config | VDMA Register Driver | HLS IP AP_START & Polling Wait       |  |
|  |  - BBox IoU Matching & Weighted Temporal Voting Filter                      |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

### 2. DDR3 内存地址映射表 (Memory Map)

系统定义了统一的 DDR3 共享内存空间，用于 PS 与各 HLS IP 核之间的高速数据交换：

| 缓存标识符 (Buffer Identifier) | 基地址宏定义 / 地址偏移量 | 数据格式与分辨率 | 访问主体与读写权限 | 描述与功能说明 |
| :--- | :--- | :--- | :--- | :--- |
| **VDMA Frame Buffers** | `FRAME_BUFFER_ADDR` | 1280×720 RGB565 | VDMA S2MM (W) / MM2S (R) | 三缓存机制，用于摄像头实时采集与 HDMI 显示环回 |
| **Binary Image Buffer** | `BIN_IMG_ADDR` | 1280×720 8-bit Gray | Preprocess IP (W) / BBox IP (R) | 二值化后单通道黑白图像缓存 |
| **Raw BBoxes Buffer** | `RAW_BOXES_X0_ADDR` | Array of `BBox` (x0,y0,x1,y1) | Projection BBox IP (W) / PS (R) | 投影分割硬件生成的原始目标框坐标数组 |
| **ROI Normalized Buffer** | `OUT_PIX_ADDR` | 28×28 Single Precision Float | ROI Resize IP (W) / NN IP (R) | 硬件裁剪、缩放与归一化后的 MNIST 格式推理矩阵 |
| **NN Inference Result** | `RESULT_ADDR` | 32-bit Integer (0 ~ 9) | NN Inference IP (W) / PS (R) | 神经网络推理输出的最优分类类别标签 |
| **Stable OSD BBoxes** | `STABLE_BOXES_X0_ADDR` | Array of `BBox` | PS ARM (W) / Overlay IP (R) | 经 PS 端 IoU 跟踪与平滑投票筛选后的稳态目标框 |
| **Stable OSD Results** | `STABLE_RESULTS_ADDR` | Array of 32-bit Integer | PS ARM (W) / Overlay IP (R) | 经 PS 端平滑滤波后的稳态分类数字结果 |

---

## 🛠️ HLS 硬件加速核设计 (HLS IP Cores)

系统包含 5 个基于 Vivado HLS 开发的独立 C++ 硬件加速核：

### 1. `preprocess_hls_ip` (图像预处理核)
* **算法**：$Gray = 0.299R + 0.587G + 0.114B$ (定点化优化 $77R + 150G + 29B + 128 \gg 8$)，结合滑动窗口局部自适应均值二值化。
* **HLS 优化**：使用 `line_ring[WIN][MAX_WIDTH]` 环形数组与 `col_sum[MAX_WIDTH]` 列累加数组，配置 `#pragma HLS RESOURCE core=RAM_S2P_BRAM`，实现 `PIPELINE II=1`。

### 2. `projection_bbox_ip` (投影分割核)
* **算法**：通过水平行投影与垂直列投影计算 `row_sum` 与 `col_sum`，按阈值切分确定文字块的 Bounding Box。
* **HLS 优化**：设置 `#pragma HLS ARRAY_PARTITION cyclic factor=4` 解除 BRAM 读写端口瓶颈；内联优化 `refine_and_add_box()` 过滤小噪声块。

### 3. `roi_resize_mnist_ip` (ROI 提取与归一化核)
* **算法**：读取目标框，自动补边 Padding，计算最近邻/双线性缩放映射，导出 $28 \times 28$ 的 `float` 数组（数值 0.0 ~ 1.0）。
* **HLS 优化**：内联 `clamp_int()` 与 `swap_int()` 防越界保护，全流水线处理。

### 4. `nn_inference_hw` (FCNN 神经网络推理核)
* **网络架构**：输入 784 维 $\rightarrow$ 64 (ReLU) $\rightarrow$ 32 (ReLU) $\rightarrow$ 24 (ReLU) $\rightarrow$ 20 (ReLU) $\rightarrow$ 16 (ReLU) $\rightarrow$ 10 (Logit Argmax)。
* **HLS 优化**：权重固化在 `#include` 头文件中，映射为 ROM；外层循环配置 `PIPELINE II=1`，内层点积展开，极大地压低推理延迟。

### 5. `overlay_hdmi_ip` (HDMI OSD 绘图核)
* **功能**：实时接收 AXI4-Stream 视频流，根据 PS 端写入的稳定框坐标与分类结果，按 7 段数码管 Segment 译码算法实时在视频流上绘制绿框与识别数字。

---

## 📊 硬件资源占用与时序分析 (Resource & Timing)

编译目标平台：**Xilinx PYNQ-Z2 (`xc7z020clg400-1`)**，Vivado 工具版本 **2018.3+**。

### 1. 资源占用表 (Resource Utilization)

| 资源类型 (Resource) | 已用数量 (Used) | 可用数量 (Available) | 利用率 (Utilization) |
| :--- | :--- | :--- | :--- |
| **Slice LUTs** | 42,131 | 53,200 | **79.19%** |
| **Slice Registers** | 46,961 | 106,400 | **44.14%** |
| **Block RAM (BRAM Tile)** | 107 | 140 | **76.43%** |
| **DSP48E1** | 45 | 220 | **20.45%** |
| **Bonded IOB** | 24 | 125 | **19.20%** |
| **MMCME2_ADV** | 1 | 4 | **25.00%** |

### 2. 时序收敛分析 (Timing Summary)

* **PL 主工作时钟 (`clk_fpga_0`)**：`75.00 MHz`
* **建立时间余量 (Setup Worst Negative Slack, WNS)**：`0.981 ns` (无 Setup 违例)
* **保持时间余量 (Hold Worst Hold Slack, WHS)**：`0.045 ns`
* **最差路径延迟 (Critical Path Delay)**：`2.396 ns`

---

## 📈 实测验证与性能表现 (Verification & Performance)

### 1. 识别准确率测试 (Accuracy)
* **软件基准仿真 (PyTorch)**：数据集结合平移、旋转与缩放数据增强后，测试集准确率为 **95.10%**（比未增强基准模型提升 0.9%）。
* **板级实测 (Board-Level Test)**：使用采集到的 100 张真实图片（共 1000 个手写数字），成功识别 945 个，板级实测准确率达 **94.51%**，与算法仿真结果基本一致。

### 2. 软件与 Golden 模型 MSE 验证
针对 C-Simulation 与 Golden CPU 结果进行了逐像素对比：
* 灰度与自适应二值化输出与 CPU Golden 结果 MSE = **0.0**。
* 神经网络推理在 10 分类输出上的判决结果与 Python 端导出结果完全匹配。

---

## 📁 项目目录结构 (Directory Structure)

```text
zynq-digit-recognition/
├── docs/                           # 实验报告与说明文档
│   └── 20230812-龙庚培-手写数字识别.pdf # 完整报告 PDF
├── hls/                            # Vivado HLS 源码与 Testbench
│   ├── preprocess/                 # 图像二值化 IP 源码 (preprocess_hls_ip.cpp)
│   ├── projection_bbox/            # 投影分割 IP 源码 (projection_bbox_ip.cpp)
│   ├── roi_resize_mnist/           # ROI 归一化 IP 源码 (roi_resize_mnist_ip.cpp)
│   ├── nn_inference/               # FCNN 推理 IP 源码 (nn_inference.cpp)
│   └── overlay_hdmi/               # HDMI OSD 叠加 IP 源码 (overlay_hdmi_ip.cpp)
├── hw/                             # Vivado 工程与 Constraints
│   ├── block_design/               # Vivado Block Design 导出的 tcl 脚本
│   └── constraints/                # Pin 管脚与时序约束 (.xdc)
├── sw/                             # PS 端 Xilinx SDK C 语言驱动与调度逻辑
│   ├── main.c                      # PS 端主程序与状态机控制
│   └── voting_tracker.c            # IoU 匹配与加权投票平滑算法
├── python/                         # PyTorch 模型训练与权值导出
│   ├── train_fcnn.py               # 模型训练与数据增强脚本
│   └── export_hls_weights.py       # 导出 C++ #include 权重头文件脚本
└── README.md                       # 本文档
```

---

## 🚀 快速上手 (Quick Start)

### 1. HLS IP 综合与导出
1. 打开 **Vivado HLS**，针对 `hls/` 下各个子目录新建工程。
2. 顶层函数指定为对应的 IP 接口函数（如 `preprocess_hls_ip`）。
3. 运行 **C Synthesis**，确认时序与 II=1 满足要求后，执行 **Export RTL** 生成 IP Zip 包。

### 2. Vivado 硬件 Block Design 集成
1. 打开 Vivado，新建基于 `xc7z020clg400-1` 的工程。
2. 将导出的 HLS IP 添加至 IP Repository。
3. 运行 `hw/block_design/` 下的 tcl 脚本自动构建 Block Design。
4. 编译生成 Bitstream 文件并导出 `.sdk` 硬件描述。

### 3. PS 端 SDK 软件编译与部署
1. 打开 **Xilinx SDK / Vitis**，导入 `sw/` 下的 C 源码。
2. 连接 PYNQ-Z2 开发板、OV5640 摄像头与 HDMI 显示器。
3. 上电并 Program FPGA，运行 PS 端 C 程序，观察 HDMI 显示器上的实时识别与绿框绘制效果。

---

## 📜 许可协议 (License)

本项目基于 [MIT License](LICENSE) 开源 - 欢迎学术交流与二次开发！