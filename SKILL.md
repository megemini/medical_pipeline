---
name: drug-ocr-pipeline
description: "构建并运行药品说明书智能识别与语音播报系统。当用户需要识别药品说明书文字、提取药品关键信息（名称/适应症/用法用量/禁忌/不良反应）、将药品信息语音播报给老年人时使用此技能。基于 PaddleOCR-VL (OCR识别) + Qwen3-VL (多模态信息提取) + Qwen3-TTS (语音合成) 的 OpenVINO 本地推理。"
---

# 药品说明书智能识别与语音播报系统

基于 OpenVINO 本地推理，通过 PaddleOCR-VL（OCR 文字识别）→ Qwen3-VL（关键信息提取）→ Qwen3-TTS（语音合成播报）三步管线，解决药品说明书字体太小、老年人看不清读不懂的问题。

## 系统架构

```
药品说明书图片
      │
      ▼
[1] 加载图片
      │
      ▼
[2] 图片分割（可选，NxN 网格 + 重叠）
      │
      ▼
[3] OCR 文字识别 (PaddleOCR-VL) ──→ 原始 OCR 文本
      │  （释放 OCR 模型）
      ▼
[4] VLM 信息提取 (Qwen3-VL) ──→ 结构化关键信息
      │  （释放 VLM 模型）
      ▼
[5] TTS 语音合成 (Qwen3-TTS) ──→ 语音音频
      │  （释放 TTS 模型）
      ▼
输出: {ocr_text, extracted_info, audio}
```

三个模型（OCR + VLM + TTS）同时加载需大量内存。`ModelManager` 采用**懒加载**策略：每个模型按需加载、用完即释放，以时间换空间。

## 项目结构

项目代码位于 `medical_pipeline/` 目录：

```
medical_pipeline/
├── gradio_helper.py              # 管线主逻辑 & Gradio 界面（含 ModelManager、drug_ocr_pipeline、make_demo）
├── ov_paddleocr_vl.py            # PaddleOCR-VL OpenVINO 推理封装
├── image_processing_paddleocr_vl.py  # PaddleOCR-VL 图像预处理器
├── qwen_3_tts_helper.py          # Qwen3-TTS OpenVINO 推理封装
├── notebook_utils.py             # Jupyter 工具函数（设备选择 widget 等）
├── medical_pipeline.ipynb        # 主入口 Notebook
├── requirements.txt              # Python 依赖
└── resource/                     # 示例药品说明书图片
    ├── 1.jpg
    └── 2.jpg
```

## 核心模型

| 模型 | 用途 | ModelScope ID | 精度 |
|------|------|---------------|------|
| PaddleOCR-VL-1.5 | OCR 文字识别 | `megemini/PaddleOCR-VL-1.5-OpenVINO` | INT8 |
| Qwen3-VL-4B-Instruct | 多模态信息提取 | `snake7gun/Qwen3-VL-4B-Instruct-int4-ov` | INT4 |
| Qwen3-TTS-CustomVoice-0.6B | 语音合成 | `snake7gun/Qwen3-TTS-CustomVoice-0.6B-fp16-ov` | FP16 |

所有模型均为预转换的 OpenVINO IR 格式（`.xml` + `.bin`），从 ModelScope 下载后直接加载推理。

## 第一步：下载代码并安装依赖

先从 GitHub 克隆项目代码，再安装依赖：

```bash
git clone https://github.com/megemini/medical_pipeline.git
cd medical_pipeline
pip install -r requirements.txt
```

如果目录已存在则跳过克隆：

```python
import os
if not os.path.exists('gradio_helper.py') or not os.path.exists('requirements.txt'):
    os.system('git clone https://github.com/megemini/medical_pipeline.git')
    os.chdir('medical_pipeline')
```

核心依赖说明：

```
openvino>=2025.4              # OpenVINO 推理引擎
nncf>=2.15.0                  # 神经网络压缩框架
torch==2.8 (CPU)              # PyTorch CPU 版
optimum-intel                 # HuggingFace + OpenVINO 集成
transformers==4.57.6          # 模型加载
gradio==6.9.0                 # Web UI
modelscope                    # 模型下载
scipy                         # 音频文件写入 (WAV)
numpy<2.0                     # 数值计算
```

Linux 上需确保以下系统库已安装：

```bash
# Ubuntu / Debian
sudo apt-get install -y python3-dev python3-venv libgl1-mesa-glx libglib2.0-0
```

## 第二步：下载模型

三个模型均从 ModelScope 下载，已转换为 OpenVINO 格式。如果模型目录已存在则跳过下载。

```python
from pathlib import Path
from modelscope import snapshot_download

# --- OCR 模型 ---
ocr_model_dir = Path("PaddleOCR-VL-1.5-OpenVINO")
if not ocr_model_dir.exists():
    snapshot_download("megemini/PaddleOCR-VL-1.5-OpenVINO", local_dir=str(ocr_model_dir))

# --- VLM 模型 ---
vlm_model_dir = Path("Qwen3-VL-4B-Instruct-int4-ov")
if not vlm_model_dir.exists():
    snapshot_download("snake7gun/Qwen3-VL-4B-Instruct-int4-ov", local_dir=str(vlm_model_dir))

# --- TTS 模型 ---
tts_model_dir = Path("Qwen3-TTS-CustomVoice-0.6B-fp16-ov")
if not tts_model_dir.exists():
    snapshot_download("snake7gun/Qwen3-TTS-CustomVoice-0.6B-fp16-ov", local_dir=str(tts_model_dir))
```

## 第三步：运行管线

### 方式 A：Python 脚本（推荐）

使用 `ModelManager` 进行懒加载，模型按需加载、用完即释放：

```python
from gradio_helper import ModelManager, drug_ocr_pipeline

mgr = ModelManager(
    ocr_model_dir="PaddleOCR-VL-1.5-OpenVINO",
    vlm_model_dir="Qwen3-VL-4B-Instruct-int4-ov",
    tts_model_dir="Qwen3-TTS-CustomVoice-0.6B-fp16-ov",
    device="AUTO",
)

result = drug_ocr_pipeline(mgr, image_path="resource/1.jpg")
print(result["ocr_text"])
print(result["extracted_info"])
# result["audio"] -> (sample_rate, wav_data) 或 None
```

### 方式 B：分步调用

```python
from gradio_helper import ModelManager, split_image, ocr_recognize, vlm_extract_info, tts_synthesize
from PIL import Image

mgr = ModelManager(
    ocr_model_dir="PaddleOCR-VL-1.5-OpenVINO",
    vlm_model_dir="Qwen3-VL-4B-Instruct-int4-ov",
    tts_model_dir="Qwen3-TTS-CustomVoice-0.6B-fp16-ov",
    device="AUTO",
)

# Step 1: 加载图片
image = Image.open("resource/1.jpg").convert("RGB")

# Step 2: 图片分割（可选）
sub_images = split_image(image, num_splits=4, overlap_ratio=0.1)

# Step 3: OCR 文字识别
ocr_model = mgr.get_ocr_model()
ocr_text = ocr_recognize(ocr_model, image, max_new_tokens=5120)
mgr.release_ocr()  # 释放 OCR 模型内存

# Step 4: 大模型信息提取
vlm_model, vlm_processor = mgr.get_vlm_model()
extracted_info = vlm_extract_info(vlm_model, vlm_processor, image, ocr_text)
mgr.release_vlm()  # 释放 VLM 模型内存

# Step 5: 语音合成
tts_model = mgr.get_tts_model()
wav_data, sr = tts_synthesize(tts_model, extracted_info, speaker="vivian", language="Chinese")
mgr.release_tts()  # 释放 TTS 模型内存
```

### 方式 C：Gradio Web 界面

```python
from gradio_helper import ModelManager, make_demo

mgr = ModelManager(
    ocr_model_dir="PaddleOCR-VL-1.5-OpenVINO",
    vlm_model_dir="Qwen3-VL-4B-Instruct-int4-ov",
    tts_model_dir="Qwen3-TTS-CustomVoice-0.6B-fp16-ov",
    device="AUTO",
)

demo = make_demo(mgr)
demo.launch(server_port=7860)
```

### 方式 D：Jupyter Notebook

打开 `medical_pipeline.ipynb`，按顺序执行所有单元格即可。

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `device` | `"AUTO"` | 推理设备（AUTO/CPU/GPU） |
| `enable_split` | `True` | 启用图片分割（文字太小时建议开启） |
| `num_splits` | `4` | 分割数量（4=2x2, 9=3x3, 16=4x4，必须是完全平方数） |
| `overlap_ratio` | `0.1` | 分割区域重叠比例（0~0.3） |
| `ocr_max_new_tokens` | `5120` | OCR 最大生成 token 数 |
| `vlm_max_new_tokens` | `1024` | VLM 最大生成 token 数 |
| `tts_max_new_tokens` | `2048` | TTS 最大生成 token 数 |
| `tts_speaker` | `"vivian"` | TTS 说话人（vivian/aiden/ryan/serena 等） |
| `tts_language` | `"Chinese"` | TTS 语言（Chinese/English/Japanese 等） |
| `tts_instruct` | `"用友好亲切的语气说话。"` | TTS 风格指令 |
| `release_between_steps` | `True` | 步骤间释放模型以节省内存 |

## 提取的关键信息

大模型会从药品说明书中提取并整理以下 5 项关键信息，用简洁通俗的语言重新表述：

1. 药品名称
2. 药品适应症（这个药治什么病）
3. 药品的用法与用量（怎么吃、吃多少）
4. 药品的禁忌（什么人不能吃、什么情况不能吃）
5. 药品的不良反应（吃药后可能出现的不舒服）

## 核心 API 参考

### ModelManager

```python
from gradio_helper import ModelManager

mgr = ModelManager(ocr_model_dir, vlm_model_dir, tts_model_dir, device="AUTO")

mgr.get_ocr_model()       # 懒加载 OCR 模型
mgr.get_vlm_model()       # 懒加载 VLM 模型，返回 (model, processor)
mgr.get_tts_model()       # 懒加载 TTS 模型

mgr.release_ocr()         # 释放 OCR 模型内存
mgr.release_vlm()         # 释放 VLM 模型内存
mgr.release_tts()         # 释放 TTS 模型内存
mgr.release_all()         # 释放所有模型
```

### drug_ocr_pipeline

```python
from gradio_helper import drug_ocr_pipeline

result = drug_ocr_pipeline(
    model_manager=mgr,
    image_path="resource/1.jpg",
    enable_split=True,
    num_splits=4,
    overlap_ratio=0.1,
    ocr_max_new_tokens=5120,
    vlm_max_new_tokens=1024,
    tts_max_new_tokens=2048,
    tts_speaker="vivian",
    tts_language="Chinese",
    tts_instruct="用友好亲切的语气说话。",
    release_between_steps=True,
)
# result["ocr_text"]       -> OCR 识别的原始文字
# result["extracted_info"] -> 大模型整理后的关键信息
# result["audio"]          -> (sample_rate, wav_data) 或 None
```

### 独立函数

```python
from gradio_helper import split_image, ocr_recognize, vlm_extract_info, tts_synthesize

split_image(image, num_splits=4, overlap_ratio=0.1)      # 图片分割
ocr_recognize(ocr_model, image, max_new_tokens=5120)      # OCR 识别
vlm_extract_info(vlm_model, vlm_processor, image, ocr_text, max_new_tokens=1024)  # 信息提取
tts_synthesize(tts_model, text, speaker="vivian", language="Chinese", instruct="...", max_new_tokens=2048)  # 语音合成
```

## 快速检查清单

1. 依赖已安装：`pip install -r requirements.txt`
2. 系统库已安装（Linux）：`ldconfig -p | grep libGL`
3. 模型目录存在且完整（含 `.xml` + `.bin` 文件）：
   - `PaddleOCR-VL-1.5-OpenVINO/`
   - `Qwen3-VL-4B-Instruct-int4-ov/`
   - `Qwen3-TTS-CustomVoice-0.6B-fp16-ov/`
4. OpenVINO 版本：`python -c "import openvino; print(openvino.__version__)"` 应 >= 2025.4
5. 可用设备：`python -c "import openvino as ov; print(ov.Core().available_devices)"`

## 常见错误排查

| 错误 | 原因 | 解决方法 |
|------|------|----------|
| `ModuleNotFoundError: ov_paddleocr_vl` | 工作目录不对 | `cd` 到 `medical_pipeline/` 目录后再运行 |
| `ModuleNotFoundError: qwen_3_tts_helper` | 工作目录不对 | 确认在 `medical_pipeline/` 目录下运行 |
| `ModuleNotFoundError: optimum` | optimum-intel 未安装 | 按 requirements.txt 安装 |
| `FileNotFoundError: model_dir` | 模型未下载 | 运行模型下载代码或手动 `snapshot_download` |
| Gradio 端口占用 | 7860 端口被占用 | `demo.launch(server_port=7861)` |
| OCR 识别文字不完整 | 图片文字太小 | 启用图片分割，增大 `num_splits` |
| TTS 合成失败 | 文本过长或含特殊字符 | `gradio_helper.clean_for_tts()` 会自动清理 |
| `ImportError: libGL.so.1` | 缺少 OpenGL 库 | `sudo apt-get install libgl1-mesa-glx` |
| 内存不足 | 多个模型同时加载 | 确保 `release_between_steps=True` |
