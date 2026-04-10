# Medical Pipeline - 药品说明书智能识别与语音播报系统

药品说明书文字细小、专业术语多，老年人阅读困难。本系统拍照即可自动识别文字、提取关键信息并语音播报，帮助老年人看清读懂药品说明书。

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

三个模型（OCR + VLM + TTS）同时加载需 6-7 GB 显存。`ModelManager` 采用**懒加载**策略：每个模型按需加载、用完即释放，以时间换空间。

## 提取的关键信息

1. 药品名称
2. 药品适应症（这个药治什么病）
3. 药品的用法与用量（怎么吃、吃多少）
4. 药品的禁忌（什么人不能吃、什么情况不能吃）
5. 药品的不良反应（吃药后可能出现的不舒服）

## 模型

| 模型 | 用途 | 来源 | 精度 |
|------|------|------|------|
| PaddleOCR-VL-1.5 | OCR 文字识别 | `megemini/PaddleOCR-VL-1.5-OpenVINO` | INT8 |
| Qwen3-VL-4B-Instruct | 多模态信息提取 | `snake7gun/Qwen3-VL-4B-Instruct-int4-ov` | INT4 |
| Qwen3-TTS-CustomVoice-0.6B | 语音合成 | `snake7gun/Qwen3-TTS-CustomVoice-0.6B-fp16-ov` | FP16 |

所有模型均为预转换的 OpenVINO IR 格式（`.xml` + `.bin`），从 ModelScope 下载后直接加载推理，无需在线转换。

## 项目结构

```
medical_pipeline/
├── gradio_helper.py              # 管线主逻辑 & Gradio 界面
├── ov_paddleocr_vl.py            # PaddleOCR-VL OpenVINO 推理封装
├── image_processing_paddleocr_vl.py  # PaddleOCR-VL 图像预处理器
├── qwen_3_tts_helper.py          # Qwen3-TTS OpenVINO 推理封装
├── notebook_utils.py             # Jupyter 工具函数
├── medical_pipeline.ipynb        # 主入口 Notebook
├── requirements.txt              # Python 依赖
├── .gitignore
└── resource/                     # 示例图片
    ├── 1.jpg
    └── 2.jpg
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载模型

通过 Notebook 中的模型下载单元格，或手动从 ModelScope 下载：

```python
from modelscope import snapshot_download

snapshot_download("megemini/PaddleOCR-VL-1.5-OpenVINO", local_dir="PaddleOCR-VL-1.5-OpenVINO")
snapshot_download("snake7gun/Qwen3-VL-4B-Instruct-int4-ov", local_dir="Qwen3-VL-4B-Instruct-int4-ov")
snapshot_download("snake7gun/Qwen3-TTS-CustomVoice-0.6B-fp16-ov", local_dir="Qwen3-TTS-CustomVoice-0.6B-fp16-ov")
```

### 3. 运行

**方式 A：Jupyter Notebook**

打开 `medical_pipeline.ipynb`，按顺序执行所有单元格。

**方式 B：Python 脚本**

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
```

**方式 C：Gradio Web 界面**

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

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_split` | `True` | 启用图片分割（文字太小时建议开启） |
| `num_splits` | `4` | 分割数量（4=2x2, 9=3x3, 16=4x4） |
| `overlap_ratio` | `0.1` | 分割区域重叠比例 |
| `ocr_max_new_tokens` | `5120` | OCR 最大生成 token 数 |
| `vlm_max_new_tokens` | `1024` | VLM 最大生成 token 数 |
| `tts_max_new_tokens` | `2048` | TTS 最大生成 token 数 |
| `tts_speaker` | `"vivian"` | TTS 说话人 |
| `tts_language` | `"Chinese"` | TTS 语言 |
| `tts_instruct` | `"用友好亲切的语气说话。"` | TTS 风格指令 |
| `release_between_steps` | `True` | 步骤间释放模型以节省内存 |

## 依赖

- **推理引擎**: OpenVINO, NNCF
- **模型框架**: PyTorch, Transformers, Optimum-Intel
- **图像处理**: Pillow, TorchVision
- **语音合成**: SciPy, Librosa
- **Web 界面**: Gradio
- **模型下载**: ModelScope / HuggingFace Hub

## License

Apache License 2.0
