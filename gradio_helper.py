"""
Gradio helper for Drug OCR Pipeline.
Provides a ModelManager for lazy model loading, and a make_demo function
to create the Gradio interface for drug instruction leaflet OCR recognition,
VLM extraction, and TTS playback.
"""

import gc
import logging
import re
import tempfile
import time
import math

import numpy as np
import gradio as gr
from PIL import Image
from scipy.io.wavfile import write as wav_write

logger = logging.getLogger("drug_ocr")


def clean_for_tts(text):
    """
    Clean text for TTS synthesis by removing content that cannot be
    properly synthesized into speech, such as emojis and markdown formatting.

    Args:
        text: Input text string

    Returns:
        str: Cleaned text suitable for TTS
    """
    # Remove emojis (Unicode ranges for common emojis)
    # NOTE: Must avoid ranges that overlap with CJK characters (U+4E00-U+9FFF)
    text = re.sub(
        r"[\U0001F600-\U0001F64F"  # emoticons
        r"\U0001F300-\U0001F5FF"   # symbols & pictographs
        r"\U0001F680-\U0001F6FF"   # transport & map
        r"\U0001F1E0-\U0001F1FF"   # flags
        r"\U00002702-\U000027B0"   # dingbats
        r"\U000024C2-\U0000324F"   # enclosed alphanumerics (stop before CJK)
        r"\U0001F200-\U0001F251"   # enclosed CJK supplement (above CJK range)
        r"\U0001F900-\U0001F9FF"   # supplemental symbols
        r"\U0001FA00-\U0001FA6F"   # chess symbols
        r"\U0001FA70-\U0001FAFF"   # symbols extended-A
        r"\U00002600-\U000026FF"   # misc symbols
        r"\U0000FE00-\U0000FE0F"   # variation selectors
        r"\U0000200D"              # zero-width joiner
        r"]+",
        "",
        text,
    )
    # Remove markdown code blocks (```...```)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    # Remove inline code (`...`) -> content
    text = re.sub(r"`([^`\n]+)`", r"\1", text)
    # Remove markdown headers (# ## ### etc.) at line start
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove markdown bold (**text**) -> text
    text = re.sub(r"\*\*([^*\n]+?)\*\*", r"\1", text)
    # Remove markdown bold (__text__) -> text
    text = re.sub(r"__([^_\n]+?)__", r"\1", text)
    # Remove markdown italic (*text*) -> text
    text = re.sub(r"\*([^*\n]+?)\*", r"\1", text)
    # Remove markdown italic (_text_) -> text (only when _ is at word boundary)
    text = re.sub(r"(?<!\w)_([^_\n]+?)_(?!\w)", r"\1", text)
    # Remove markdown links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove markdown images ![alt](url)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    # Remove markdown horizontal rules (---, ***, ___)
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Remove markdown bullet list markers (- , * , + ) at line start, keep content
    text = re.sub(r"^(\s*)[-*+]\s+", r"\1", text, flags=re.MULTILINE)
    # Remove markdown numbered list markers (1. 2. etc.) at line start, keep content
    text = re.sub(r"^(\s*)\d+\.\s+", r"\1", text, flags=re.MULTILINE)
    # Remove markdown table pipes
    text = re.sub(r"\|", " ", text)
    # Remove markdown table separator lines (---:---:---)
    text = re.sub(r"^[-: ]+$", "", text, flags=re.MULTILINE)
    # Collapse multiple blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    # Remove leading/trailing whitespace overall
    text = text.strip()
    return text


class ModelManager:
    """
    Lazy model manager that loads models on demand and releases them after use.

    Instead of loading all 3 models (OCR + VLM + TTS) into memory at once,
    this manager loads each model only when needed and optionally releases
    the previous model before loading the next one to save memory.

    Usage:
        mgr = ModelManager(
            ocr_model_dir="PaddleOCR-VL-1.5-OpenVINO",
            vlm_model_dir="Qwen3-VL-4B-Instruct-int4-ov",
            tts_model_dir="Qwen3-TTS-CustomVoice-0.6B-fp16-ov",
            device="AUTO",
        )

        ocr_model = mgr.get_ocr_model()   # OCR model loaded now
        vlm_model, vlm_proc = mgr.get_vlm_model()  # VLM model loaded now
        tts_model = mgr.get_tts_model()   # TTS model loaded now

        mgr.release_ocr()   # Free OCR model memory
        mgr.release_vlm()   # Free VLM model memory
        mgr.release_tts()   # Free TTS model memory
    """

    def __init__(
        self,
        ocr_model_dir,
        vlm_model_dir,
        tts_model_dir,
        device="AUTO",
    ):
        self.ocr_model_dir = str(ocr_model_dir)
        self.vlm_model_dir = str(vlm_model_dir)
        self.tts_model_dir = str(tts_model_dir)
        self.device = device

        # Lazy-loaded model instances
        self._ocr_model = None
        self._vlm_model = None
        self._vlm_processor = None
        self._tts_model = None

        # Shared OpenVINO core
        self._ov_core = None

    def _get_ov_core(self):
        if self._ov_core is None:
            import openvino as ov
            self._ov_core = ov.Core()
        return self._ov_core

    def get_ocr_model(self):
        """Get PaddleOCR-VL model, loading it lazily if not already loaded."""
        if self._ocr_model is None:
            logger.info("加载 OCR 模型 (PaddleOCR-VL)...")
            start = time.perf_counter()

            import openvino as ov
            from ov_paddleocr_vl import OVPaddleOCRVLForCausalLM

            core = self._get_ov_core()
            self._ocr_model = OVPaddleOCRVLForCausalLM(
                core=core,
                ov_model_path=self.ocr_model_dir,
                device=self.device,
                llm_int4_compress=False,
                llm_int8_compress=True,
                vision_int8_quant=False,
                llm_int8_quant=True,
                llm_infer_list=[],
                vision_infer=[],
            )

            elapsed = time.perf_counter() - start
            logger.info("OCR 模型加载完成, 耗时: %.2fs", elapsed)
        return self._ocr_model

    def get_vlm_model(self):
        """Get VLM model and processor, loading lazily if not already loaded."""
        if self._vlm_model is None or self._vlm_processor is None:
            logger.info("加载 VLM 模型 (Qwen3-VL)...")
            start = time.perf_counter()

            from optimum.intel.openvino import OVModelForVisualCausalLM
            from transformers import AutoProcessor

            self._vlm_model = OVModelForVisualCausalLM.from_pretrained(
                self.vlm_model_dir, device=self.device
            )
            self._vlm_processor = AutoProcessor.from_pretrained(
                self.vlm_model_dir,
                min_pixels=256 * 28 * 28,
                max_pixels=1280 * 28 * 28,
            )

            elapsed = time.perf_counter() - start
            logger.info("VLM 模型加载完成, 耗时: %.2fs", elapsed)
        return self._vlm_model, self._vlm_processor

    def get_tts_model(self):
        """Get TTS model, loading lazily if not already loaded."""
        if self._tts_model is None:
            logger.info("加载 TTS 模型 (Qwen3-TTS)...")
            start = time.perf_counter()

            from qwen_3_tts_helper import OVQwen3TTSModel

            self._tts_model = OVQwen3TTSModel.from_pretrained(
                model_dir=self.tts_model_dir,
                device=self.device,
            )

            elapsed = time.perf_counter() - start
            logger.info("TTS 模型加载完成, 耗时: %.2fs", elapsed)
        return self._tts_model

    def release_ocr(self):
        """Release OCR model to free memory."""
        if self._ocr_model is not None:
            logger.info("释放 OCR 模型内存...")
            del self._ocr_model
            self._ocr_model = None
            gc.collect()

    def release_vlm(self):
        """Release VLM model and processor to free memory."""
        if self._vlm_model is not None:
            logger.info("释放 VLM 模型内存...")
            del self._vlm_model
            del self._vlm_processor
            self._vlm_model = None
            self._vlm_processor = None
            gc.collect()

    def release_tts(self):
        """Release TTS model to free memory."""
        if self._tts_model is not None:
            logger.info("释放 TTS 模型内存...")
            del self._tts_model
            self._tts_model = None
            gc.collect()

    def release_all(self):
        """Release all models to free memory."""
        self.release_ocr()
        self.release_vlm()
        self.release_tts()

    def get_tts_speakers_and_languages(self):
        """Get supported TTS speakers and languages (loads TTS model if needed)."""
        try:
            tts = self.get_tts_model()
            return tts.get_supported_speakers(), tts.get_supported_languages()
        except Exception:
            return ["vivian"], ["Chinese"]


def split_image(image, num_splits=4, overlap_ratio=0.1):
    """
    Split an image into num_splits parts (NxN grid) with overlap.

    Args:
        image: PIL.Image object
        num_splits: Number of splits, must be a perfect square (e.g. 4=2x2, 9=3x3)
        overlap_ratio: Overlap ratio (0~1) between split regions

    Returns:
        List[PIL.Image]: List of split sub-images
    """
    grid_size = int(math.sqrt(num_splits))
    if grid_size * grid_size != num_splits:
        raise ValueError(f"num_splits must be a perfect square (e.g. 4, 9, 16), got: {num_splits}")

    w, h = image.size
    cell_w = w / grid_size
    cell_h = h / grid_size
    overlap_w = cell_w * overlap_ratio
    overlap_h = cell_h * overlap_ratio

    sub_images = []
    for row in range(grid_size):
        for col in range(grid_size):
            left = max(0, col * cell_w - overlap_w)
            upper = max(0, row * cell_h - overlap_h)
            right = min(w, (col + 1) * cell_w + overlap_w)
            lower = min(h, (row + 1) * cell_h + overlap_h)
            sub_img = image.crop((int(left), int(upper), int(right), int(lower)))
            sub_images.append(sub_img)

    return sub_images


def ocr_recognize(paddleocr_model, image, max_new_tokens=5120):
    """
    Use PaddleOCR-VL model to perform OCR on an image.

    Args:
        paddleocr_model: Loaded PaddleOCR-VL model
        image: PIL.Image object
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        str: OCR recognized text
    """
    logger.info("OCR 识别开始, 图片尺寸: %s, max_new_tokens: %d", image.size, max_new_tokens)
    start = time.perf_counter()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "OCR:"},
            ],
        }
    ]

    generation_config = {
        "bos_token_id": paddleocr_model.tokenizer.bos_token_id,
        "eos_token_id": paddleocr_model.tokenizer.eos_token_id,
        "pad_token_id": paddleocr_model.tokenizer.pad_token_id,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }

    response, _ = paddleocr_model.chat(messages=messages, generation_config=generation_config)

    elapsed = time.perf_counter() - start
    logger.info("OCR 识别完成, 耗时: %.2fs, 识别文字长度: %d", elapsed, len(response))
    return response


def vlm_extract_info(vlm_model, vlm_processor, image, ocr_text, max_new_tokens=1024):
    """
    Use Qwen3-VL multimodal model to extract key information from drug instruction leaflet.

    Args:
        vlm_model: Loaded VLM model
        vlm_processor: VLM processor
        image: PIL.Image object (drug instruction leaflet image)
        ocr_text: OCR recognized text
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        str: Extracted key information text
    """
    logger.info("VLM 信息提取开始, OCR 输入文字长度: %d", len(ocr_text))
    start = time.perf_counter()

    prompt = f"""以下是药品说明书的 OCR 识别结果，供参考：

{ocr_text}

请根据以上 OCR 识别结果和图片内容，提取并整理以下关键信息，用清晰易懂的语言重新表述，方便老年人阅读理解：

1. 药品名称
2. 药品适应症（这个药治什么病）
3. 药品的用法与用量（怎么吃、吃多少）
4. 药品的禁忌（什么人不能吃、什么情况不能吃）
5. 药品的不良反应（吃药后可能出现的不舒服）

要求：
- 只输出整理后的关键信息，不要重复或复述 OCR 原文
- 用简洁、通俗的语言回答，避免使用专业术语
- 不要使用表情符号、emoji
- 不要使用markdown格式符号（如#、**、-等），直接用纯文本输出
- 用自然流畅的口语化表达，方便语音播报
- 总字数控制在 {max_new_tokens} 字以内"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    logger.info("VLM 正在预处理输入...")
    inputs = vlm_processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    logger.info("VLM 输入 token 数: %d", inputs["input_ids"].shape[1])

    logger.info("VLM 正在生成回复 (max_new_tokens=%d)...", max_new_tokens)
    gen_start = time.perf_counter()

    from transformers import TextStreamer
    generated_ids = vlm_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        streamer=TextStreamer(vlm_processor.tokenizer, skip_prompt=True, skip_special_tokens=True),
    )

    gen_elapsed = time.perf_counter() - gen_start

    # Only decode the newly generated tokens (exclude the input prompt)
    input_len = inputs["input_ids"].shape[1]
    new_generated_ids = generated_ids[:, input_len:]
    result = vlm_processor.batch_decode(new_generated_ids, skip_special_tokens=True)[0]

    # Clean result for TTS compatibility (remove emojis, markdown, etc.)
    result = clean_for_tts(result)

    elapsed = time.perf_counter() - start
    logger.info("VLM 信息提取完成, 生成耗时: %.2fs, 总耗时: %.2fs, 结果长度: %d", gen_elapsed, elapsed, len(result))
    return result


def tts_synthesize(tts_model, text, speaker="vivian", language="Chinese", instruct="用友好亲切的语气说话。", max_new_tokens=2048):
    """
    Use Qwen3-TTS model to synthesize speech from text.

    Args:
        tts_model: Loaded TTS model
        text: Text to synthesize
        speaker: Speaker name
        language: Language
        instruct: Style instruction
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        tuple: (wav_data, sample_rate) audio data and sample rate
    """
    logger.info("TTS 语音合成开始, 说话人: %s, 语言: %s, 输入文字长度: %d", speaker, language, len(text))
    start = time.perf_counter()

    wavs, sr = tts_model.generate_custom_voice(
        text=text,
        speaker=speaker,
        language=language,
        instruct=instruct,
        non_streaming_mode=True,
        max_new_tokens=max_new_tokens,
    )

    elapsed = time.perf_counter() - start

    if wavs is not None:
        audio_duration = len(wavs[0]) / sr
        logger.info("TTS 语音合成完成, 耗时: %.2fs, 音频时长: %.2fs, 采样率: %d Hz", elapsed, audio_duration, sr)
        return wavs[0], sr

    logger.warning("TTS 语音合成失败, 耗时: %.2fs", elapsed)
    return None, None


def drug_ocr_pipeline(
    model_manager,
    image_path,
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
):
    """
    Drug instruction leaflet intelligent recognition and voice broadcast pipeline.

    Uses ModelManager for lazy model loading. Models are loaded on demand
    and optionally released between steps to save memory.

    Args:
        model_manager: ModelManager instance for lazy model loading
        image_path: Path to drug instruction leaflet image
        enable_split: Whether to enable image splitting
        num_splits: Number of image splits (must be a perfect square)
        overlap_ratio: Overlap ratio between split images
        ocr_max_new_tokens: Max new tokens for OCR generation
        vlm_max_new_tokens: Max new tokens for VLM generation
        tts_max_new_tokens: Max new tokens for TTS generation
        tts_speaker: TTS speaker
        tts_language: TTS language
        tts_instruct: TTS style instruction
        release_between_steps: If True, release each model after its step completes
                               to save memory (adds re-loading overhead on subsequent runs)

    Returns:
        dict: Result dictionary containing ocr_text, extracted_info, audio
    """
    pipeline_start = time.perf_counter()
    logger.info("=" * 60)
    logger.info("药品说明书识别管线启动")
    logger.info("  图片路径: %s", image_path)
    logger.info("  图片分割: %s (num_splits=%d, overlap=%.2f)", enable_split, num_splits, overlap_ratio)
    logger.info("  TTS 设置: speaker=%s, language=%s", tts_speaker, tts_language)
    logger.info("  步骤间释放模型: %s", release_between_steps)
    logger.info("=" * 60)

    result = {}

    # Step 1: Load image
    logger.info("[Step 1/5] 加载图片...")
    step_start = time.perf_counter()
    image = Image.open(image_path).convert("RGB")
    logger.info("[Step 1/5] 图片加载完成, 尺寸: %s, 耗时: %.2fs", image.size, time.perf_counter() - step_start)

    # Step 2: Image splitting (optional)
    if enable_split:
        logger.info("[Step 2/5] 图片分割 (num_splits=%d, overlap=%.2f)...", num_splits, overlap_ratio)
        step_start = time.perf_counter()
        sub_images = split_image(image, num_splits=num_splits, overlap_ratio=overlap_ratio)
        ocr_images = [image] + sub_images
        logger.info("[Step 2/5] 图片分割完成, 原始1张 + 分割%d张 = 共%d张, 耗时: %.2fs", len(sub_images), len(ocr_images), time.perf_counter() - step_start)
    else:
        logger.info("[Step 2/5] 跳过图片分割")
        ocr_images = [image]

    # Step 3: OCR text recognition (lazy load OCR model)
    logger.info("[Step 3/5] OCR 文字识别 (%d 张图片)...", len(ocr_images))
    step_start = time.perf_counter()
    ocr_model = model_manager.get_ocr_model()
    all_ocr_texts = []
    for i, img in enumerate(ocr_images):
        label = "原始图片" if i == 0 else f"分割图片 {i}/{len(ocr_images)-1}"
        logger.info("[Step 3/5]   识别 %s (%d/%d)...", label, i + 1, len(ocr_images))
        ocr_text = ocr_recognize(ocr_model, img, max_new_tokens=ocr_max_new_tokens)
        all_ocr_texts.append(ocr_text)

    combined_ocr_text = "\n\n".join(all_ocr_texts)
    result["ocr_text"] = combined_ocr_text
    logger.info("[Step 3/5] OCR 识别全部完成, 总文字长度: %d, 耗时: %.2fs", len(combined_ocr_text), time.perf_counter() - step_start)

    # Release OCR model to free memory before loading VLM
    if release_between_steps:
        model_manager.release_ocr()
        logger.info("[Step 3/5] OCR 模型已释放，为 VLM 模型腾出内存")

    # Step 4: VLM text extraction (lazy load VLM model)
    logger.info("[Step 4/5] VLM 大模型信息提取...")
    step_start = time.perf_counter()
    vlm_model, vlm_processor = model_manager.get_vlm_model()
    extracted_info = vlm_extract_info(vlm_model, vlm_processor, image, combined_ocr_text, max_new_tokens=vlm_max_new_tokens)
    result["extracted_info"] = extracted_info
    logger.info("[Step 4/5] VLM 信息提取完成, 结果长度: %d, 耗时: %.2fs", len(extracted_info), time.perf_counter() - step_start)

    # Release VLM model to free memory before loading TTS
    if release_between_steps:
        model_manager.release_vlm()
        logger.info("[Step 4/5] VLM 模型已释放，为 TTS 模型腾出内存")

    # Step 5: TTS synthesis (lazy load TTS model)
    logger.info("[Step 5/5] TTS 语音合成...")
    step_start = time.perf_counter()
    tts_model = model_manager.get_tts_model()
    wav_data, sr = tts_synthesize(
        tts_model, extracted_info,
        speaker=tts_speaker,
        language=tts_language,
        instruct=tts_instruct,
        max_new_tokens=tts_max_new_tokens,
    )

    if wav_data is not None:
        result["audio"] = (sr, wav_data)
        audio_duration = len(wav_data) / sr
        logger.info("[Step 5/5] TTS 语音合成完成, 音频时长: %.2fs, 耗时: %.2fs", audio_duration, time.perf_counter() - step_start)
    else:
        result["audio"] = None
        logger.warning("[Step 5/5] TTS 语音合成失败")

    # Release TTS model
    if release_between_steps:
        model_manager.release_tts()
        logger.info("[Step 5/5] TTS 模型已释放")

    pipeline_elapsed = time.perf_counter() - pipeline_start
    logger.info("=" * 60)
    logger.info("管线执行完成, 总耗时: %.2fs", pipeline_elapsed)
    logger.info("=" * 60)

    return result


def make_demo(model_manager, ocr_max_new_tokens=5120, vlm_max_new_tokens=1024, tts_max_new_tokens=2048):
    """
    Create Gradio demo for Drug OCR Pipeline.

    Args:
        model_manager: ModelManager instance for lazy model loading
        ocr_max_new_tokens: Max new tokens for OCR generation
        vlm_max_new_tokens: Max new tokens for VLM generation
        tts_max_new_tokens: Max new tokens for TTS generation

    Returns:
        Gradio Blocks demo
    """

    def gradio_pipeline(
        image_input,
        enable_split,
        num_splits,
        overlap_ratio,
        ocr_max_tokens,
        vlm_max_tokens,
        tts_max_tokens,
        tts_speaker,
        tts_language,
        tts_instruct,
        progress=gr.Progress(track_tqdm=True),
    ):
        """Gradio interface main processing function"""
        if image_input is None:
            return "请上传药品说明书图片", "", None

        # Convert uploaded image to PIL Image
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = Image.fromarray(image_input).convert("RGB") if not isinstance(image_input, Image.Image) else image_input

        # Save as temp file for pipeline
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name

        try:
            result = drug_ocr_pipeline(
                model_manager=model_manager,
                image_path=tmp_path,
                enable_split=enable_split,
                num_splits=int(num_splits),
                overlap_ratio=overlap_ratio,
                ocr_max_new_tokens=int(ocr_max_tokens),
                vlm_max_new_tokens=int(vlm_max_tokens),
                tts_max_new_tokens=int(tts_max_tokens),
                tts_speaker=tts_speaker,
                tts_language=tts_language,
                tts_instruct=tts_instruct,
            )

            ocr_text = result["ocr_text"]
            extracted_info = result["extracted_info"]

            # Save audio as temp file
            audio_path = None
            if result["audio"] is not None:
                sr, wav_data = result["audio"]
                audio_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                wav_write(audio_tmp.name, sr, wav_data.astype(np.float32))
                audio_path = audio_tmp.name

            return ocr_text, extracted_info, audio_path
        finally:
            import os
            os.unlink(tmp_path)

    # Get TTS supported speakers and languages (lazy load TTS model)
    supported_speakers, supported_languages = model_manager.get_tts_speakers_and_languages()
    # Release TTS model after getting metadata (we only needed it for speaker/language lists)
    model_manager.release_tts()

    with gr.Blocks(title="药品说明书智能识别与语音播报") as demo:
        gr.Markdown("# 药品说明书智能识别与语音播报系统")
        gr.Markdown("上传药品说明书图片，系统将自动识别文字、提取关键信息并语音播报，帮助老年人看清读懂药品说明书。")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="药品说明书图片", type="filepath")

                with gr.Accordion("图片分割设置", open=True):
                    enable_split = gr.Checkbox(value=True, label="启用图片分割（文字太小时建议开启）")
                    num_splits = gr.Dropdown(choices=[4, 9, 16], value=4, label="分割数量")
                    overlap_ratio = gr.Slider(minimum=0.0, maximum=0.3, value=0.1, step=0.05, label="重叠比例")

                with gr.Accordion("生成参数设置", open=True):
                    ocr_max_tokens = gr.Slider(minimum=1024, maximum=8192, value=ocr_max_new_tokens, step=512, label="OCR 最大生成 token 数")
                    vlm_max_tokens = gr.Slider(minimum=256, maximum=4096, value=vlm_max_new_tokens, step=128, label="VLM 最大生成 token 数")
                    tts_max_tokens = gr.Slider(minimum=512, maximum=4096, value=tts_max_new_tokens, step=128, label="TTS 最大生成 token 数")

                with gr.Accordion("语音合成设置", open=True):
                    tts_speaker = gr.Dropdown(choices=supported_speakers, value="vivian", label="说话人")
                    tts_language = gr.Dropdown(choices=supported_languages, value="Chinese", label="语言")
                    tts_instruct = gr.Textbox(value="用友好亲切的语气说话。", label="风格指令")

                run_btn = gr.Button("开始识别", variant="primary")

            with gr.Column(scale=1):
                ocr_output = gr.Textbox(label="OCR 识别结果", lines=10, max_lines=20)
                info_output = gr.Textbox(label="关键信息整理", lines=15, max_lines=30)
                audio_output = gr.Audio(label="语音播报", type="filepath")

        run_btn.click(
            fn=gradio_pipeline,
            inputs=[
                image_input,
                enable_split,
                num_splits,
                overlap_ratio,
                ocr_max_tokens,
                vlm_max_tokens,
                tts_max_tokens,
                tts_speaker,
                tts_language,
                tts_instruct,
            ],
            outputs=[ocr_output, info_output, audio_output],
        )

    return demo
