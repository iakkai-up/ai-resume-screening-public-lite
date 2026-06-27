import time
from dataclasses import dataclass

from openai import OpenAI
import streamlit as st


REQUEST_TIMEOUT_SECONDS = 45
MAX_ATTEMPTS = 2
TEST_TIMEOUT_SECONDS = 20
TEST_IMAGE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAFElEQVR4nGP4TyJgGNUwqmH4agAAr639H708R/EAAAAASUVORK5CYII="
OCR_PROMPT = """请从这些简历图片中提取可读文字。

要求：
1. 只输出简历中的文字内容，不要总结、评分或补充不存在的信息。
2. 尽量保留原有段落顺序。
3. 表格可以按“列1 | 列2 | 列3”的形式输出。
4. 如果图片不清晰或无法识别，请输出你能确认的文字；完全无法识别时输出空字符串。"""


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model_name: str


def _read_secret(name: str) -> str:
    """安全读取 Streamlit secrets；没有配置时返回空字符串。"""
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


def get_screening_config() -> LLMConfig:
    return LLMConfig(
        api_key=_read_secret("API_KEY"),
        base_url=_read_secret("BASE_URL"),
        model_name=_read_secret("MODEL_NAME"),
    )


def get_ocr_config() -> LLMConfig:
    return LLMConfig(
        api_key=_read_secret("OCR_API_KEY"),
        base_url=_read_secret("OCR_BASE_URL"),
        model_name=_read_secret("OCR_MODEL_NAME") or _read_secret("VISION_MODEL_NAME"),
    )


def _safe_error_message(exc: Exception, config: LLMConfig) -> str:
    message = str(exc)
    if config.api_key:
        message = message.replace(config.api_key, "[API_KEY已隐藏]")
    return message


def test_llm_connection(config: LLMConfig) -> tuple[bool, str]:
    if not config.api_key or not config.base_url or not config.model_name:
        return False, "筛选模型配置不完整，无法联网测试。"

    try:
        client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=TEST_TIMEOUT_SECONDS)
        client.chat.completions.create(
            model=config.model_name,
            messages=[
                {"role": "system", "content": "你只回复 OK。"},
                {"role": "user", "content": "连接测试，请回复 OK。"},
            ],
            max_tokens=8,
            temperature=0,
        )
    except Exception as exc:
        return False, f"筛选 API 测试失败：{_safe_error_message(exc, config)}"

    return True, "筛选 API 测试通过，当前模型可以响应文本请求。"


def test_vision_connection(config: LLMConfig) -> tuple[bool, str]:
    if not config.api_key or not config.base_url or not config.model_name:
        return False, "OCR 配置不完整，无法联网测试。"

    try:
        client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=TEST_TIMEOUT_SECONDS)
        client.chat.completions.create(
            model=config.model_name,
            messages=[
                {"role": "system", "content": "你只回复 OK。"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "连接测试，请回复 OK。"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{TEST_IMAGE_BASE64}",
                            },
                        },
                    ],
                },
            ],
            max_tokens=8,
            temperature=0,
        )
    except Exception as exc:
        return False, f"OCR API 测试失败：{_safe_error_message(exc, config)}"

    return True, "OCR API 测试通过，当前模型可以接受图片输入。"


def call_llm(prompt: str, config: LLMConfig | None = None) -> tuple[str, str | None]:
    """
    调用 OpenAI 兼容的大模型接口。

    配置位置：
    .streamlit/secrets.toml

    必填配置：
    API_KEY、BASE_URL、MODEL_NAME
    """
    config = config or get_screening_config()

    if not config.api_key or not config.base_url or not config.model_name:
        return "", "筛选模型配置缺失：请在侧边栏“模型与 API 设置”中填写 API Key、Base URL 和筛选模型，或启用Mock演示模式。"

    last_error = ""
    try:
        client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=REQUEST_TIMEOUT_SECONDS)
    except Exception as exc:
        return "", f"API客户端初始化失败：{_safe_error_message(exc, config)}"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=config.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "你是谨慎的HR辅助分析助手，只能基于岗位相关因素输出JSON。",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content or "", None
        except Exception as exc:
            last_error = _safe_error_message(exc, config)
            if attempt < MAX_ATTEMPTS:
                time.sleep(1)

    return "", f"API调用失败，已重试{MAX_ATTEMPTS}次：{last_error}"


def call_vision_ocr(images: list[dict[str, str]], config: LLMConfig | None = None) -> tuple[str, str | None]:
    """
    调用支持图片输入的 OpenAI 兼容模型，把图片简历转成原始文本。

    images 中每项需要包含：
    - mime_type: image/png、image/jpeg 等
    - base64: 图片内容的 base64 字符串
    """
    if not images:
        return "", "OCR识别失败：没有可识别的图片内容。"

    config = config or get_ocr_config()

    if not config.api_key or not config.base_url or not config.model_name:
        return "", "OCR配置缺失：请在侧边栏“模型与 API 设置”的 OCR 高级配置中填写 API Key、Base URL 和 OCR 模型。OCR 不会自动使用筛选模型。"

    message_content = [{"type": "text", "text": OCR_PROMPT}]
    for image in images:
        message_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image['mime_type']};base64,{image['base64']}",
                },
            }
        )

    last_error = ""
    try:
        client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=REQUEST_TIMEOUT_SECONDS)
    except Exception as exc:
        return "", f"OCR客户端初始化失败：{_safe_error_message(exc, config)}"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=config.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "你是谨慎的OCR助手，只提取图片中的文字，不做招聘判断。",
                    },
                    {"role": "user", "content": message_content},
                ],
                temperature=0,
            )
            text = (response.choices[0].message.content or "").strip()
            if not text:
                return "", "OCR未识别到文字内容。请换用更清晰的图片，或先人工转换为 TXT/DOCX 后再上传。"
            return text, None
        except Exception as exc:
            last_error = _safe_error_message(exc, config)
            if attempt < MAX_ATTEMPTS:
                time.sleep(1)

    return "", (
        f"OCR识别失败，已重试{MAX_ATTEMPTS}次：{last_error}。"
        "请确认 OCR_MODEL_NAME 或 VISION_MODEL_NAME 使用的是支持图片输入的模型。"
    )
