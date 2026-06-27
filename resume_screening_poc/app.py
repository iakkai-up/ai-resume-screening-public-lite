from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import html
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from services.exporter import results_to_csv
from services.file_parser import (
    get_resume_file_extension,
    is_supported_resume_file,
    parse_resume_file,
)
from services.llm_client import (
    LLMConfig,
    call_llm,
    get_ocr_config,
    get_screening_config,
    test_llm_connection,
    test_vision_connection,
)
from services.scorer import (
    AI_DISCLAIMER,
    get_mock_results,
    make_error_result,
    normalize_result,
    safe_parse_json,
)


BASE_DIR = Path(__file__).parent
PROMPT_DIR = BASE_DIR / "prompts"
SAMPLE_JD_PATH = BASE_DIR / "sample_data" / "sample_jd.txt"
DEFAULT_SCREENING_BASE_URL = "https://api.deepseek.com"
DEFAULT_SCREENING_MODEL = "deepseek-v4-flash"
DEFAULT_SCREENING_WORKERS = 5
CUSTOM_MODEL_LABEL = "自定义模型名"
SCREENING_MODEL_SELECT_KEY = "screening_model_select"
CUSTOM_SCREENING_MODEL_KEY = "custom_screening_model_name"
OCR_MODEL_SELECT_KEY = "ocr_model_select"
CUSTOM_OCR_MODEL_KEY = "custom_ocr_model_name"
SCREENING_CONNECTION_RESULT_KEY = "screening_connection_result"
OCR_CONNECTION_RESULT_KEY = "ocr_connection_result"
SCREENING_REQUESTED_KEY = "screening_requested"
SCREENING_IN_PROGRESS_KEY = "screening_in_progress"
RESUME_UPLOADER_VERSION_KEY = "resume_uploader_version"
REMOVED_UPLOAD_FILE_KEYS_KEY = "removed_upload_file_keys"


def load_prompt(file_name: str) -> str:
    """读取 prompts 目录下的提示词文件。"""
    return (PROMPT_DIR / file_name).read_text(encoding="utf-8")


def build_prompt(template_name: str, title: str, content: str) -> str:
    """把提示词模板和用户输入拼在一起，便于大模型理解任务。"""
    template = load_prompt(template_name)
    return f"{template}\n\n【{title}】\n{content}"


def load_sample_jd() -> str:
    """读取内置示例 JD，方便现场演示快速开始。"""
    return SAMPLE_JD_PATH.read_text(encoding="utf-8")


def _value_or_default(value: str, default: str) -> str:
    return value.strip() if value and value.strip() else default.strip()


def normalize_model_options(raw_options) -> list[str]:
    if raw_options is None:
        candidates = []
    elif isinstance(raw_options, str):
        candidates = raw_options.replace("\n", ",").split(",")
    elif isinstance(raw_options, (list, tuple)):
        candidates = raw_options
    else:
        candidates = []

    options = []
    seen = set()
    for item in candidates:
        model_name = str(item).strip()
        if model_name and model_name not in seen:
            options.append(model_name)
            seen.add(model_name)
    return options


def read_model_options(secret_name: str) -> list[str]:
    try:
        return normalize_model_options(st.secrets.get(secret_name, []))
    except Exception:
        return []


def build_model_options(default_model_name: str, secret_options: list[str]) -> list[str]:
    options = []
    seen = set()
    for model_name in [default_model_name, *secret_options]:
        cleaned = str(model_name).strip()
        if cleaned and cleaned not in seen:
            options.append(cleaned)
            seen.add(cleaned)
    return options


def show_connection_result(success: bool, message: str) -> None:
    api_name = "OCR API" if message.startswith("OCR") else "筛选 API"
    status_text = "已通过" if success else "未通过"
    if success and "图片输入" in message:
        detail = "图片输入可用"
    elif success:
        detail = "文本请求可用"
    else:
        detail = message.partition("：")[2].strip() or message

    status_class = "connection-status-ok" if success else "connection-status-error"
    separator = "，" if success else "："
    status_summary = f"{api_name} {status_text}{separator}{detail}"
    st.markdown(
        f"""
            <div class="connection-status {status_class}">
                <span class="connection-status-dot"></span>
                <span class="connection-status-copy">
                    <span class="connection-status-title">{html.escape(status_summary)}</span>
                </span>
            </div>
        """,
        unsafe_allow_html=True,
    )


def save_screening_connection_result(config: LLMConfig) -> None:
    st.session_state[SCREENING_CONNECTION_RESULT_KEY] = test_llm_connection(config)


def save_ocr_connection_result(config: LLMConfig) -> None:
    st.session_state[OCR_CONNECTION_RESULT_KEY] = test_vision_connection(config)


def clear_screening_run_flags() -> None:
    st.session_state[SCREENING_REQUESTED_KEY] = False
    st.session_state[SCREENING_IN_PROGRESS_KEY] = False


def render_model_select(
    label: str,
    default_model_name: str,
    model_options: list[str],
    select_key: str,
    custom_key: str,
    custom_label: str,
    custom_placeholder: str,
    disabled: bool = False,
) -> str:
    select_options = [*model_options, CUSTOM_MODEL_LABEL]
    default_index = model_options.index(default_model_name) if default_model_name in model_options else len(model_options)

    selected_model = st.selectbox(
        label,
        select_options,
        index=default_index,
        key=select_key,
        disabled=disabled,
    )
    if selected_model != CUSTOM_MODEL_LABEL:
        return selected_model

    return st.text_input(
        custom_label,
        value=default_model_name if default_model_name not in model_options else "",
        placeholder=custom_placeholder,
        key=custom_key,
        disabled=disabled,
    ).strip()


def resolve_ocr_config(
    ocr_api_key: str,
    ocr_base_url: str,
    ocr_model_name: str,
    default_ocr_config: LLMConfig,
) -> LLMConfig:
    return LLMConfig(
        api_key=_value_or_default(ocr_api_key, default_ocr_config.api_key),
        base_url=_value_or_default(ocr_base_url, default_ocr_config.base_url),
        model_name=_value_or_default(ocr_model_name, default_ocr_config.model_name),
    )


def render_model_api_settings(disabled: bool = False) -> tuple[LLMConfig, LLMConfig]:
    default_screening_config = get_screening_config()
    default_ocr_config = get_ocr_config()
    default_base_url = default_screening_config.base_url or DEFAULT_SCREENING_BASE_URL
    default_screening_model = default_screening_config.model_name or DEFAULT_SCREENING_MODEL
    screening_model_options = build_model_options(
        default_screening_model,
        read_model_options("MODEL_OPTIONS"),
    )
    ocr_model_options = build_model_options(
        default_ocr_config.model_name,
        read_model_options("OCR_MODEL_OPTIONS"),
    )

    with st.container(border=True):
        st.markdown('<div class="side-control-label">模型与 API 设置</div>', unsafe_allow_html=True)
        st.markdown('<div class="side-section-label">筛选模型 API</div>', unsafe_allow_html=True)
        screening_api_key = st.text_input(
            "筛选 API Key",
            type="password",
            placeholder="留空用默认 Key",
            disabled=disabled,
        )
        screening_base_url = st.text_input(
            "筛选 Base URL",
            value=default_base_url,
            placeholder="https://api.deepseek.com",
            disabled=disabled,
        )
        screening_model_name = render_model_select(
            "筛选模型名",
            default_screening_model,
            screening_model_options,
            SCREENING_MODEL_SELECT_KEY,
            CUSTOM_SCREENING_MODEL_KEY,
            "自定义筛选模型名",
            "例如 deepseek-v4-flash",
            disabled=disabled,
        )
        st.caption("筛选 Key、Base URL 和模型名都是当前会话临时设置；刷新或重启后清空。只有 secrets.toml 会持久保存。")
        if default_screening_config.api_key:
            st.caption("已读取到本地 API_KEY；仅表示配置存在，未验证可用性。")
        resolved_screening_api_key = _value_or_default(screening_api_key, default_screening_config.api_key)
        resolved_screening_base_url = _value_or_default(screening_base_url, default_base_url)
        screening_config = LLMConfig(
            api_key=resolved_screening_api_key,
            base_url=resolved_screening_base_url,
            model_name=screening_model_name,
        )
        test_screening_config = LLMConfig(
            api_key=resolved_screening_api_key,
            base_url=resolved_screening_base_url,
            model_name=screening_model_name,
        )
        st.button(
            "测试筛选 API",
            use_container_width=True,
            on_click=save_screening_connection_result,
            args=(test_screening_config,),
            disabled=disabled,
        )
        if SCREENING_CONNECTION_RESULT_KEY in st.session_state:
            success, message = st.session_state[SCREENING_CONNECTION_RESULT_KEY]
            show_connection_result(success, message)

        with st.expander("OCR 高级配置", expanded=OCR_CONNECTION_RESULT_KEY in st.session_state):
            st.markdown('<div class="side-section-label">OCR 模型 API</div>', unsafe_allow_html=True)
            ocr_api_key = st.text_input(
                "OCR API Key",
                type="password",
                placeholder="留空读取 OCR_API_KEY",
                disabled=disabled,
            )
            ocr_base_url = st.text_input(
                "OCR Base URL",
                value=default_ocr_config.base_url,
                placeholder="留空读取 OCR_BASE_URL",
                disabled=disabled,
            )
            ocr_model_name = render_model_select(
                "OCR 模型名",
                default_ocr_config.model_name,
                ocr_model_options,
                OCR_MODEL_SELECT_KEY,
                CUSTOM_OCR_MODEL_KEY,
                "自定义 OCR 模型名",
                "视觉模型名",
                disabled=disabled,
            )
            st.caption("OCR Key、Base URL 和模型名留空会读取 secrets 中的 OCR 配置；不会沿用筛选 API。")
            test_ocr_config = resolve_ocr_config(
                ocr_api_key,
                ocr_base_url,
                ocr_model_name,
                default_ocr_config,
            )
            st.button(
                "测试 OCR API",
                use_container_width=True,
                on_click=save_ocr_connection_result,
                args=(test_ocr_config,),
                disabled=disabled,
            )
            if OCR_CONNECTION_RESULT_KEY in st.session_state:
                success, message = st.session_state[OCR_CONNECTION_RESULT_KEY]
                show_connection_result(success, message)

    ocr_config = resolve_ocr_config(
        ocr_api_key,
        ocr_base_url,
        ocr_model_name,
        default_ocr_config,
    )
    return screening_config, ocr_config


def apply_page_style(dark_mode: bool) -> None:
    """为 Streamlit 页面注入轻量主题样式。"""
    if dark_mode:
        colors = {
            "bg": "#030302",
            "panel": "#0d0d0c",
            "panel_soft": "#151311",
            "sidebar": "#080807",
            "text": "#f2eee9",
            "muted": "#a49a92",
            "border": "rgba(255, 255, 255, 0.09)",
            "primary": "#c8734e",
            "primary_hover": "#d3835e",
            "primary_text": "#fff7ef",
            "field": "#151311",
            "grid_header": "#1a1c24",
            "grid_hover": "rgba(172, 177, 195, 0.1)",
            "toggle_off": "#1d1a17",
            "title_shadow": "0 12px 34px rgba(0, 0, 0, 0.36)",
            "shadow": "0 26px 80px rgba(0, 0, 0, 0.42)",
        }
    else:
        colors = {
            "bg": "#f7f3ee",
            "panel": "#fffaf5",
            "panel_soft": "#f1e7dd",
            "sidebar": "#fbf6f0",
            "text": "#241c17",
            "muted": "#7b6b60",
            "border": "rgba(75, 54, 40, 0.16)",
            "primary": "#c8734e",
            "primary_hover": "#b96543",
            "primary_text": "#fff7ef",
            "field": "#ffffff",
            "grid_header": "#fff7ef",
            "grid_hover": "rgba(200, 115, 78, 0.1)",
            "toggle_off": "#e8ddd3",
            "title_shadow": "0 10px 28px rgba(92, 57, 36, 0.13)",
            "shadow": "0 22px 64px rgba(92, 57, 36, 0.11)",
        }

    st.markdown(
        f"""
        <style>
            @font-face {{
                font-family: "ZCOOL XiaoWei Local";
                src: url("/app/static/fonts/ZCOOLXiaoWei-Regular.ttf?v=1") format("truetype");
                font-weight: 400;
                font-style: normal;
                font-display: swap;
            }}

            :root {{
                --app-bg: {colors["bg"]};
                --app-panel: {colors["panel"]};
                --app-panel-soft: {colors["panel_soft"]};
                --app-sidebar: {colors["sidebar"]};
                --app-text: {colors["text"]};
                --app-muted: {colors["muted"]};
                --app-border: {colors["border"]};
                --app-primary: {colors["primary"]};
                --app-primary-hover: {colors["primary_hover"]};
                --app-primary-text: {colors["primary_text"]};
                --app-field: {colors["field"]};
                --app-grid-header: {colors["grid_header"]};
                --app-grid-hover: {colors["grid_hover"]};
                --app-toggle-off: {colors["toggle_off"]};
                --app-title-shadow: {colors["title_shadow"]};
                --app-shadow: {colors["shadow"]};
                --app-title-font: "Source Han Serif SC Medium", "Source Han Serif SC", "Source Han Serif CN Medium", "Source Han Serif CN", "Noto Serif CJK SC", "Noto Serif SC", "Songti SC", "STSong", "SimSun", serif;
                --app-sidebar-heading-font: "SimSun", "宋体", "Songti SC", "STSong", serif;
                --app-accent-soft: color-mix(in srgb, var(--app-primary) 18%, var(--app-panel));
                --app-accent-softer: color-mix(in srgb, var(--app-primary) 10%, var(--app-panel));
            }}

            .stApp {{
                background:
                    radial-gradient(circle at 31% 8%, color-mix(in srgb, var(--app-primary) 11%, transparent), transparent 35rem),
                    radial-gradient(circle at 68% 90%, color-mix(in srgb, var(--app-primary) 8%, transparent), transparent 30rem),
                    var(--app-bg);
                color: var(--app-text);
            }}

            [data-testid="stHeader"] {{
                background: transparent;
            }}

            [data-testid="stExpandSidebarButton"],
            [data-testid="stSidebar"] [data-testid="stBaseButton-headerNoPadding"] {{
                background: color-mix(in srgb, var(--app-primary) 12%, var(--app-panel)) !important;
                border: 1px solid color-mix(in srgb, var(--app-primary) 34%, var(--app-border)) !important;
                border-radius: 8px !important;
                color: var(--app-text) !important;
            }}

            [data-testid="stExpandSidebarButton"] *,
            [data-testid="stSidebar"] [data-testid="stBaseButton-headerNoPadding"] * {{
                color: var(--app-text) !important;
                opacity: 1 !important;
            }}

            [data-testid="stExpandSidebarButton"]:hover,
            [data-testid="stSidebar"] [data-testid="stBaseButton-headerNoPadding"]:hover {{
                background: color-mix(in srgb, var(--app-primary) 20%, var(--app-panel)) !important;
                border-color: color-mix(in srgb, var(--app-primary) 52%, var(--app-border)) !important;
            }}

            [data-testid="stSidebar"] {{
                background: var(--app-sidebar);
                border-right: 1px solid var(--app-border);
            }}

            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] h3 {{
                color: var(--app-text) !important;
            }}

            [data-testid="stAppViewContainer"] > .main .block-container {{
                max-width: 1180px;
                padding-top: 1.7rem;
                padding-bottom: 3rem;
            }}

            .page-title {{
                margin: 0 0 1.5rem;
                color: color-mix(in srgb, var(--app-text) 92%, var(--app-primary));
                font-family: var(--app-title-font) !important;
                font-size: clamp(2.35rem, 5vw, 3.55rem);
                font-weight: 600;
                line-height: 1.12;
                letter-spacing: 0;
                text-align: center;
                text-wrap: balance;
                transform: translateY(-0.45rem);
                font-optical-sizing: auto;
                -webkit-font-smoothing: antialiased;
                text-shadow: var(--app-title-shadow);
            }}

            h2, h3,
            [data-testid="stMarkdownContainer"] p,
            [data-testid="stWidgetLabel"] label,
            [data-testid="stWidgetLabel"] p {{
                color: var(--app-text);
                letter-spacing: 0;
            }}

            [data-testid="stCaptionContainer"] p,
            .stMarkdown small {{
                color: var(--app-muted);
                font-size: 0.86rem;
                line-height: 1.5;
            }}

            .api-mode-caption {{
                margin: 0.95rem 0 0.55rem;
                padding-bottom: 0.2rem;
                color: var(--app-muted);
                font-size: 0.86rem;
                line-height: 1.5;
            }}

            .start-action-spacer {{
                height: 1.9rem;
            }}

            .upload-file-summary {{
                margin-top: 0.65rem;
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 0.75rem;
                flex-wrap: wrap;
                padding: 0.5rem 0.7rem;
                border: 1px solid color-mix(in srgb, var(--app-primary) 24%, var(--app-border));
                border-radius: 8px;
                background: color-mix(in srgb, var(--app-primary) 5%, var(--app-panel));
                color: var(--app-text);
                font-size: 0.9rem;
                font-weight: 650;
                line-height: 1.45;
            }}

            .upload-file-summary strong {{
                color: var(--app-primary);
                font-size: 1.02rem;
            }}

            .upload-file-summary-detail {{
                color: var(--app-muted);
                font-size: 0.84rem;
                font-weight: 600;
            }}

            .upload-file-summary-warning {{
                border-color: color-mix(in srgb, #d94a3a 38%, var(--app-border));
                background: color-mix(in srgb, #d94a3a 6%, var(--app-panel));
            }}

            .unsupported-file-warning {{
                margin-top: 0.65rem;
                padding: 0.55rem 0.75rem;
                border: 1px solid color-mix(in srgb, var(--app-primary) 30%, var(--app-border));
                border-radius: 8px;
                background: color-mix(in srgb, var(--app-primary) 6%, var(--app-panel));
                color: var(--app-muted);
                font-size: 0.88rem;
                line-height: 1.55;
            }}

            .unsupported-file-warning-title,
            .unsupported-file-warning-files {{
                color: var(--app-text);
            }}

            .unsupported-file-warning-files {{
                margin: 0.15rem 0;
            }}

            .upload-cleanup-action-spacer {{
                height: 0.55rem;
            }}

            [data-testid="stVerticalBlockBorderWrapper"] {{
                background: color-mix(in srgb, var(--app-panel) 92%, transparent);
                border: 1px solid var(--app-border);
                border-radius: 8px;
                box-shadow: var(--app-shadow);
            }}

            [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {{
                background: color-mix(in srgb, var(--app-panel) 82%, transparent);
                box-shadow: none;
            }}

            [data-testid="stHorizontalBlock"] {{
                align-items: center;
            }}

            .control-label {{
                margin-bottom: 0.35rem;
                color: var(--app-muted);
                font-size: 0.92rem;
                font-weight: 650;
                line-height: 1.2;
            }}

            .control-label-right {{
                text-align: right;
            }}

            .mode-switch-title {{
                margin-bottom: 0.35rem;
                color: var(--app-muted);
                font-size: 0.86rem;
                font-weight: 700;
                line-height: 1.2;
                text-align: center;
            }}

            .side-control-label,
            [data-testid="stSidebar"] .side-control-label {{
                margin-bottom: 0.45rem;
                color: var(--app-text);
                font-family: var(--app-sidebar-heading-font) !important;
                font-size: 1.02rem;
                font-weight: 700;
                line-height: 1.2;
            }}

            .side-section-label,
            [data-testid="stSidebar"] .side-section-label {{
                margin: 0.85rem 0 0.45rem;
                color: var(--app-text);
                font-family: var(--app-sidebar-heading-font) !important;
                font-size: 0.94rem;
                font-weight: 700;
                line-height: 1.2;
            }}

            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] .side-control-label,
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] .side-control-label *,
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] .side-section-label,
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] .side-section-label * {{
                font-family: var(--app-sidebar-heading-font) !important;
            }}

            [data-testid="stTextArea"] textarea,
            [data-testid="stTextInput"] input,
            [data-testid="stTextInput"] div[data-baseweb="input"],
            [data-testid="stTextInput"] div[data-baseweb="base-input"],
            [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"],
            [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
            [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
            [data-testid="stDataFrame"],
            .stCodeBlock {{
                background: var(--app-field) !important;
                border-color: var(--app-border) !important;
                color: var(--app-text) !important;
                border-radius: 8px !important;
            }}

            [data-testid="stTextInput"] input {{
                -webkit-text-fill-color: var(--app-text) !important;
            }}

            [data-testid="stTextInput"] input::placeholder {{
                color: color-mix(in srgb, var(--app-muted) 88%, transparent) !important;
                -webkit-text-fill-color: color-mix(in srgb, var(--app-muted) 88%, transparent) !important;
                opacity: 1 !important;
            }}

            [data-testid="stTextInput"] div[data-baseweb="input"] {{
                align-items: center !important;
                border: 1px solid var(--app-border) !important;
                box-shadow: none !important;
                display: flex !important;
                height: 2.5rem !important;
                overflow: hidden !important;
                padding: 0 !important;
            }}

            [data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {{
                border-color: var(--app-primary) !important;
                box-shadow: 0 0 0 1px var(--app-primary) !important;
            }}

            [data-testid="stTextInput"] div[data-baseweb="base-input"],
            [data-testid="stTextInput"] input {{
                background: transparent !important;
                border: 0 !important;
                border-radius: 0 !important;
                box-shadow: none !important;
                height: 100% !important;
                outline: 0 !important;
            }}

            [data-testid="stTextInput"] div[data-baseweb="base-input"] {{
                align-items: center !important;
                display: flex !important;
                flex: 1 1 auto !important;
                min-width: 0 !important;
                padding: 0 !important;
            }}

            [data-testid="stTextInput"] input {{
                flex: 1 1 auto !important;
                min-width: 0 !important;
            }}

            /* Streamlit adds focus shadows to nested password parts; keep the ring on the wrapper only. */
            [data-testid="stTextInput"] div[data-baseweb="base-input"]:not(#streamlit-text-input-shadow-reset),
            [data-testid="stTextInput"] div[data-baseweb="base-input"]:focus-within:not(#streamlit-text-input-shadow-reset),
            [data-testid="stTextInput"] input:not(#streamlit-text-input-shadow-reset),
            [data-testid="stTextInput"] input:focus:not(#streamlit-text-input-shadow-reset),
            [data-testid="stTextInput"] input:focus-visible:not(#streamlit-text-input-shadow-reset) {{
                box-shadow: none !important;
                outline: 0 !important;
            }}

            [data-testid="stTextInput"] button,
            [data-testid="stTextInput"] button:hover,
            [data-testid="stTextInput"] button:focus {{
                background: transparent !important;
                border: 0 !important;
                box-shadow: none !important;
                color: var(--app-text) !important;
                height: 100% !important;
                justify-content: center !important;
                margin: 0 !important;
                min-width: 2.25rem !important;
                outline: 0 !important;
                padding: 0 !important;
                width: 2.25rem !important;
            }}

            [data-testid="stTextInput"] svg,
            [data-testid="stTextInput"] svg * {{
                color: var(--app-text) !important;
                fill: currentColor !important;
                stroke: currentColor !important;
            }}

            [data-testid="stDataFrame"] [data-testid="stElementToolbarButtonContainer"] {{
                background: var(--app-field) !important;
                border: 1px solid var(--app-border) !important;
                box-shadow: 0 10px 28px rgba(0, 0, 0, 0.08) !important;
            }}

            [data-testid="stDataFrame"] [data-testid="stBaseButton-elementToolbar"],
            [data-testid="stDataFrame"] [data-testid="stElementToolbarButton"],
            [data-testid="stDataFrame"] [data-testid="stTooltipHoverTarget"] {{
                background: transparent !important;
                color: var(--app-text) !important;
            }}

            [data-testid="stDataFrame"] [data-testid="stElementToolbarButtonIcon"],
            [data-testid="stDataFrame"] [data-testid="stElementToolbarButtonIcon"] * {{
                color: var(--app-text) !important;
                fill: currentColor !important;
                stroke: currentColor !important;
            }}

            [data-testid="stDataFrameResizable"],
            .stDataFrameGlideDataEditor,
            [data-testid="data-grid-canvas"] {{
                background: var(--app-field) !important;
            }}

            .stDataFrameGlideDataEditor {{
                --gdg-accent-color: var(--app-primary) !important;
                --gdg-accent-fg: var(--app-primary-text) !important;
                --gdg-accent-light: var(--app-grid-hover) !important;
                --gdg-text-dark: var(--app-text) !important;
                --gdg-text-medium: color-mix(in srgb, var(--app-text) 82%, transparent) !important;
                --gdg-text-light: color-mix(in srgb, var(--app-text) 58%, transparent) !important;
                --gdg-text-bubble: color-mix(in srgb, var(--app-text) 72%, transparent) !important;
                --gdg-bg-icon-header: color-mix(in srgb, var(--app-text) 64%, transparent) !important;
                --gdg-fg-icon-header: var(--app-text) !important;
                --gdg-text-header: color-mix(in srgb, var(--app-text) 72%, transparent) !important;
                --gdg-text-group-header: color-mix(in srgb, var(--app-text) 72%, transparent) !important;
                --gdg-bg-group-header: var(--app-grid-header) !important;
                --gdg-bg-group-header-hovered: var(--app-grid-hover) !important;
                --gdg-text-header-selected: var(--app-text) !important;
                --gdg-bg-cell: var(--app-field) !important;
                --gdg-bg-cell-medium: var(--app-field) !important;
                --gdg-bg-header: var(--app-grid-header) !important;
                --gdg-bg-header-has-focus: var(--app-grid-hover) !important;
                --gdg-bg-header-hovered: var(--app-grid-hover) !important;
                --gdg-bg-bubble: var(--app-grid-header) !important;
                --gdg-bg-bubble-selected: var(--app-grid-hover) !important;
                --gdg-bg-search-result: var(--app-grid-hover) !important;
                --gdg-border-color: var(--app-border) !important;
                --gdg-horizontal-border-color: var(--app-border) !important;
                --gdg-drilldown-border: var(--app-border) !important;
                --gdg-link-color: var(--app-primary) !important;
                --gdg-resize-indicator-color: var(--app-primary) !important;
            }}

            [data-testid="stDataFrame"] *,
            div[data-baseweb="popover"] * {{
                scrollbar-color: color-mix(in srgb, var(--app-muted) 52%, var(--app-field)) var(--app-field);
            }}

            [data-testid="stDataFrame"] *::-webkit-scrollbar,
            div[data-baseweb="popover"] *::-webkit-scrollbar {{
                width: 10px;
                height: 10px;
            }}

            [data-testid="stDataFrame"] *::-webkit-scrollbar-track,
            div[data-baseweb="popover"] *::-webkit-scrollbar-track {{
                background: var(--app-field);
            }}

            [data-testid="stDataFrame"] *::-webkit-scrollbar-thumb,
            div[data-baseweb="popover"] *::-webkit-scrollbar-thumb {{
                background: color-mix(in srgb, var(--app-muted) 52%, var(--app-field));
                border: 2px solid var(--app-field);
                border-radius: 999px;
            }}

            .results-table-wrap {{
                width: 100%;
                overflow-x: auto;
                background: var(--app-field);
                border: 1px solid var(--app-border);
                border-radius: 8px;
            }}

            .results-table {{
                width: max-content;
                min-width: 100%;
                border-collapse: collapse;
                color: var(--app-text);
                font-size: 0.92rem;
                line-height: 1.45;
            }}

            .results-table th,
            .results-table td {{
                max-width: 18rem;
                padding: 0.65rem 0.75rem;
                border-bottom: 1px solid var(--app-border);
                text-align: left;
                vertical-align: top;
                white-space: normal;
            }}

            .results-table th {{
                background: var(--app-grid-header);
                color: color-mix(in srgb, var(--app-text) 82%, transparent);
                font-weight: 700;
            }}

            .results-table tr:last-child td {{
                border-bottom: 0;
            }}

            .results-table tbody tr:hover td {{
                background: var(--app-grid-hover);
            }}

            .results-download-action-spacer {{
                height: 0.8rem;
            }}

            [data-testid="stSelectbox"] [data-baseweb="select"] > div,
            div[data-baseweb="tooltip"],
            div[data-baseweb="tooltip"] > div,
            [role="tooltip"],
            div[data-baseweb="popover"],
            div[data-baseweb="popover"] > div,
            div[data-baseweb="popover"] ul,
            div[data-baseweb="popover"] ul[role="listbox"],
            div[data-baseweb="popover"] [data-testid="stSelectboxVirtualDropdown"],
            div[data-baseweb="popover"] [data-testid="stSelectboxVirtualDropdown"] > div,
            div[data-baseweb="popover"] li[role="option"] {{
                background: var(--app-field) !important;
                border-color: var(--app-border) !important;
                color: var(--app-text) !important;
            }}

            [data-testid="stSelectbox"] [data-baseweb="select"] *,
            [data-testid="stSelectbox"] [data-testid="stWidgetLabel"] *,
            div[data-baseweb="tooltip"],
            div[data-baseweb="tooltip"] *,
            [role="tooltip"],
            [role="tooltip"] *,
            div[data-baseweb="popover"] li[role="option"],
            div[data-baseweb="popover"] li[role="option"] * {{
                color: var(--app-text) !important;
            }}

            div[data-baseweb="tooltip"],
            div[data-baseweb="tooltip"] > div,
            [role="tooltip"] {{
                border: 1px solid var(--app-border) !important;
                border-radius: 8px !important;
                box-shadow: 0 14px 36px rgba(0, 0, 0, 0.12) !important;
            }}

            [data-testid="stSelectbox"] svg,
            [data-testid="stSelectbox"] svg * {{
                color: var(--app-text) !important;
                fill: currentColor !important;
                stroke: currentColor !important;
            }}

            div[data-baseweb="popover"] li[role="option"]:hover,
            div[data-baseweb="popover"] li[role="option"]:focus,
            div[data-baseweb="popover"] li[role="option"]:focus-visible,
            div[data-baseweb="popover"] li[role="option"][data-highlighted="true"],
            div[data-baseweb="popover"] li[role="option"][aria-selected="true"] {{
                background: var(--app-grid-hover) !important;
            }}

            [data-testid="stTextArea"] textarea::placeholder {{
                color: color-mix(in srgb, var(--app-muted) 88%, transparent) !important;
                opacity: 1 !important;
            }}

            [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {{
                background: var(--app-accent-softer) !important;
                border: 2px solid color-mix(in srgb, var(--app-primary) 72%, var(--app-border)) !important;
                box-shadow: none !important;
                outline: none !important;
                overflow: visible !important;
                position: relative !important;
                min-height: 5.25rem !important;
                padding: 0.75rem 1rem !important;
                align-items: center !important;
                gap: 0.75rem !important;
                flex-wrap: wrap !important;
            }}

            [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]:hover {{
                background: color-mix(in srgb, var(--app-primary) 8%, var(--app-panel)) !important;
                border-color: color-mix(in srgb, var(--app-primary) 30%, var(--app-border)) !important;
            }}

            [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]:hover button[kind="secondary"] {{
                background: color-mix(in srgb, var(--app-primary) 20%, var(--app-panel)) !important;
                border-color: color-mix(in srgb, var(--app-primary) 54%, var(--app-border)) !important;
            }}

            [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]:focus,
            [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]:focus-visible,
            [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]:focus-within {{
                background: color-mix(in srgb, var(--app-primary) 8%, var(--app-panel)) !important;
                border-color: color-mix(in srgb, var(--app-primary) 44%, var(--app-border)) !important;
                box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--app-primary) 34%, var(--app-border)) !important;
                outline: none !important;
            }}

            [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] > div:not([data-testid]) {{
                position: relative !important;
                z-index: 2 !important;
                max-width: 100% !important;
            }}

            [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] > div:not([data-testid]):not(:has(button)) {{
                position: absolute !important;
                inset: 0.75rem !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                width: auto !important;
                height: auto !important;
                max-width: none !important;
                border: 2px dashed color-mix(in srgb, var(--app-primary) 72%, var(--app-border)) !important;
                border-radius: 8px !important;
                background: color-mix(in srgb, var(--app-primary) 12%, var(--app-panel)) !important;
                color: transparent !important;
                font-size: 0 !important;
                pointer-events: none !important;
                z-index: 4 !important;
            }}

            [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] > div:not([data-testid]):not(:has(button))::before {{
                content: "松开鼠标上传文件";
                color: var(--app-text);
                font-size: 1rem;
                font-weight: 700;
                line-height: 1.2;
            }}

            [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] > div:not([data-testid]):not(:has(button)) * {{
                color: transparent !important;
                font-size: 0 !important;
            }}

            [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] > span,
            [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"],
            [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] > div {{
                background: transparent !important;
                color: var(--app-text) !important;
                position: relative;
                z-index: 2;
            }}

            [data-testid="stFileUploader"] [data-testid="stFileChips"] {{
                display: flex !important;
                flex-wrap: wrap !important;
                align-items: center !important;
                gap: 0.6rem !important;
                width: 100% !important;
                background: transparent !important;
            }}

            [data-testid="stFileUploader"] [data-testid="stFileChips"] > div,
            [data-testid="stFileUploader"] [data-testid="stFileChips"] > div > div {{
                background: transparent !important;
            }}

            [data-testid="stFileUploader"] [data-testid="stFileChip"],
            [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {{
                position: relative !important;
                flex: 0 1 18rem !important;
                width: auto !important;
                min-width: 12rem !important;
                max-width: min(100%, 18rem) !important;
                padding: 0.42rem 0.55rem !important;
                background: var(--app-panel) !important;
                border: 1px solid var(--app-border) !important;
                border-radius: 8px !important;
                color: var(--app-text) !important;
                box-shadow: 0 8px 22px rgba(0, 0, 0, 0.05) !important;
                overflow: hidden !important;
            }}

            [data-testid="stFileUploader"] [data-testid="stFileChip"] div,
            [data-testid="stFileUploader"] [data-testid="stFileChip"] span,
            [data-testid="stFileUploader"] [data-testid="stFileChip"] small,
            [data-testid="stFileUploader"] [data-testid="stFileChip"] p,
            [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] div,
            [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] span,
            [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] small,
            [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] p {{
                background: transparent !important;
                color: var(--app-text) !important;
            }}

            [data-testid="stFileUploader"] [data-testid="stFileChipName"] {{
                max-width: 100% !important;
            }}

            [data-testid="stFileUploader"] [data-testid="stFileChip"]:has(button[aria-label^="Remove "]:not([aria-label$=".pdf" i]):not([aria-label$=".docx" i]):not([aria-label$=".txt" i]):not([aria-label$=".png" i]):not([aria-label$=".jpg" i]):not([aria-label$=".jpeg" i])),
            [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"]:has(button[aria-label^="Remove "]:not([aria-label$=".pdf" i]):not([aria-label$=".docx" i]):not([aria-label$=".txt" i]):not([aria-label$=".png" i]):not([aria-label$=".jpg" i]):not([aria-label$=".jpeg" i])) {{
                padding-left: 2.85rem !important;
                border-color: #d94a3a !important;
                background: color-mix(in srgb, #d94a3a 7%, var(--app-panel)) !important;
            }}

            [data-testid="stFileUploader"] [data-testid="stFileChip"]:has(button[aria-label^="Remove "]:not([aria-label$=".pdf" i]):not([aria-label$=".docx" i]):not([aria-label$=".txt" i]):not([aria-label$=".png" i]):not([aria-label$=".jpg" i]):not([aria-label$=".jpeg" i]))::before,
            [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"]:has(button[aria-label^="Remove "]:not([aria-label$=".pdf" i]):not([aria-label$=".docx" i]):not([aria-label$=".txt" i]):not([aria-label$=".png" i]):not([aria-label$=".jpg" i]):not([aria-label$=".jpeg" i]))::before {{
                content: "!";
                position: absolute;
                left: 0.55rem;
                top: 50%;
                z-index: 6;
                display: flex;
                align-items: center;
                justify-content: center;
                width: 1.65rem;
                height: 1.65rem;
                border: 2px solid #d94a3a;
                border-radius: 999px;
                background: color-mix(in srgb, #d94a3a 9%, var(--app-panel));
                color: #d94a3a;
                font-size: 1.12rem;
                font-weight: 800;
                line-height: 1;
                transform: translateY(-50%);
            }}

            [data-testid="stFileUploader"] button[aria-label="Add files"] {{
                flex: 0 0 auto !important;
                width: 2.25rem !important;
                min-width: 2.25rem !important;
                height: 2.25rem !important;
                min-height: 2.25rem !important;
                padding: 0 !important;
                border: 1px solid var(--app-border) !important;
                border-radius: 8px !important;
                background: var(--app-panel) !important;
                color: var(--app-text) !important;
                box-shadow: 0 8px 22px rgba(0, 0, 0, 0.05) !important;
            }}

            [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button[kind="secondary"] {{
                background: var(--app-accent-soft) !important;
                border: 1px solid color-mix(in srgb, var(--app-primary) 30%, var(--app-border)) !important;
                color: var(--app-text) !important;
            }}

            [data-testid="stFileUploader"] [data-testid="stBaseButton-minimal"],
            [data-testid="stFileUploader"] [data-testid="stBaseButton-borderlessIcon"] {{
                position: relative !important;
                z-index: 5 !important;
                pointer-events: auto !important;
            }}

            [data-testid="stFileUploader"] *,
            [data-testid="stFileUploader"] button *,
            [data-testid="stFileUploader"] small,
            [data-testid="stFileUploader"] span {{
                color: var(--app-text) !important;
                opacity: 1 !important;
            }}

            [data-testid="stFileUploader"] svg {{
                color: var(--app-text) !important;
                stroke: currentColor !important;
            }}

            [data-testid="stFileUploader"] svg * {{
                color: var(--app-text) !important;
                stroke: currentColor !important;
            }}

            [data-testid="stFileUploader"] small {{
                color: var(--app-muted) !important;
            }}

            [data-testid="stExpander"] {{
                border: 1px solid var(--app-border) !important;
                border-radius: 8px !important;
                background: var(--app-field) !important;
                color: var(--app-text) !important;
                overflow: hidden;
                box-shadow: none !important;
            }}

            [data-testid="stExpander"] details,
            [data-testid="stExpander"] summary {{
                background: var(--app-accent-softer) !important;
                border: 0 !important;
                color: var(--app-text) !important;
                box-shadow: none !important;
                margin: 0 !important;
            }}

            [data-testid="stExpander"] details:not([open]) summary {{
                border-bottom: 0 !important;
                border-radius: 8px !important;
            }}

            [data-testid="stExpander"] details[open] summary {{
                background: var(--app-accent-softer) !important;
                border-bottom: 1px solid var(--app-border) !important;
            }}

            [data-testid="stExpander"] summary:hover,
            [data-testid="stExpander"] summary:focus,
            [data-testid="stExpander"] summary:focus-visible {{
                background: color-mix(in srgb, var(--app-primary) 14%, var(--app-panel)) !important;
                color: var(--app-text) !important;
            }}

            [data-testid="stExpander"] summary *,
            [data-testid="stExpander"] summary svg,
            [data-testid="stExpander"] summary svg * {{
                color: var(--app-text) !important;
                fill: currentColor !important;
                stroke: currentColor !important;
            }}

            [data-testid="stTextArea"] textarea:focus,
            [data-testid="stTextInput"] input:focus,
            [data-testid="stTextInput"] div[data-baseweb="input"]:focus-within,
            [data-testid="stTextInput"] div[data-baseweb="base-input"]:focus-within,
            [data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {{
                border-color: var(--app-primary) !important;
                box-shadow: 0 0 0 1px var(--app-primary) !important;
            }}

            .stButton > button,
            .stDownloadButton > button {{
                border-radius: 8px;
                border: 1px solid color-mix(in srgb, var(--app-primary) 30%, var(--app-border));
                background: var(--app-accent-soft);
                color: var(--app-text) !important;
                font-weight: 600;
            }}

            .stButton > button *,
            .stDownloadButton > button * {{
                color: inherit !important;
            }}

            .stButton > button:hover,
            .stDownloadButton > button:hover {{
                border-color: color-mix(in srgb, var(--app-primary) 72%, var(--app-border));
                background: color-mix(in srgb, var(--app-primary) 24%, var(--app-panel));
                color: var(--app-text) !important;
            }}

            .stButton > button:disabled,
            .stButton > button[disabled],
            .stDownloadButton > button:disabled,
            .stDownloadButton > button[disabled] {{
                border: 1px solid color-mix(in srgb, var(--app-primary) 26%, var(--app-border)) !important;
                background: color-mix(in srgb, var(--app-primary) 9%, var(--app-panel)) !important;
                box-shadow: none !important;
                color: color-mix(in srgb, var(--app-text) 62%, var(--app-muted)) !important;
                cursor: not-allowed !important;
                opacity: 1 !important;
            }}

            .stButton > button:disabled *,
            .stButton > button[disabled] *,
            .stDownloadButton > button:disabled *,
            .stDownloadButton > button[disabled] * {{
                color: inherit !important;
                opacity: 1 !important;
            }}

            .stButton > button[kind="primary"],
            .stButton > button[data-testid="baseButton-primary"] {{
                border-color: var(--app-primary);
                background: var(--app-primary);
                color: var(--app-primary-text) !important;
            }}

            .stButton > button[kind="primary"]:hover,
            .stButton > button[data-testid="baseButton-primary"]:hover {{
                background: var(--app-primary-hover);
                color: var(--app-primary-text) !important;
            }}

            .stButton > button[kind="primary"]:disabled,
            .stButton > button[data-testid="baseButton-primary"]:disabled,
            .stButton > button[kind="primary"][disabled],
            .stButton > button[data-testid="baseButton-primary"][disabled] {{
                border-color: color-mix(in srgb, var(--app-primary) 42%, var(--app-border)) !important;
                background: color-mix(in srgb, var(--app-primary) 20%, var(--app-panel)) !important;
                color: color-mix(in srgb, var(--app-text) 64%, var(--app-muted)) !important;
            }}

            [data-testid="stToggle"],
            [data-testid="stCheckbox"] {{
                color: var(--app-text) !important;
            }}

            input[type="checkbox"] {{
                accent-color: var(--app-primary) !important;
            }}

            [data-testid="stToggle"] label,
            [data-testid="stCheckbox"] label {{
                color: var(--app-text) !important;
                font-weight: 650;
            }}

            [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(.mode-switch-title)
            [data-testid="stVerticalBlockBorderWrapper"]:has(.mode-switch-title) {{
                width: min(14rem, 100%);
                margin: 0 0 0 auto;
                padding: 0.75rem 0.95rem !important;
                border-color: color-mix(in srgb, var(--app-primary) 22%, var(--app-border));
                background: color-mix(in srgb, var(--app-panel-soft) 76%, transparent);
                box-shadow: none;
            }}

            [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(.mode-switch-title)
            [data-testid="stVerticalBlockBorderWrapper"]:has(.mode-switch-title)
            [data-testid="stVerticalBlock"] {{
                gap: 0.25rem;
            }}

            [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(.mode-switch-title)
            [data-testid="stVerticalBlockBorderWrapper"]:has(.mode-switch-title)
            [data-testid="stToggle"] {{
                display: flex;
                justify-content: center;
            }}

            [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(.mode-switch-title)
            [data-testid="stVerticalBlockBorderWrapper"]:has(.mode-switch-title)
            [data-testid="stToggle"] label {{
                justify-content: center;
                width: auto;
                margin: 0 auto;
                gap: 0.55rem;
            }}

            [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(.mode-switch-title)
            [data-testid="stVerticalBlockBorderWrapper"]:has(.mode-switch-title)
            [data-testid="stToggle"] p {{
                font-size: 0.88rem;
                line-height: 1.2;
            }}

            [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(.mode-switch-title)
            [data-testid="stVerticalBlockBorderWrapper"]:has(.mode-switch-title)
            input[type="checkbox"] {{
                accent-color: var(--app-primary) !important;
            }}

            [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(.mode-switch-title)
            > [data-testid="stVerticalBlock"]:has(.mode-switch-title) {{
                width: min(14rem, 100%) !important;
                margin: 0 0 0 auto;
            }}

            [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(.mode-switch-title)
            > [data-testid="stVerticalBlock"]:has(.mode-switch-title)
            [data-testid="stVerticalBlock"]:has(.mode-switch-title) {{
                padding: 0.75rem 0.95rem !important;
                border-color: color-mix(in srgb, var(--app-primary) 22%, var(--app-border)) !important;
                background: color-mix(in srgb, var(--app-panel-soft) 76%, transparent);
                box-shadow: none;
            }}

            [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(.mode-switch-title)
            > [data-testid="stVerticalBlock"]:has(.mode-switch-title)
            .st-key-use_mock_mode {{
                width: 100%;
                display: flex;
                justify-content: center;
            }}

            [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(.mode-switch-title)
            > [data-testid="stVerticalBlock"]:has(.mode-switch-title)
            .st-key-use_mock_mode [data-testid="stCheckbox"] {{
                margin: 0 auto;
            }}

            .st-key-use_mock_mode label:has(input:checked) > div:first-child,
            .st-key-dark_mode label:has(input:checked) > div:first-child {{
                background: var(--app-primary) !important;
            }}

            .st-key-use_mock_mode label:not(:has(input:checked)) > div:first-child,
            .st-key-dark_mode label:not(:has(input:checked)) > div:first-child {{
                background: var(--app-toggle-off) !important;
                border: 1px solid color-mix(in srgb, var(--app-primary) 24%, var(--app-border)) !important;
            }}

            .st-key-use_mock_mode label > div:first-child > div,
            .st-key-dark_mode label > div:first-child > div {{
                background: var(--app-primary-text) !important;
            }}

            [data-testid="stAlert"] {{
                border-radius: 8px;
                border: 1px solid color-mix(in srgb, var(--app-primary) 28%, var(--app-border)) !important;
                background: color-mix(in srgb, var(--app-primary) 11%, var(--app-panel)) !important;
                color: var(--app-text) !important;
                overflow: hidden;
            }}

            [data-testid="stAlertContainer"] {{
                background: transparent !important;
                color: var(--app-text) !important;
            }}

            [data-testid^="stAlertContent"] {{
                background: transparent !important;
            }}

            [data-testid="stAlert"] * {{
                color: var(--app-text) !important;
            }}

            [data-testid="stMarkdownContainer"]:has(.connection-status) {{
                margin-bottom: 0.55rem;
                margin-top: -0.35rem;
            }}

            .connection-status {{
                --connection-color: #2fbf68;
                --connection-color-soft: rgba(47, 191, 104, 0.13);
                --connection-color-glow: rgba(47, 191, 104, 0.42);
                align-items: center;
                background: color-mix(in srgb, var(--connection-color) 7%, var(--app-panel));
                border: 1px solid color-mix(in srgb, var(--connection-color) 34%, var(--app-border));
                border-radius: 8px;
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1);
                box-sizing: border-box;
                display: flex;
                gap: 0.62rem;
                margin: 0.18rem 0 0;
                padding: 0.5rem 0.72rem;
            }}

            .connection-status-error {{
                --connection-color: #dc4b41;
                --connection-color-soft: rgba(220, 75, 65, 0.12);
                --connection-color-glow: rgba(220, 75, 65, 0.4);
            }}

            .connection-status-dot {{
                background: var(--connection-color);
                border-radius: 999px;
                box-shadow:
                    0 0 0 0.25rem var(--connection-color-soft),
                    0 0 1rem var(--connection-color-glow);
                flex: 0 0 auto;
                height: 0.62rem;
                width: 0.62rem;
            }}

            .connection-status-copy {{
                display: flex;
                min-width: 0;
            }}

            .connection-status-title {{
                color: var(--app-text);
                font-size: 0.9rem;
                font-weight: 720;
                line-height: 1.3;
                overflow-wrap: anywhere;
            }}

            .centered-alert {{
                align-items: center;
                background: color-mix(in srgb, var(--app-primary) 11%, var(--app-panel));
                border: 1px solid color-mix(in srgb, var(--app-primary) 28%, var(--app-border));
                border-radius: 8px;
                box-sizing: border-box;
                color: var(--app-text);
                display: flex;
                justify-content: center;
                line-height: 1.35;
                margin-bottom: 1rem;
                min-height: 3.5rem;
                padding: 0.5rem 1.25rem;
                text-align: center;
                width: 100%;
            }}

            hr {{
                border-color: var(--app-border);
            }}

            @media (max-width: 700px) {{
                [data-testid="stAppViewContainer"] > .main .block-container {{
                    padding-top: 1rem;
                }}

                .page-title {{
                    font-size: 2rem;
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def analyze_with_api(
    jd_text: str,
    uploaded_files,
    screening_config: LLMConfig,
    ocr_config: LLMConfig,
    progress_callback=None,
    max_workers: int = DEFAULT_SCREENING_WORKERS,
) -> tuple[list[dict], list[str]]:
    """真实 API 模式：解析 JD，并发解析简历并生成匹配结果。"""
    errors = []
    results = []

    jd_prompt = build_prompt("jd_extract_prompt.txt", "岗位JD原文", jd_text)
    jd_raw, jd_call_error = call_llm(jd_prompt, screening_config)
    if jd_call_error:
        errors.append(f"JD解析调用失败：{jd_call_error}")

    jd_data, jd_error = safe_parse_json(jd_raw)

    if jd_error:
        errors.append(f"JD解析JSON失败：{jd_error}")
        jd_data = {"raw_jd": jd_text, "ai_output": jd_raw}

    total_files = len(uploaded_files)
    worker_count = max(1, min(max_workers, total_files))
    completed_files = 0

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_file_name = {
            executor.submit(
                analyze_resume_file_with_api,
                uploaded_file,
                jd_data,
                screening_config,
                ocr_config,
            ): uploaded_file.name
            for uploaded_file in uploaded_files
        }

        for future in as_completed(future_to_file_name):
            file_name = future_to_file_name[future]
            try:
                result, error = future.result()
            except Exception as exc:
                error = f"{file_name}：处理失败：{exc}"
                result = make_error_result(file_name, error)

            if error:
                errors.append(error)
            results.append(result)
            completed_files += 1

            if progress_callback:
                progress_callback(completed_files, total_files, file_name)

    results.sort(key=lambda item: item.get("匹配度", 0), reverse=True)
    return results, errors


def analyze_resume_file_with_api(
    uploaded_file,
    jd_data: dict,
    screening_config: LLMConfig,
    ocr_config: LLMConfig,
) -> tuple[dict, str | None]:
    file_name = uploaded_file.name
    resume_text, parse_error = parse_resume_file(uploaded_file, ocr_config)

    if parse_error:
        return make_error_result(file_name, parse_error), f"{file_name}：{parse_error}"

    analysis_input = {
        "job": jd_data,
        "resume_text": resume_text,
        "note": "请只基于岗位相关因素进行辅助匹配评分。",
    }
    analysis_prompt = build_prompt(
        "resume_screening_prompt.txt",
        "JD结构化信息与简历原文",
        json.dumps(analysis_input, ensure_ascii=False, indent=2),
    )
    analysis_raw, analysis_call_error = call_llm(analysis_prompt, screening_config)
    if analysis_call_error:
        return (
            make_error_result(file_name, analysis_call_error),
            f"{file_name}：综合分析调用失败：{analysis_call_error}",
        )

    analysis_data, analysis_error = safe_parse_json(analysis_raw)

    if analysis_error:
        resume_data = {"raw_resume": resume_text[:4000], "ai_output": analysis_raw}
        return (
            make_error_result(file_name, analysis_error, resume_data, analysis_raw),
            f"{file_name}：综合分析JSON失败：{analysis_error}",
        )

    return normalize_combined_analysis_result(file_name, analysis_data, analysis_raw), None


def normalize_combined_analysis_result(file_name: str, analysis_data: dict, raw_ai_output: str) -> dict:
    resume_data = analysis_data.get("resume", {})
    match_data = analysis_data.get("match", {})

    if not isinstance(resume_data, dict):
        resume_data = {}
    if not isinstance(match_data, dict):
        match_data = {}

    return normalize_result(file_name, resume_data, match_data, raw_ai_output)


def build_uploaded_file_fingerprint(uploaded_file) -> str:
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()


def build_uploaded_file_key(uploaded_file, fingerprint: str, occurrence: int) -> str:
    return f"{fingerprint}:{uploaded_file.name}:{occurrence}"


def build_uploaded_file_records(uploaded_files, removed_file_keys=None) -> list[dict]:
    removed_file_keys = set(removed_file_keys or [])
    records = []
    seen_fingerprints = set()
    occurrence_counts = {}

    for uploaded_file in uploaded_files or []:
        fingerprint = build_uploaded_file_fingerprint(uploaded_file)
        occurrence_key = (uploaded_file.name, fingerprint)
        occurrence_counts[occurrence_key] = occurrence_counts.get(occurrence_key, 0) + 1
        file_key = build_uploaded_file_key(uploaded_file, fingerprint, occurrence_counts[occurrence_key])

        is_supported = is_supported_resume_file(uploaded_file)
        is_removed = file_key in removed_file_keys
        is_duplicate = False
        if is_supported and not is_removed:
            is_duplicate = fingerprint in seen_fingerprints
            if not is_duplicate:
                seen_fingerprints.add(fingerprint)

        records.append(
            {
                "file": uploaded_file,
                "key": file_key,
                "name": uploaded_file.name,
                "extension": get_resume_file_extension(uploaded_file.name),
                "is_supported": is_supported,
                "is_duplicate": is_duplicate,
                "is_removed": is_removed,
            }
        )

    return records


def split_uploaded_file_records(file_records: list[dict]) -> tuple[list, list[str], list[str]]:
    accepted_files = []
    rejected_names = []
    duplicate_names = []

    for file_record in file_records:
        if file_record["is_removed"]:
            continue
        if not file_record["is_supported"]:
            rejected_names.append(file_record["name"])
            continue
        if file_record["is_duplicate"]:
            duplicate_names.append(file_record["name"])
            continue
        accepted_files.append(file_record["file"])

    return accepted_files, rejected_names, duplicate_names


def filter_supported_resume_files(uploaded_files, removed_file_keys=None) -> tuple[list, list[str], list[str]]:
    return split_uploaded_file_records(build_uploaded_file_records(uploaded_files, removed_file_keys))


def should_reset_uploader_after_removing(file_records: list[dict], file_keys_to_remove: list[str]) -> bool:
    visible_file_keys = {file_record["key"] for file_record in file_records if not file_record["is_removed"]}
    return bool(visible_file_keys) and visible_file_keys.issubset(set(file_keys_to_remove))


def build_uploaded_file_summary(
    total_count: int,
    accepted_count: int,
    rejected_count: int,
    duplicate_count: int = 0,
) -> tuple[str, str]:
    main_text = f"已上传 {total_count} 个文件"
    if total_count == 0:
        return main_text, "等待上传文件"
    if rejected_count or duplicate_count:
        detail_parts = [f"可处理 {accepted_count} 个"]
        if rejected_count:
            detail_parts.append(f"暂不处理 {rejected_count} 个")
        if duplicate_count:
            detail_parts.append(f"已跳过重复 {duplicate_count} 个")
        return main_text, "，".join(detail_parts)
    return main_text, "全部文件均可处理"


def show_uploaded_file_summary(
    total_count: int,
    accepted_count: int,
    rejected_count: int,
    duplicate_count: int = 0,
) -> None:
    main_text, detail_text = build_uploaded_file_summary(total_count, accepted_count, rejected_count, duplicate_count)
    summary_class = "upload-file-summary upload-file-summary-warning" if rejected_count or duplicate_count else "upload-file-summary"
    escaped_main_text = html.escape(main_text).replace(str(total_count), f"<strong>{total_count}</strong>", 1)
    st.markdown(
        f"""
            <div class="{summary_class}">
                <span>{escaped_main_text}</span>
                <span class="upload-file-summary-detail">{html.escape(detail_text)}</span>
            </div>
        """,
        unsafe_allow_html=True,
    )


def show_centered_alert(message: str) -> None:
    st.markdown(f'<div class="centered-alert">{html.escape(message)}</div>', unsafe_allow_html=True)


def show_upload_cleanup_action_spacer() -> None:
    st.markdown('<div class="upload-cleanup-action-spacer"></div>', unsafe_allow_html=True)


def show_results_download_action_spacer() -> None:
    st.markdown('<div class="results-download-action-spacer"></div>', unsafe_allow_html=True)


def get_uploaded_file_record_marker(file_record: dict) -> str:
    if file_record["is_duplicate"]:
        return "重"
    if not file_record["is_supported"]:
        return "!"
    marker_by_extension = {
        "pdf": "PDF",
        "docx": "DOC",
        "txt": "TXT",
        "png": "图",
        "jpg": "图",
        "jpeg": "图",
    }
    return marker_by_extension.get(file_record["extension"], "文")


def escape_css_content(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def build_uploaded_file_chip_marker_css(file_records: list[dict]) -> str:
    if not file_records:
        return ""

    rules = []
    for index, file_record in enumerate(file_records, start=1):
        chip_selector = (
            '[data-testid="stFileUploader"] [data-testid="stFileChips"] '
            f'> div:first-child > div:nth-child({index}) [data-testid="stFileChip"]'
        )
        icon_selector = f"{chip_selector} > div:first-child"

        if file_record["is_removed"]:
            rules.append(f"{chip_selector} {{ display: none !important; }}")
            continue

        marker = escape_css_content(get_uploaded_file_record_marker(file_record))
        marker_color = "#d94a3a" if file_record["is_duplicate"] or not file_record["is_supported"] else "var(--app-primary)"
        marker_background = (
            "color-mix(in srgb, #d94a3a 16%, var(--app-panel))"
            if file_record["is_duplicate"] or not file_record["is_supported"]
            else "color-mix(in srgb, var(--app-primary) 14%, var(--app-panel))"
        )
        marker_border = (
            "color-mix(in srgb, #d94a3a 58%, var(--app-border))"
            if file_record["is_duplicate"] or not file_record["is_supported"]
            else "color-mix(in srgb, var(--app-primary) 50%, var(--app-border))"
        )
        chip_border = (
            "color-mix(in srgb, #d94a3a 42%, var(--app-border))"
            if file_record["is_duplicate"]
            else "color-mix(in srgb, var(--app-primary) 20%, var(--app-border))"
        )
        chip_background = (
            "color-mix(in srgb, #d94a3a 6%, var(--app-panel))"
            if file_record["is_duplicate"]
            else "var(--app-field)"
        )

        rules.append(
            f"""
            {chip_selector} {{
                border: 1px solid {chip_border} !important;
                background: {chip_background} !important;
            }}
            {icon_selector} {{
                background: {marker_background} !important;
                border: 1px solid {marker_border} !important;
                color: transparent !important;
                position: relative !important;
            }}
            {icon_selector} svg,
            {icon_selector} img {{
                opacity: 0 !important;
            }}
            {icon_selector}::after {{
                align-items: center;
                color: {marker_color};
                content: "{marker}";
                display: flex;
                font-size: 0.72rem;
                font-weight: 850;
                inset: 0;
                justify-content: center;
                line-height: 1;
                position: absolute;
            }}
            """
        )

    return "\n".join(rules)


def show_uploaded_file_chip_markers(file_records: list[dict]) -> None:
    marker_css = build_uploaded_file_chip_marker_css(file_records)
    if marker_css:
        st.markdown(f"<style>{marker_css}</style>", unsafe_allow_html=True)


def show_unsupported_file_warning(rejected_file_names: list[str]) -> None:
    escaped_names = "<br>".join(f"- {html.escape(file_name)}" for file_name in rejected_file_names)
    st.markdown(
        f"""
            <div class="unsupported-file-warning">
                <div class="unsupported-file-warning-title">当前版本暂不处理：</div>
                <div class="unsupported-file-warning-files">{escaped_names}</div>
                <div>请上传 PDF、DOCX、TXT、PNG、JPG 或 JPEG 文件。</div>
            </div>
        """,
        unsafe_allow_html=True,
    )


def show_duplicate_file_warning(duplicate_file_names: list[str]) -> None:
    escaped_names = "<br>".join(f"- {html.escape(file_name)}" for file_name in duplicate_file_names)
    st.markdown(
        f"""
            <div class="unsupported-file-warning">
                <div class="unsupported-file-warning-title">已跳过重复文件：</div>
                <div class="unsupported-file-warning-files">{escaped_names}</div>
                <div>这些文件内容与本次上传中的其他文件一致，不会重复筛选。</div>
            </div>
        """,
        unsafe_allow_html=True,
    )


def show_results(results: list[dict], errors: list[str]) -> None:
    """展示排序表、候选人详情和CSV下载按钮。"""
    if errors:
        st.warning("部分内容处理失败，已保留可展示结果。")
        for error in errors:
            st.write(f"- {error}")

    if not results:
        show_centered_alert("还没有可展示的候选人结果。")
        return

    with st.container(border=True):
        st.subheader("候选人排序表")
        table_columns = [
            "文件名",
            "姓名",
            "学校",
            "专业",
            "学历",
            "匹配度",
            "推荐等级",
            "命中硬性条件",
            "缺失硬性条件",
            "匹配理由",
            "风险点",
        ]
        table_data = pd.DataFrame(results)
        for column in table_columns:
            if column not in table_data.columns:
                table_data[column] = ""
        preview_html = table_data[table_columns].to_html(index=False, classes="results-table")
        st.markdown(f'<div class="results-table-wrap">{preview_html}</div>', unsafe_allow_html=True)

        csv_bytes = results_to_csv(results)
        show_results_download_action_spacer()
        st.download_button(
            "下载CSV结果",
            data=csv_bytes,
            file_name="resume_screening_results.csv",
            mime="text/csv",
        )

    with st.container(border=True):
        st.subheader("候选人详情")
        options = [f"{item.get('姓名', '未知姓名')} - {item.get('文件名', '')}" for item in results]
        selected = st.selectbox("选择候选人", options)
        selected_index = options.index(selected)
        candidate = results[selected_index]

        st.write(f"**姓名：** {candidate.get('姓名', '')}")
        st.write(f"**学校：** {candidate.get('学校', '')}")
        st.write(f"**专业：** {candidate.get('专业', '')}")
        st.write(f"**学历：** {candidate.get('学历', '')}")
        st.write(f"**匹配度：** {candidate.get('匹配度', 0)}")
        st.write(f"**推荐等级：** {candidate.get('推荐等级', '')}")
        st.write(f"**命中硬性条件：** {candidate.get('命中硬性条件', '')}")
        st.write(f"**缺失硬性条件：** {candidate.get('缺失硬性条件', '')}")
        st.write(f"**证据摘要：** {candidate.get('证据摘要', '')}")
        st.write(f"**匹配理由：** {candidate.get('匹配理由', '')}")
        st.write(f"**风险点：** {candidate.get('风险点', '')}")
        st.write(f"**面试追问：** {candidate.get('面试追问', '')}")

        if candidate.get("原始AI输出"):
            with st.expander("查看原始AI输出"):
                st.code(candidate["原始AI输出"])


def main() -> None:
    st.set_page_config(page_title="AI辅助简历筛选", layout="wide")

    screening_requested = bool(st.session_state.get(SCREENING_REQUESTED_KEY, False))
    screening_in_progress = bool(st.session_state.get(SCREENING_IN_PROGRESS_KEY, False))

    with st.sidebar:
        with st.container(border=True):
            st.markdown('<div class="side-control-label">主题</div>', unsafe_allow_html=True)
            dark_mode = st.toggle("深色模式", value=False, key="dark_mode", disabled=screening_in_progress)
        screening_config, ocr_config = render_model_api_settings(disabled=screening_in_progress)

    apply_page_style(dark_mode)

    st.markdown('<h1 class="page-title">AI辅助简历筛选</h1>', unsafe_allow_html=True)
    show_centered_alert(AI_DISCLAIMER)

    if "jd_text" not in st.session_state:
        st.session_state["jd_text"] = ""

    with st.container(border=True):
        control_col, _, mock_col = st.columns([1, 1, 1])
        with control_col:
            if st.button("填入示例JD", disabled=screening_in_progress):
                st.session_state["jd_text"] = load_sample_jd()
        with mock_col:
            with st.container(border=True):
                st.markdown('<div class="mode-switch-title">运行模式</div>', unsafe_allow_html=True)
                use_mock = st.toggle("Mock模式", value=True, key="use_mock_mode", disabled=screening_in_progress)

        jd_text = st.text_area(
            "粘贴岗位JD",
            height=220,
            placeholder="请在这里粘贴岗位职责、任职要求、加分项等内容。",
            key="jd_text",
            disabled=screening_in_progress,
        )

        if RESUME_UPLOADER_VERSION_KEY not in st.session_state:
            st.session_state[RESUME_UPLOADER_VERSION_KEY] = 0
        if REMOVED_UPLOAD_FILE_KEYS_KEY not in st.session_state:
            st.session_state[REMOVED_UPLOAD_FILE_KEYS_KEY] = []

        uploader_key = f"resume_file_uploader_{st.session_state[RESUME_UPLOADER_VERSION_KEY]}"
        uploaded_files = st.file_uploader(
            "上传简历文件（支持拖拽上传，PDF、DOCX、TXT、PNG、JPG/JPEG，可多选）",
            accept_multiple_files=True,
            key=uploader_key,
            disabled=screening_in_progress,
        )
        removed_file_keys = set(st.session_state.get(REMOVED_UPLOAD_FILE_KEYS_KEY, []))
        if not uploaded_files and removed_file_keys:
            st.session_state[REMOVED_UPLOAD_FILE_KEYS_KEY] = []
            removed_file_keys = set()

        uploaded_file_records = build_uploaded_file_records(uploaded_files, removed_file_keys)
        valid_uploaded_files, rejected_file_names, duplicate_file_names = split_uploaded_file_records(uploaded_file_records)
        duplicate_file_keys = [file_record["key"] for file_record in uploaded_file_records if file_record["is_duplicate"]]
        rejected_file_keys = [
            file_record["key"]
            for file_record in uploaded_file_records
            if not file_record["is_removed"] and not file_record["is_supported"]
        ]
        visible_file_records = [file_record for file_record in uploaded_file_records if not file_record["is_removed"]]

        show_uploaded_file_chip_markers(uploaded_file_records)

        if duplicate_file_names:
            show_duplicate_file_warning(duplicate_file_names)
            show_upload_cleanup_action_spacer()
            if st.button("一键去除重复项", use_container_width=True, disabled=screening_in_progress):
                if should_reset_uploader_after_removing(uploaded_file_records, duplicate_file_keys):
                    st.session_state[REMOVED_UPLOAD_FILE_KEYS_KEY] = []
                    st.session_state[RESUME_UPLOADER_VERSION_KEY] += 1
                else:
                    st.session_state[REMOVED_UPLOAD_FILE_KEYS_KEY] = sorted(removed_file_keys.union(duplicate_file_keys))
                st.rerun()

        if rejected_file_names:
            show_unsupported_file_warning(rejected_file_names)
            show_upload_cleanup_action_spacer()
            if st.button("一键去除非法项目", use_container_width=True, disabled=screening_in_progress):
                if should_reset_uploader_after_removing(uploaded_file_records, rejected_file_keys):
                    st.session_state[REMOVED_UPLOAD_FILE_KEYS_KEY] = []
                    st.session_state[RESUME_UPLOADER_VERSION_KEY] += 1
                else:
                    st.session_state[REMOVED_UPLOAD_FILE_KEYS_KEY] = sorted(removed_file_keys.union(rejected_file_keys))
                st.rerun()

        if visible_file_records:
            if st.button("清除文件", use_container_width=True, disabled=screening_in_progress):
                st.session_state[REMOVED_UPLOAD_FILE_KEYS_KEY] = []
                st.session_state[RESUME_UPLOADER_VERSION_KEY] += 1
                st.rerun()
        st.markdown(
            '<div class="api-mode-caption">真实 API 模式会把 JD、简历文本和图片OCR内容发送到你配置的模型服务商；请勿上传未获授权的真实敏感简历。</div>',
            unsafe_allow_html=True,
        )

    start_screening = st.button("开始筛选", type="primary", use_container_width=True, disabled=screening_in_progress)
    if start_screening:
        st.session_state[SCREENING_REQUESTED_KEY] = True
        screening_requested = True

    if screening_requested:
        if use_mock:
            st.session_state["results"] = get_mock_results()
            st.session_state["errors"] = []
            clear_screening_run_flags()
        else:
            if not jd_text.strip():
                clear_screening_run_flags()
                st.error("请先粘贴岗位JD，或勾选Mock演示模式。")
                return
            if not uploaded_files:
                clear_screening_run_flags()
                st.error("请上传至少一份简历，或勾选Mock演示模式。")
                return
            if not valid_uploaded_files:
                clear_screening_run_flags()
                st.error("请上传至少一份 PDF、DOCX、TXT、PNG、JPG 或 JPEG 简历，或勾选Mock演示模式。")
                return
            if not screening_in_progress:
                st.session_state[SCREENING_IN_PROGRESS_KEY] = True
                st.rerun()

            progress_bar = st.progress(0)
            progress_text = st.empty()

            def update_progress(current: int, total: int, file_name: str) -> None:
                progress_bar.progress(current / total, text=f"已完成：{file_name}（{current}/{total}）")
                progress_text.caption(f"最近完成：{file_name}")

            try:
                with st.spinner("正在解析简历并调用大模型，请稍候..."):
                    results, errors = analyze_with_api(
                        jd_text,
                        valid_uploaded_files,
                        screening_config,
                        ocr_config,
                        update_progress,
                    )
                    st.session_state["results"] = results
                    st.session_state["errors"] = errors
            finally:
                clear_screening_run_flags()
            progress_bar.progress(1.0, text="已完成")
            progress_text.empty()
            st.rerun()

    show_results(st.session_state.get("results", []), st.session_state.get("errors", []))

    st.caption("本地PoC Demo：不保存上传文件，不对接企业系统。")


if __name__ == "__main__":
    main()
