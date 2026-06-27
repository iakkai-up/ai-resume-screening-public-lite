import base64
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from services.exporter import results_to_csv
from services.file_parser import UNSUPPORTED_RESUME_FORMAT_MESSAGE, is_supported_resume_file, parse_resume_file
from services.llm_client import (
    LLMConfig,
    TEST_IMAGE_BASE64,
    call_llm,
    call_vision_ocr,
    get_ocr_config,
    test_llm_connection,
    test_vision_connection,
)
from services.scorer import normalize_result, safe_parse_json
from app import (
    SCREENING_IN_PROGRESS_KEY,
    analyze_with_api,
    build_uploaded_file_chip_marker_css,
    build_model_options,
    build_uploaded_file_records,
    build_uploaded_file_summary,
    filter_supported_resume_files,
    get_uploaded_file_record_marker,
    normalize_model_options,
    resolve_ocr_config,
    should_reset_uploader_after_removing,
    split_uploaded_file_records,
)


class UploadedFileStub:
    def __init__(self, name: str, content: bytes) -> None:
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


class UnreadableUploadedFileStub:
    def __init__(self, name: str) -> None:
        self.name = name

    def getvalue(self) -> bytes:
        raise AssertionError("Unsupported files should be rejected before reading content")


class CoreBehaviorTest(unittest.TestCase):
    def test_safe_parse_json_accepts_markdown_block(self) -> None:
        data, error = safe_parse_json('```json\n{"score": 88}\n```')

        self.assertIsNone(error)
        self.assertEqual(data, {"score": 88})

    def test_normalize_result_includes_review_evidence_fields(self) -> None:
        result = normalize_result(
            "resume.txt",
            {"name": "张明"},
            {
                "score": 91,
                "matched_must_have": ["Python"],
                "missing_must_have": [],
                "evidence": ["项目经历包含数据分析 Demo"],
            },
        )

        self.assertEqual(result["命中硬性条件"], "Python")
        self.assertEqual(result["证据摘要"], "项目经历包含数据分析 Demo")

    def test_parse_txt_supports_gbk(self) -> None:
        uploaded_file = UploadedFileStub("resume.txt", "姓名：张明".encode("gbk"))

        text, error = parse_resume_file(uploaded_file)

        self.assertIsNone(error)
        self.assertIn("张明", text)

    def test_supported_resume_file_uses_extension_whitelist(self) -> None:
        self.assertTrue(is_supported_resume_file(UploadedFileStub("resume.PDF", b"")))
        self.assertTrue(is_supported_resume_file(UploadedFileStub("resume.docx", b"")))
        self.assertTrue(is_supported_resume_file(UploadedFileStub("resume.txt", b"")))
        self.assertTrue(is_supported_resume_file(UploadedFileStub("resume.png", b"")))
        self.assertTrue(is_supported_resume_file(UploadedFileStub("resume.JPG", b"")))
        self.assertTrue(is_supported_resume_file(UploadedFileStub("resume.jpeg", b"")))

        for file_name in ("archive.zip", "table.xlsx", "script.exe", "resume"):
            with self.subTest(file_name=file_name):
                self.assertFalse(is_supported_resume_file(UploadedFileStub(file_name, b"")))

    def test_uploaded_file_summary_reports_total_and_rejected_counts(self) -> None:
        self.assertEqual(build_uploaded_file_summary(0, 0, 0), ("已上传 0 个文件", "等待上传文件"))
        self.assertEqual(build_uploaded_file_summary(3, 3, 0), ("已上传 3 个文件", "全部文件均可处理"))
        self.assertEqual(
            build_uploaded_file_summary(4, 3, 1),
            ("已上传 4 个文件", "可处理 3 个，暂不处理 1 个"),
        )
        self.assertEqual(
            build_uploaded_file_summary(4, 2, 1, 1),
            ("已上传 4 个文件", "可处理 2 个，暂不处理 1 个，已跳过重复 1 个"),
        )

    def test_filter_supported_resume_files_skips_duplicate_content(self) -> None:
        uploaded_files = [
            UploadedFileStub("resume-a.txt", b"same resume"),
            UploadedFileStub("resume-b.txt", b"same resume"),
            UploadedFileStub("resume-c.txt", b"different resume"),
            UploadedFileStub("archive.zip", b"same resume"),
        ]

        accepted_files, rejected_names, duplicate_names = filter_supported_resume_files(uploaded_files)

        self.assertEqual([uploaded_file.name for uploaded_file in accepted_files], ["resume-a.txt", "resume-c.txt"])
        self.assertEqual(rejected_names, ["archive.zip"])
        self.assertEqual(duplicate_names, ["resume-b.txt"])

    def test_uploaded_file_records_mark_supported_file_types_and_duplicates(self) -> None:
        uploaded_files = [
            UploadedFileStub("resume.pdf", b"pdf"),
            UploadedFileStub("resume.docx", b"docx"),
            UploadedFileStub("resume.txt", b"txt"),
            UploadedFileStub("resume.png", b"image"),
            UploadedFileStub("resume-copy.jpg", b"image"),
        ]

        file_records = build_uploaded_file_records(uploaded_files)

        self.assertEqual([get_uploaded_file_record_marker(file_record) for file_record in file_records], ["PDF", "DOC", "TXT", "图", "重"])

    def test_uploaded_file_chip_marker_css_targets_native_file_cards(self) -> None:
        file_records = build_uploaded_file_records(
            [
                UploadedFileStub("resume.pdf", b"pdf"),
                UploadedFileStub("resume-copy.pdf", b"pdf"),
            ]
        )

        marker_css = build_uploaded_file_chip_marker_css(file_records)

        self.assertIn("nth-child(1)", marker_css)
        self.assertIn('content: "PDF"', marker_css)
        self.assertIn("nth-child(2)", marker_css)
        self.assertIn('content: "重"', marker_css)

    def test_removed_duplicate_file_keys_hide_duplicates_from_queue(self) -> None:
        uploaded_files = [
            UploadedFileStub("resume-a.txt", b"same resume"),
            UploadedFileStub("resume-b.txt", b"same resume"),
            UploadedFileStub("resume-c.txt", b"different resume"),
        ]
        file_records = build_uploaded_file_records(uploaded_files)
        duplicate_file_keys = [file_record["key"] for file_record in file_records if file_record["is_duplicate"]]

        filtered_records = build_uploaded_file_records(uploaded_files, duplicate_file_keys)
        accepted_files, rejected_names, duplicate_names = split_uploaded_file_records(filtered_records)

        self.assertEqual([uploaded_file.name for uploaded_file in accepted_files], ["resume-a.txt", "resume-c.txt"])
        self.assertEqual(rejected_names, [])
        self.assertEqual(duplicate_names, [])

    def test_removed_rejected_file_keys_hide_illegal_items_from_queue(self) -> None:
        uploaded_files = [UploadedFileStub("archive.zip", b"zip")]
        file_records = build_uploaded_file_records(uploaded_files)
        rejected_file_keys = [file_record["key"] for file_record in file_records if not file_record["is_supported"]]

        filtered_records = build_uploaded_file_records(uploaded_files, rejected_file_keys)
        accepted_files, rejected_names, duplicate_names = split_uploaded_file_records(filtered_records)

        self.assertTrue(should_reset_uploader_after_removing(file_records, rejected_file_keys))
        self.assertEqual(accepted_files, [])
        self.assertEqual(rejected_names, [])
        self.assertEqual(duplicate_names, [])

    def test_screening_controls_disable_while_screening_in_progress(self) -> None:
        app = AppTest.from_file(str(ROOT_DIR / "app.py"))
        app.session_state[SCREENING_IN_PROGRESS_KEY] = True

        app.run(timeout=10)

        self.assertEqual(len(app.exception), 0)
        toggles = {toggle.label: toggle for toggle in app.toggle}
        buttons = {button.label: button for button in app.button}
        self.assertTrue(toggles["深色模式"].disabled)
        self.assertTrue(toggles["Mock模式"].disabled)
        self.assertTrue(buttons["开始筛选"].disabled)
        self.assertTrue(buttons["测试筛选 API"].disabled)
        self.assertTrue(buttons["测试 OCR API"].disabled)

    def test_upload_cleanup_buttons_disable_while_screening_in_progress(self) -> None:
        app = AppTest.from_file(str(ROOT_DIR / "app.py"))
        app.run(timeout=10)
        app.file_uploader[0].set_value(
            [
                ("resume-a.txt", b"same resume", "text/plain"),
                ("resume-b.txt", b"same resume", "text/plain"),
                ("archive.zip", b"zip", "application/zip"),
            ]
        )
        app.session_state[SCREENING_IN_PROGRESS_KEY] = True

        app.run(timeout=10)

        self.assertEqual(len(app.exception), 0)
        buttons = {button.label: button for button in app.button}
        self.assertTrue(buttons["一键去除重复项"].disabled)
        self.assertTrue(buttons["一键去除非法项目"].disabled)
        self.assertTrue(buttons["清除文件"].disabled)

    def test_model_options_accept_toml_array_and_csv_string(self) -> None:
        self.assertEqual(normalize_model_options(["a", "b", "a", ""]), ["a", "b"])
        self.assertEqual(normalize_model_options("a, b\nc"), ["a", "b", "c"])

    def test_model_options_keep_default_first(self) -> None:
        options = build_model_options("default-model", ["model-a", "default-model"])

        self.assertEqual(options, ["default-model", "model-a"])

    def test_model_options_do_not_add_unconfigured_fallbacks(self) -> None:
        self.assertEqual(build_model_options("default-model", []), ["default-model"])

    def test_analyze_with_api_uses_one_screening_call_per_resume(self) -> None:
        uploaded_file = UploadedFileStub("resume.txt", "姓名：张明\nPython".encode("utf-8"))
        calls = []

        def fake_call_llm(prompt: str, config: LLMConfig) -> tuple[str, str | None]:
            calls.append(prompt)
            if "岗位JD原文" in prompt:
                return json.dumps({"job_title": "AI产品实习生", "must_have": ["Python"]}, ensure_ascii=False), None
            return (
                json.dumps(
                    {
                        "resume": {"name": "张明", "school": "上海交通大学", "major": "计算机", "degree": "本科"},
                        "match": {
                            "score": 88,
                            "recommendation": "强烈建议面试",
                            "matched_must_have": ["Python"],
                            "missing_must_have": [],
                            "evidence": ["简历写明 Python 项目"],
                        },
                    },
                    ensure_ascii=False,
                ),
                None,
            )

        with patch("app.call_llm", side_effect=fake_call_llm):
            results, errors = analyze_with_api(
                "需要 Python",
                [uploaded_file],
                LLMConfig("key", "https://screening.example.com", "model"),
                LLMConfig("", "", ""),
                max_workers=1,
            )

        self.assertEqual(errors, [])
        self.assertEqual(len(calls), 2)
        self.assertEqual(results[0]["姓名"], "张明")
        self.assertEqual(results[0]["匹配度"], 88)
        self.assertEqual(results[0]["命中硬性条件"], "Python")

    def test_analyze_with_api_keeps_other_results_when_one_file_fails(self) -> None:
        uploaded_files = [
            UploadedFileStub("good.txt", "姓名：李雷\nPython".encode("utf-8")),
            UploadedFileStub("bad.txt", b"bad"),
        ]

        def fake_parse_resume_file(uploaded_file, ocr_config: LLMConfig) -> tuple[str, str | None]:
            if uploaded_file.name == "bad.txt":
                return "", "文件损坏"
            return "姓名：李雷\nPython", None

        def fake_call_llm(prompt: str, config: LLMConfig) -> tuple[str, str | None]:
            if "岗位JD原文" in prompt:
                return json.dumps({"job_title": "AI产品实习生", "must_have": ["Python"]}, ensure_ascii=False), None
            return (
                json.dumps(
                    {
                        "resume": {"name": "李雷"},
                        "match": {"score": 80, "recommendation": "建议面试", "matched_must_have": ["Python"]},
                    },
                    ensure_ascii=False,
                ),
                None,
            )

        with patch("app.parse_resume_file", side_effect=fake_parse_resume_file), patch(
            "app.call_llm",
            side_effect=fake_call_llm,
        ):
            results, errors = analyze_with_api(
                "需要 Python",
                uploaded_files,
                LLMConfig("key", "https://screening.example.com", "model"),
                LLMConfig("", "", ""),
                max_workers=1,
            )

        self.assertEqual(len(results), 2)
        self.assertTrue(any("bad.txt：文件损坏" == error for error in errors))
        self.assertEqual(results[0]["文件名"], "good.txt")
        self.assertEqual(results[0]["匹配度"], 80)
        self.assertEqual(results[1]["文件名"], "bad.txt")
        self.assertEqual(results[1]["匹配度"], 0)

    def test_unsupported_resume_format_is_rejected_before_reading_content(self) -> None:
        for file_name in ("archive.zip", "table.xlsx", "script.exe", "resume"):
            with self.subTest(file_name=file_name):
                text, error = parse_resume_file(UnreadableUploadedFileStub(file_name))

                self.assertEqual(text, "")
                self.assertEqual(error, UNSUPPORTED_RESUME_FORMAT_MESSAGE)

    def test_image_resume_uses_vision_ocr(self) -> None:
        uploaded_file = UploadedFileStub("resume.png", b"fake image bytes")
        ocr_config = LLMConfig("ocr-key", "https://ocr.example.com", "ocr-model")

        with patch("services.file_parser.call_vision_ocr", return_value=("姓名：张明", None)) as mock_ocr:
            text, error = parse_resume_file(uploaded_file, ocr_config)

        self.assertIsNone(error)
        self.assertEqual(text, "姓名：张明")
        mock_ocr.assert_called_once()
        self.assertEqual(mock_ocr.call_args.args[0][0]["mime_type"], "image/png")
        self.assertIs(mock_ocr.call_args.args[1], ocr_config)

    def test_scanned_pdf_falls_back_to_vision_ocr(self) -> None:
        import fitz

        document = fitz.open()
        document.new_page()
        pdf_bytes = document.write()
        document.close()
        uploaded_file = UploadedFileStub("scan.pdf", pdf_bytes)
        ocr_config = LLMConfig("ocr-key", "https://ocr.example.com", "ocr-model")

        with patch("services.file_parser.call_vision_ocr", return_value=("姓名：张明", None)) as mock_ocr:
            text, error = parse_resume_file(uploaded_file, ocr_config)

        self.assertIsNone(error)
        self.assertEqual(text, "姓名：张明")
        mock_ocr.assert_called_once()
        self.assertEqual(mock_ocr.call_args.args[0][0]["mime_type"], "image/png")
        self.assertIs(mock_ocr.call_args.args[1], ocr_config)

    def test_missing_ocr_config_blocks_image_only(self) -> None:
        image_text, image_error = parse_resume_file(UploadedFileStub("resume.png", b"fake image bytes"), LLMConfig("", "", ""))
        txt_text, txt_error = parse_resume_file(UploadedFileStub("resume.txt", "姓名：张明".encode("utf-8")), LLMConfig("", "", ""))

        self.assertEqual(image_text, "")
        self.assertIn("OCR配置缺失", image_error)
        self.assertIsNone(txt_error)
        self.assertIn("张明", txt_text)

    def test_screening_call_uses_screening_config_model(self) -> None:
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))])
        create = Mock(return_value=response)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        config = LLMConfig("screening-key", "https://screening.example.com", "screening-model")

        with patch("services.llm_client.OpenAI", return_value=client) as mock_openai:
            text, error = call_llm("prompt", config)

        self.assertIsNone(error)
        self.assertEqual(text, '{"ok": true}')
        self.assertEqual(mock_openai.call_args.kwargs["api_key"], "screening-key")
        self.assertEqual(mock_openai.call_args.kwargs["base_url"], "https://screening.example.com")
        self.assertEqual(create.call_args.kwargs["model"], "screening-model")

    def test_ocr_call_uses_ocr_config_model(self) -> None:
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="姓名：张明"))])
        create = Mock(return_value=response)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        config = LLMConfig("ocr-key", "https://ocr.example.com", "ocr-model")
        images = [{"mime_type": "image/png", "base64": "ZmFrZQ=="}]

        with patch("services.llm_client.OpenAI", return_value=client) as mock_openai:
            text, error = call_vision_ocr(images, config)

        self.assertIsNone(error)
        self.assertEqual(text, "姓名：张明")
        self.assertEqual(mock_openai.call_args.kwargs["api_key"], "ocr-key")
        self.assertEqual(mock_openai.call_args.kwargs["base_url"], "https://ocr.example.com")
        self.assertEqual(create.call_args.kwargs["model"], "ocr-model")

    def test_ocr_config_uses_ocr_secrets(self) -> None:
        secrets = {
            "API_KEY": "screening-key",
            "BASE_URL": "https://screening.example.com",
            "MODEL_NAME": "screening-model",
            "OCR_API_KEY": "ocr-key",
            "OCR_BASE_URL": "https://ocr.example.com",
            "OCR_MODEL_NAME": "ocr-model",
        }

        with patch("services.llm_client._read_secret", side_effect=lambda name: secrets.get(name, "")):
            ocr_config = get_ocr_config()

        self.assertEqual(ocr_config, LLMConfig("ocr-key", "https://ocr.example.com", "ocr-model"))

    def test_ocr_config_does_not_fall_back_to_screening_config(self) -> None:
        secrets = {
            "API_KEY": "screening-key",
            "BASE_URL": "https://screening.example.com",
            "MODEL_NAME": "screening-model",
        }

        with patch("services.llm_client._read_secret", side_effect=lambda name: secrets.get(name, "")):
            ocr_config = get_ocr_config()
            text, error = call_vision_ocr([{"mime_type": "image/png", "base64": "ZmFrZQ=="}])

        self.assertEqual(ocr_config.api_key, "")
        self.assertEqual(ocr_config.base_url, "")
        self.assertEqual(ocr_config.model_name, "")
        self.assertEqual(text, "")
        self.assertIn("OCR配置缺失", error)
        self.assertIn("不会自动使用筛选模型", error)

    def test_blank_ocr_sidebar_fields_use_ocr_secret_config(self) -> None:
        default_ocr_config = LLMConfig("ocr-key", "https://ocr.example.com", "ocr-model")

        ocr_config = resolve_ocr_config("", "", "", default_ocr_config)

        self.assertEqual(ocr_config, default_ocr_config)

    def test_api_key_is_hidden_from_error_messages(self) -> None:
        config = LLMConfig("secret-test-key", "https://screening.example.com", "screening-model")

        with patch("services.llm_client.OpenAI", side_effect=Exception("bad key secret-test-key")):
            text, error = call_llm("prompt", config)

        self.assertEqual(text, "")
        self.assertNotIn("secret-test-key", error)
        self.assertIn("[API_KEY已隐藏]", error)

    def test_screening_connection_test_uses_current_config(self) -> None:
        create = Mock(return_value=SimpleNamespace(choices=[]))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        config = LLMConfig("screening-key", "https://screening.example.com", "screening-model")

        with patch("services.llm_client.OpenAI", return_value=client) as mock_openai:
            success, message = test_llm_connection(config)

        self.assertTrue(success)
        self.assertIn("测试通过", message)
        self.assertEqual(mock_openai.call_args.kwargs["api_key"], "screening-key")
        self.assertEqual(create.call_args.kwargs["model"], "screening-model")

    def test_vision_connection_test_sends_image_content(self) -> None:
        create = Mock(return_value=SimpleNamespace(choices=[]))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        config = LLMConfig("ocr-key", "https://ocr.example.com", "ocr-model")

        with patch("services.llm_client.OpenAI", return_value=client) as mock_openai:
            success, message = test_vision_connection(config)

        self.assertTrue(success)
        self.assertIn("图片输入", message)
        self.assertEqual(mock_openai.call_args.kwargs["api_key"], "ocr-key")
        self.assertEqual(create.call_args.kwargs["model"], "ocr-model")
        user_content = create.call_args.kwargs["messages"][1]["content"]
        self.assertEqual(user_content[1]["type"], "image_url")

    def test_vision_connection_test_image_meets_provider_size_floor(self) -> None:
        png_bytes = base64.b64decode(TEST_IMAGE_BASE64)
        width = int.from_bytes(png_bytes[16:20], "big")
        height = int.from_bytes(png_bytes[20:24], "big")

        self.assertGreater(width, 10)
        self.assertGreater(height, 10)

    def test_connection_test_hides_api_key_from_errors(self) -> None:
        config = LLMConfig("secret-test-key", "https://screening.example.com", "screening-model")

        with patch("services.llm_client.OpenAI", side_effect=Exception("bad key secret-test-key")):
            success, message = test_llm_connection(config)

        self.assertFalse(success)
        self.assertNotIn("secret-test-key", message)
        self.assertIn("[API_KEY已隐藏]", message)

    def test_results_to_csv_keeps_chinese_columns(self) -> None:
        csv_bytes = results_to_csv(
            [
                {
                    "文件名": "resume.txt",
                    "姓名": "张明",
                    "匹配度": 90,
                    "命中硬性条件": "Python",
                    "缺失硬性条件": "无",
                    "证据摘要": "项目经历",
                }
            ]
        )

        csv_text = csv_bytes.decode("utf-8-sig")
        self.assertIn("命中硬性条件", csv_text)
        self.assertIn("张明", csv_text)


if __name__ == "__main__":
    unittest.main()
