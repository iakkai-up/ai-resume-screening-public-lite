# AI Resume Screening PoC Public Lite

This is a local demo for AI-assisted resume screening. It is designed for evaluation, learning, and personal demo use.

Important:

> The AI output is only a reference for HR review. It must not be used as an automatic hiring, rejection, or ranking decision.

This public repository is a simplified release. The private PoC keeps non-public prompt experiments, evaluation reports, full test matrices, and commercial planning notes.

## What It Does

- Paste a job description.
- Upload multiple resumes.
- Supports PDF, DOCX, TXT, PNG, JPG, and JPEG files.
- Uses a configured vision/OCR model for image resumes and scanned PDFs.
- Calls an OpenAI-compatible chat completion API for screening.
- Shows candidate ranking, match reasons, risks, and interview questions.
- Exports results to CSV.
- Includes Mock mode, so the demo can run without an API key.

OCR quality depends on the configured vision/OCR model and valid API credentials.

## Quick Start

On Windows, double-click:

```text
启动Demo.bat
```

If the browser does not open automatically, visit:

```text
http://localhost:8501
```

The launcher creates a local `.venv` and installs dependencies on first run.

## API Setup

Copy the example config:

```powershell
Copy-Item resume_screening_poc\.streamlit\secrets.toml.example resume_screening_poc\.streamlit\secrets.toml
```

Edit:

```text
resume_screening_poc\.streamlit\secrets.toml
```

Use your own API key, base URL, and model names. Do not commit `secrets.toml`.

Minimal text-screening config:

```toml
API_KEY = "your_api_key"
BASE_URL = "https://your-openai-compatible-endpoint.example.com"
MODEL_NAME = "your-text-model"
MODEL_OPTIONS = ["your-text-model"]
```

Optional OCR config for image resumes and scanned PDFs:

```toml
OCR_API_KEY = "your_ocr_api_key"
OCR_BASE_URL = "https://your-openai-compatible-endpoint.example.com"
OCR_MODEL_NAME = "your-vision-model"
OCR_MODEL_OPTIONS = ["your-vision-model"]
```

You can also enter temporary API settings in the app sidebar. Those values are session-only and are not written back to disk.

## Mock Mode

Mock mode is enabled by default.

- No model API call.
- No API key required.
- No real resume upload required.
- Returns sample candidate results for a stable demo.

To use real screening, disable Mock mode, paste a JD, upload sample or synthetic resumes, and configure your own model API.

## Sample Data

The `resume_screening_poc/sample_data` folder includes synthetic examples only. Do not upload real candidate resumes to this repository.

The public release intentionally does not include the full private evaluation set.

## Tests

Run from the repository root:

```powershell
python -m unittest discover -s resume_screening_poc\tests
```

If you use the launcher-created environment:

```powershell
.venv\Scripts\python.exe -m unittest discover -s resume_screening_poc\tests
```

## License

This repository is source-available, not open source. See [LICENSE](LICENSE).

In short: personal evaluation and demo use are allowed; commercial resale, hosted services, or copying this as a competing product require written permission from Kevin.

## Safety Notes

- Do not commit API keys, model credentials, real resumes, logs, or private HR data.
- Do not use protected or irrelevant personal attributes for scoring.
- Do not treat AI output as an automatic employment decision.
- Consult legal/compliance professionals before using this in a real hiring process.
