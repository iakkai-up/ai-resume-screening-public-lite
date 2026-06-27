# Public Release Checklist

Use this checklist before creating the GitHub public repository.

## Content Check

- No `.git` history copied from the private repo.
- No `.streamlit/secrets.toml`.
- No API keys, tokens, passwords, or private endpoint credentials.
- No real resumes, HR records, logs, screenshots, or exports.
- No private research report, full test matrix, private TODO, or commercial planning note.
- Only synthetic sample data is included.

## Local Verification

From this folder:

```powershell
python -m unittest discover -s resume_screening_poc\tests
```

Run the demo:

```powershell
.\启动Demo.bat
```

Confirm:

- Mock mode works without API credentials.
- Real screening requires the user's own OpenAI-compatible API settings.
- OCR wording clearly says quality depends on the chosen vision/OCR model and API credentials.

## Clean Public Repo

Create the public GitHub repository from this folder as a fresh repository:

```powershell
git init
git add .
git commit -m "Initial public-lite release"
git branch -M main
git remote add origin https://github.com/<your-account>/<public-repo>.git
git push -u origin main
```

Do not push from a repo that contains the private PoC history.
