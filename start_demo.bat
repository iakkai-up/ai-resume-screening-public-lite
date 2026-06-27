@echo off
setlocal

set "ROOT_DIR=%~dp0"
set "APP_DIR=%ROOT_DIR%resume_screening_poc"
set "VENV_DIR=%ROOT_DIR%.venv"
set "PORT=8501"
for %%I in ("%ROOT_DIR%..\..") do set "PROJECT_USER_DIR=%%~fI"

if not exist "%APP_DIR%\app.py" (
    echo Cannot find resume_screening_poc\app.py.
    echo Please keep this launcher in the project root folder.
    pause
    exit /b 1
)

set "PYTHON_CMD="
where py >nul 2>nul
if not errorlevel 1 set "PYTHON_CMD=py -3"

if not defined PYTHON_CMD (
    where python >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD set "BUNDLED_PY=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if not defined PYTHON_CMD if not exist "%BUNDLED_PY%" set "BUNDLED_PY=%PROJECT_USER_DIR%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if not defined PYTHON_CMD if exist "%BUNDLED_PY%" (
    set "PYTHON_CMD="%BUNDLED_PY%""
)

if not defined PYTHON_CMD (
    echo Python was not found.
    echo Please install Python 3.10 or later, then run this file again.
    pause
    exit /b 1
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo First run: preparing local environment...
    %PYTHON_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Failed to create the local environment.
        pause
        exit /b 1
    )
)

set "RUN_PY=%VENV_DIR%\Scripts\python.exe"

"%RUN_PY%" -c "import streamlit" >nul 2>nul
if errorlevel 1 (
    echo Installing required packages. This may take a few minutes on first run...
    "%RUN_PY%" -m pip install -r "%APP_DIR%\requirements.txt"
    if errorlevel 1 (
        echo Failed to install required packages.
        pause
        exit /b 1
    )
)

echo Starting demo...
echo Browser address: http://localhost:%PORT%
powershell -NoProfile -Command "$client = New-Object Net.Sockets.TcpClient; try { $client.Connect('127.0.0.1', %PORT%); exit 0 } catch { exit 1 } finally { $client.Dispose() }" >nul 2>nul
if not errorlevel 1 (
    echo Demo is already running. Opening existing page...
    start "" "http://localhost:%PORT%"
    exit /b 0
)
start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 4; Start-Process 'http://localhost:%PORT%'"

cd /d "%APP_DIR%"
"%RUN_PY%" -m streamlit run app.py --server.port %PORT% --server.headless true

echo Demo stopped.
pause
