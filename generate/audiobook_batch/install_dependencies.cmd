@echo off
setlocal
set "APP_DIR=%~dp0"
set "REPO_ROOT=%APP_DIR%..\..\"
set "PYTHON=%REPO_ROOT%.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
where uv >nul 2>nul
if %errorlevel% equ 0 (
    uv pip install --cache-dir "%APP_DIR%.uv-cache" --python "%PYTHON%" -r "%APP_DIR%requirements.txt"
) else (
    "%PYTHON%" -m pip install -r "%APP_DIR%requirements.txt"
)
pause
