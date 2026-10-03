@echo off
setlocal
set "APP_DIR=%~dp0"
set "REPO_ROOT=%APP_DIR%..\..\"
set "PYTHON=%REPO_ROOT%.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
"%PYTHON%" "%APP_DIR%app.py" %*

