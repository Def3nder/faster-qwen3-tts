@echo off
setlocal

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%" || exit /b 1

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Python environment not found at .venv\Scripts\python.exe
    echo Run setup_windows.bat first.
    exit /b 1
)

".venv\Scripts\python.exe" -m pytest -q ^
    tests\test_generate_mp3_with_embedding.py ^
    --cov=generate ^
    --cov-branch ^
    --cov-report=term-missing

set "TEST_EXIT_CODE=%ERRORLEVEL%"
exit /b %TEST_EXIT_CODE%
