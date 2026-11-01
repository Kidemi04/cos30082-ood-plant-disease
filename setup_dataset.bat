@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
py -3 -c "import sys; assert sys.version_info >= (3, 10)" >nul 2>&1
if not errorlevel 1 (
    py -3 scripts\bootstrap_dataset.py %*
) else (
    python -c "import sys; assert sys.version_info >= (3, 10)" >nul 2>&1
    if errorlevel 1 (
        echo Python 3.10 or newer is required. Install Python from python.org.
        pause
        exit /b 1
    )
    python scripts\bootstrap_dataset.py %*
)
set "RESULT=%ERRORLEVEL%"
echo.
if "%RESULT%"=="0" (echo Dataset setup succeeded.) else (echo Dataset setup failed. See the message above.)
pause
exit /b %RESULT%
