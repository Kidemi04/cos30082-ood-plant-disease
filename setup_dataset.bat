@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
if not defined CONDA_ENV set "CONDA_ENV=aml"

rem Prefer the conda environment (default "aml"; override with: set CONDA_ENV=name).
set "PY="
for %%R in ("%CONDA_PREFIX%\..\.." "%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3") do (
    if not defined PY if exist "%%~R\envs\%CONDA_ENV%\python.exe" set "PY=%%~R\envs\%CONDA_ENV%\python.exe"
)
rem Already inside the target env (e.g. Anaconda Prompt with "conda activate aml").
if not defined PY if /i "%CONDA_DEFAULT_ENV%"=="%CONDA_ENV%" if exist "%CONDA_PREFIX%\python.exe" set "PY=%CONDA_PREFIX%\python.exe"

if defined PY (
    echo Using conda environment "%CONDA_ENV%": %PY%
    "%PY%" scripts\bootstrap_dataset.py %*
) else (
    py -3 -c "import sys; assert sys.version_info >= (3, 10)" >nul 2>&1
    if not errorlevel 1 (
        py -3 scripts\bootstrap_dataset.py %*
    ) else (
        python -c "import sys; assert sys.version_info >= (3, 10)" >nul 2>&1
        if errorlevel 1 (
            echo Conda environment "%CONDA_ENV%" not found, and no Python 3.10+ on PATH.
            pause
            exit /b 1
        )
        python scripts\bootstrap_dataset.py %*
    )
)
set "RESULT=%ERRORLEVEL%"
echo.
if "%RESULT%"=="0" (echo Dataset setup succeeded.) else (echo Dataset setup failed. See the message above.)
pause
exit /b %RESULT%
