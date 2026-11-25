@echo off
REM Starts the plant disease UI and opens it in the browser (http://127.0.0.1:7860).
cd /d "%~dp0\.."
set PYTHONIOENCODING=utf-8
set GRADIO_ANALYTICS_ENABLED=False
REM Once the BioCLIP base files are cached, never touch the network (safe on offline demo machines).
if exist "%USERPROFILE%\.cache\huggingface\hub\models--imageomics--bioclip" set HF_HUB_OFFLINE=1

set PY=python
if exist "%USERPROFILE%\anaconda3\envs\aml\python.exe" set PY="%USERPROFILE%\anaconda3\envs\aml\python.exe"
if exist "%USERPROFILE%\miniconda3\envs\aml\python.exe" set PY="%USERPROFILE%\miniconda3\envs\aml\python.exe"

%PY% ui\app.py
pause
