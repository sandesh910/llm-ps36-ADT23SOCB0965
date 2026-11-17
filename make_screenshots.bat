@echo off
cd /d "%~dp0"
taskkill /F /IM streamlit.exe >nul 2>nul
call venv\Scripts\activate
pip install -q playwright > shots_log.txt 2>&1
python make_screenshots.py >> shots_log.txt 2>&1
