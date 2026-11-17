@echo off
cd /d "%~dp0"
where python >nul 2>nul || (echo Python not found. Install it from python.org and tick "Add to PATH". & pause & exit /b)
if not exist venv (
  echo Creating virtual environment...
  python -m venv venv
)
call venv\Scripts\activate
echo Installing packages...
pip install -q -r requirements.txt
if not exist .env copy .env.example .env >nul
echo Starting the app at http://localhost:8501 ...
streamlit run app.py --server.headless true
pause
