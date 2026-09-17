@echo off
title LunorAI - Intelligent Knowledge Assistant
cd /d "%~dp0"

echo ============================================================
echo   Launching LunorAI - Intelligent Knowledge Assistant
echo ============================================================
echo.

if not exist ".venv\Scripts\streamlit.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\streamlit.exe
    pause
    exit /b 1
)

echo Starting Streamlit server...
echo Access the app at: http://localhost:8501
echo.
call .venv\Scripts\streamlit.exe run app.py
pause
