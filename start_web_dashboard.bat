@echo off
title Pothole Detection & Citizen Grievance Web Application
cls
set "PYTHON_EXE=C:\Users\harsh\AppData\Local\Python\pythoncore-3.14-64\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

echo ============================================================================
echo   LAUNCHING POTHOLE DETECTION & GOVERNMENT GRIEVANCE SYSTEM
echo ============================================================================
echo   [1/2] Verifying automated pipeline calibration and model readiness...
"%PYTHON_EXE%" auto_setup.py

echo.
echo   [2/2] Launching interactive Web Dashboard at http://localhost:8501 ...
echo ============================================================================
"%PYTHON_EXE%" -m streamlit run app.py
pause
