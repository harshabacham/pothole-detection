@echo off
title Pothole Detection System - Automated Setup & Dependency Installer
cls
set "PYTHON_EXE=C:\Users\harsh\AppData\Local\Python\pythoncore-3.14-64\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

echo ============================================================================
echo   POTHOLE DETECTION SYSTEM: ONE-CLICK AUTOMATED INSTALLER
echo ============================================================================
echo   [1/3] Installing Python dependencies from requirements.txt...
"%PYTHON_EXE%" -m pip install -r requirements.txt

echo.
echo   [2/3] Performing automated system calibration and model self-check...
"%PYTHON_EXE%" auto_setup.py

echo.
echo ============================================================================
echo   [3/3] Installation & Setup Complete!
echo   Double-click 'test_my_dashcam.bat' to test and select your road dashcam.
echo   Double-click 'start_vehicle_dashcam.bat' to begin driving with the car dashcam.
echo   Double-click 'start_web_dashboard.bat' to open the web dashboard.
echo ============================================================================
pause
