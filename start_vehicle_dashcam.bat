@echo off
title Car Dashcam Autonomous Road Pothole Scanner & Gov Dispatcher
cls
set "PYTHON_EXE=C:\Users\harsh\AppData\Local\Python\pythoncore-3.14-64\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

echo ============================================================================
echo   CAR DASHCAM AUTONOMOUS POTHOLE SCANNER & GOV DISPATCHER
echo ============================================================================
echo   [1/2] Checking system self-calibration and camera hardware...
"%PYTHON_EXE%" auto_setup.py

echo.
echo   [2/2] Starting autonomous road surveillance & real-time government reporting...
echo   Press 'q' in the camera window to stop anytime.
echo ============================================================================
"%PYTHON_EXE%" scripts\run_vehicle_dashcam.py --source auto --portal NHAI --min-severity Medium --show-display
pause
