@echo off
title 1-Click Car Dashcam Setup & Preview Wizard
cls
set "PYTHON_EXE=C:\Users\harsh\AppData\Local\Python\pythoncore-3.14-64\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

echo ============================================================================
echo   1-CLICK CAR DASHBOARD CAMERA SETUP WIZARD
echo ============================================================================
echo   Probing camera devices and launching live preview...
echo ============================================================================
"%PYTHON_EXE%" scripts\test_camera.py
pause
