@echo off
title Autonomous Dashcam AI - Project Evaluation Showcase
color 0A
cls

echo ======================================================================
echo    AUTONOMOUS VEHICLE DASHCAM AI ROAD ANOMALY SURVEILLANCE SYSTEM
echo        Government of India Grievance & ADAS Engineering Platform
echo ======================================================================
echo.
echo [1/3] Detecting Python runtime environment...

set "PYTHON_EXE=C:\Users\harsh\AppData\Local\Python\pythoncore-3.14-64\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

echo [OK] Python runtime located: %PYTHON_EXE%
echo.

echo [2/3] Executing Real-Life Automotive Edge Verification...
echo ----------------------------------------------------------------------
"%PYTHON_EXE%" scripts\test_real_life_modules.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Real-life test suite failed.
    pause
    exit /b %ERRORLEVEL%
)
echo.

echo [3/3] Executing Municipal PWD Work Orders & 3D GIS Verification...
echo ----------------------------------------------------------------------
"%PYTHON_EXE%" scripts\test_municipal_and_gis.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Municipal GIS test suite failed.
    pause
    exit /b %ERRORLEVEL%
)
echo.

echo ======================================================================
echo   *** ALL REAL-WORLD & MUNICIPAL ENGINES 100% OPERATIONAL ***
echo ======================================================================
echo.
echo Launching Interactive Streamlit Web Dashboard for Examiner Panel...
echo (Opening in your default browser: http://localhost:8501)
echo.
"%PYTHON_EXE%" -m streamlit run app.py
pause
