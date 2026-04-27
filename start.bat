@echo off
echo =====================================================
echo   Advanced Certificate Authenticity Verification System
echo =====================================================
echo.

REM Create required directories
if not exist "uploads" mkdir uploads
if not exist "reports" mkdir reports
if not exist "reference_templates" mkdir reference_templates

echo [1/2] Installing dependencies...
pip install -r requirements.txt
echo.

echo [2/2] Starting server on http://localhost:8003
echo.
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8003
pause
