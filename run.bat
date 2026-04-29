@echo off
echo ========================================
echo   Starting ECG Classification App
echo ========================================
echo.

:: تفعيل البيئة الافتراضية
call venv\Scripts\activate.bat

:: تشغيل التطبيق
echo Starting Streamlit server...
streamlit run app.py --server.port=8501

pause