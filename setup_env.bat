@echo off
echo ========================================
echo   Creating Virtual Environment for ECG App
echo ========================================
echo.

:: إنشاء البيئة الافتراضية
echo Creating virtual environment...
python -m venv venv

:: تفعيل البيئة
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: تثبيت المتطلبات
echo Installing requirements...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ========================================
echo   Setup Complete!
echo   Run the app with: streamlit run app.py
echo ========================================
pause