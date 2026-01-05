@echo off
call venv\Scripts\activate
echo Starting EmotionCamera Web Interface...
streamlit run source\streamlit_app.py > streamlit_log.txt 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo An error occurred. Please check streamlit_log.txt
)
pause
