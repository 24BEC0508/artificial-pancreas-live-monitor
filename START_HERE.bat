@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo  ARTIFICIAL PANCREAS - SETUP AND START
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Creating a project-local Python environment...
    py -3 -m venv .venv 2>nul
    if errorlevel 1 (
        python -m venv .venv
        if errorlevel 1 goto :venv_error
    )
)

echo Installing/checking project dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :install_error

echo.
echo Starting the Artificial Pancreas Control Lab...
echo Keep this window open while using the application.
echo Press Ctrl+C here to stop the server.
echo.
".venv\Scripts\python.exe" -m streamlit run app.py
goto :end

:venv_error
echo.
echo ERROR: Could not create the virtual environment.
echo Install Python 3.10+ and ensure python or py is available on PATH.
pause
goto :end

:install_error
echo.
echo ERROR: Dependency installation failed. Check your internet connection
and the Python version, then run START_HERE.bat again.
pause
goto :end

:end
endlocal
