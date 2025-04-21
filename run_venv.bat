@echo off

REM Switch to the directory where the script is located
cd /d "%~dp0"

REM Check if venv exists
IF NOT EXIST "venv\Scripts\activate.bat" (
    echo.
    echo venv not found, creating...
    python -m venv venv
    echo venv created successfully
)

REM Open a new window and activate the virtual environment
echo Launching a new terminal window with the virtual environment...
start "" cmd /k "cd /d \"%~dp0\" & call venv\Scripts\activate.bat & echo venv activated"

echo.
echo New window launched. %~n0 will exit automatically in 5 seconds...
timeout /t 5 >nul
exit /b
