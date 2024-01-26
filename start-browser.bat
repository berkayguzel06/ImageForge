@echo off

:: Set the project name
set PROJECT_NAME=ImageForce

echo Installing and running %PROJECT_NAME%...

:: Check if .venv already exists
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
) else (
    echo Virtual environment already exists. Skipping creation.
)

:: Activate virtual environment
echo Activating virtual environment...
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    call .venv\Scripts\activate
)


:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

:: Start the user interface
echo Starting %PROJECT_NAME% user interface...
start /B python app.py

:: Inform the user
echo %PROJECT_NAME% is running. Please check your browser.

deactivate

:: Pause to keep the command prompt window open
pause
