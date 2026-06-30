@echo off
setlocal

REM Resolve the directory of this script
set SCRIPT_DIR=%~dp0

REM Activate virtual environment
call "%SCRIPT_DIR\..\..\..\..\.venv\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment.
	pause
    exit /b 1
)

REM Run pytest on unit_tests folder
pytest -v unit_tests -o python_files=ut_*.py

REM Remove .pytest_cache folder 
REM if exist .pytest_cache (
REM   rmdir /s /q .pytest_cache
REM )

endlocal

pause