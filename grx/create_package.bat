REM build package whl file and locate it at <folder contains simnext clone>\outputs

pip wheel "%~dp0." -w "%~dp0..\..\..\outputs\wheels" --no-deps
REM Remove build directory if it exists
if exist "%~dp0build" (
    rmdir /s /q "%~dp0build"
)
pause