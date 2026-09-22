@echo off
setlocal
cd /d "%~dp0"

set "DASHBOARD_PYTHON=%LOCALAPPDATA%\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe"
if not exist "%DASHBOARD_PYTHON%" set "DASHBOARD_PYTHON=python"
"%DASHBOARD_PYTHON%" "%~dp0launch_train_monitoring.py" %*
if errorlevel 1 pause

:finished
endlocal
