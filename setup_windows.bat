@echo off
REM Compatibility wrapper. The maintained installer is native PowerShell.
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1"
exit /b %errorlevel%
