@echo off
REM Wrapper for start-local-review.ps1. Examples:
REM   start-local-review.bat -Root "E:\Photos\2026-10-01" -Output "D:\photo-review\2026-10-01"
REM   start-local-review.bat -Output "D:\photo-review\2026-10-01" -Reopen
chcp 65001 >nul
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-local-review.ps1" %*
set CODE=%errorlevel%
if not "%CODE%"=="0" pause
exit /b %CODE%
