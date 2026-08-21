@echo off
chcp 65001 >nul
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-development.ps1"
if errorlevel 1 (
  echo.
  echo 开发环境准备失败，请查看上面的错误信息。
  pause
  exit /b 1
)
echo.
pause
