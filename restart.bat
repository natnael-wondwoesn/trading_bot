@echo off
echo Killing existing processes...
taskkill /f /im python.exe 2>nul
timeout /t 2 /nobreak >nul

echo Starting production system...
python production_main.py
