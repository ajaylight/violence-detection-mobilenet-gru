@echo off
setlocal

cd /d "%~dp0"

if not exist "logs" mkdir "logs"

"C:\vd310\Scripts\python.exe" ".\training\model_server.py" 1>> ".\logs\model-server.out.log" 2>> ".\logs\model-server.err.log"
