@echo off
echo Building Off-Axis Face Tracker...
.venv\Scripts\python -m nuitka app.py
echo.
echo Done. Output: dist\app.dist\OffAxisTracker.exe
pause
