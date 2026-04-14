@echo off
echo Building Off-Axis Face Tracker...
.venv\Scripts\pyinstaller off_axis_tracker.spec --clean --noconfirm
echo.
echo Done. Output: dist\OffAxisTracker\OffAxisTracker.exe
pause
