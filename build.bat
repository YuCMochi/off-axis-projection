@echo off
echo Building Off-Axis Face Tracker...
.venv\Scripts\python -m nuitka app.py ^
    --standalone ^
    --enable-plugin=numpy ^
    --enable-plugin=tk-inter ^
    --include-package=mediapipe ^
    --include-package=cv2 ^
    --include-package-data=mediapipe ^
    --include-package-data=cv2 ^
    --windows-disable-console ^
    --output-filename=OffAxisTracker ^
    --output-dir=dist
echo.
echo Done. Output: dist\app.dist\OffAxisTracker.exe
pause
