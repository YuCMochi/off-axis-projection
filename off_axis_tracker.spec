# off_axis_tracker.spec
# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# Collect mediapipe bundled model files and OpenCV data
mp_datas   = collect_data_files("mediapipe")
cv2_datas  = collect_data_files("cv2")

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=collect_dynamic_libs("mediapipe"),
    datas=mp_datas + cv2_datas + [('face_landmarker.task', '.')],
    hiddenimports=[
        "mediapipe",
        "mediapipe.python",
        "mediapipe.python.solutions",
        "mediapipe.python.solutions.face_mesh",
        "cv2",
        "numpy",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="OffAxisTracker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,       # no console window (set True if you want debug console)
    icon=None,           # add path to .ico here if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="OffAxisTracker",
)
