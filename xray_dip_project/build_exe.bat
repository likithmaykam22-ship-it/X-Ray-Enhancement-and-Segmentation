@echo off
setlocal
python -m pip install -r requirements.txt
python -m PyInstaller --noconfirm --clean --windowed --onedir --name XRayDIPLab xray_dip_app.py
 echo.
echo Build complete. Run dist\XRayDIPLab\XRayDIPLab.exe
echo.
pause
