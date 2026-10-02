@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Slinkify
set "PY="

rem Prefer an explicitly activated environment, then a project environment, then PATH.
if defined VIRTUAL_ENV call :try_python "%VIRTUAL_ENV%\Scripts\python.exe"
if defined CONDA_PREFIX call :try_python "%CONDA_PREFIX%\python.exe"
call :try_python "%~dp0.venv\Scripts\python.exe"
call :try_python "python"
if not defined PY (
  for /f "delims=" %%I in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do call :try_python "%%I"
)

rem Conda is a fallback. Activate it first so native package DLLs are available.
if not defined PY call :try_conda "%USERPROFILE%\anaconda3"
if not defined PY call :try_conda "%USERPROFILE%\miniconda3"
if not defined PY call :try_conda "C:\ProgramData\anaconda3"
if not defined PY call :try_conda "C:\ProgramData\miniconda3"

if not defined PY (
  echo Could not find a working Python 3.10 or newer.
  echo Install Python from python.org or Anaconda, then run this file again.
  echo.
  pause
  exit /b 1
)

echo Python in use:
"%PY%" -c "import sys; print(sys.version.split()[0], sys.executable)"
echo.

rem Verify real imports, including Pillow used by previews, before opening the app.
"%PY%" -c "import gradio, manifold3d, ezdxf, scipy, matplotlib, numpy, PIL; from packaging.version import Version; assert Version('6.28') <= Version(gradio.__version__) < Version('7')" >nul 2>nul
if errorlevel 1 (
  echo Installing the packages in requirements.txt - this may take a few minutes...
  echo.
  "%PY%" -m pip install -r "%~dp0requirements.txt"
  if errorlevel 1 goto :package_error
  "%PY%" -c "import gradio, manifold3d, ezdxf, scipy, matplotlib, numpy, PIL; from packaging.version import Version; assert Version('6.28') <= Version(gradio.__version__) < Version('7')"
  if errorlevel 1 goto :package_error
)

echo Starting Slinkify. The app will print its local browser address below.
echo Keep this window open while you use the app; close it to stop the app.
echo.
"%PY%" "%~dp0slinky_app.py" %*
set "SLINKY_EXIT=%ERRORLEVEL%"
echo.
if "%SLINKY_EXIT%"=="0" (echo Slinkify stopped.) else (echo Slinkify exited with error %SLINKY_EXIT%. See the messages above.)
pause
exit /b %SLINKY_EXIT%

:package_error
echo.
echo Package installation or import failed - see the messages above.
echo Check this Python environment, then run the launcher again.
pause
exit /b 1

:try_python
if defined PY exit /b 0
"%~1" -c "import sys; assert sys.version_info >= (3, 10)" >nul 2>nul
if not errorlevel 1 set "PY=%~1"
exit /b 0

:try_conda
if not exist "%~1\python.exe" exit /b 0
if exist "%~1\Scripts\activate.bat" call "%~1\Scripts\activate.bat" "%~1"
call :try_python "%~1\python.exe"
exit /b 0
