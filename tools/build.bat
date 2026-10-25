@echo off
rem Configure + build the ipf tests/examples with MSVC and Ninja.
rem Usage: tools\build.bat [Debug|Release] [17|20] [asan]   (asan = AddressSanitizer build)
setlocal
set CONFIG=%~1
if "%CONFIG%"=="" set CONFIG=Release
set STD=%~2
if "%STD%"=="" set STD=17
set EXTRA=
if /i "%~3"=="asan" set EXTRA=-DIPF_SANITIZE=ON
set SUFFIX=
if not "%EXTRA%"=="" set SUFFIX=-asan
set ROOT=%~dp0..
set BUILD=%ROOT%\build\%CONFIG%-cpp%STD%%SUFFIX%
set VSROOT=C:\Program Files\Microsoft Visual Studio\18\Community
set VSCMAKE=%VSROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake
call "%VSROOT%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul || exit /b 1
"%VSCMAKE%\CMake\bin\cmake.exe" -S "%ROOT%" -B "%BUILD%" -G Ninja -DCMAKE_MAKE_PROGRAM="%VSCMAKE%\Ninja\ninja.exe" -DCMAKE_BUILD_TYPE=%CONFIG% -DCMAKE_CXX_STANDARD=%STD% -DCMAKE_CXX_COMPILER=cl %EXTRA% >nul || exit /b 1
"%VSCMAKE%\CMake\bin\cmake.exe" --build "%BUILD%" || exit /b 1
