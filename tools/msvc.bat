@echo off
rem Runs a command inside the MSVC x64 developer environment.
rem Usage: tools\msvc.bat <command> [args...]
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
%*
