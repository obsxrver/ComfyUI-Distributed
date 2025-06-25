@echo off
title ComfyUI Multi-GPU CLI Launcher
cd /d "%~dp0"
color 0B
cls

echo ComfyUI Multi-GPU CLI Launcher
echo ===============================

:: Check if gpu_config.json exists (relative path)
set "CONFIG_PATH=ComfyUI\custom_nodes\ComfyUI-Distributed\gpu_config.json"
if not exist "%CONFIG_PATH%" (
    color 0C
    echo Error: Configuration file not found!
    echo Expected location: %CONFIG_PATH%
    pause
    exit /b 1
)

:: Check if venv exists (relative path)
set "VENV_PATH=venv"
if not exist "%VENV_PATH%\Scripts\activate.bat" (
    color 0C
    echo Error: Virtual environment not found!
    echo Expected location: %CD%\%VENV_PATH%
    echo Make sure this batch file is in the ComfyUI_Dev directory with venv folder
    pause
    exit /b 1
)

:: Check if ComfyUI CLI is available in venv
call "%VENV_PATH%\Scripts\activate.bat" && comfy --help >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    color 0C
    echo Error: ComfyUI CLI not found in virtual environment!
    echo Make sure ComfyUI CLI is installed in: %VENV_PATH%
    pause
    exit /b 1
)

:: Read configuration with PowerShell and store in environment variables
echo Loading configuration...

powershell -NoProfile -Command "try { $json = Get-Content '%CONFIG_PATH%' -Raw | ConvertFrom-Json; $master = $json.master; $settings = $json.settings; $retryDelay = if ($settings.retry_delay_ms) { $settings.retry_delay_ms } else { 500 }; Write-Output \"RETRY_DELAY=$retryDelay\"; Write-Output \"MASTER_PORT=$($master.port)\"; Write-Output \"MASTER_CUDA=$($master.cuda_device)\"; foreach ($worker in $json.workers) { if ($worker.enabled) { Write-Output \"WORKER_$($worker.id)_NAME=$($worker.name)\"; Write-Output \"WORKER_$($worker.id)_PORT=$($worker.port)\"; Write-Output \"WORKER_$($worker.id)_CUDA=$($worker.cuda_device)\"; Write-Output \"WORKER_$($worker.id)_EXTRA=$($worker.extra_args)\" } } } catch { Write-Output \"PowerShell Error: $_\"; pause; exit 1 }" > %TEMP%\config_vars.txt

if %ERRORLEVEL% NEQ 0 (
    color 0C
    echo Error running PowerShell configuration reader!
    pause
    exit /b 1
)

for /f "tokens=*" %%i in (%TEMP%\config_vars.txt) do (
    set %%i
)

del %TEMP%\config_vars.txt >nul 2>&1

:: Check Master
echo Starting Master instance...
:MASTER
powershell -NoProfile -Command "$ErrorActionPreference = 'SilentlyContinue'; $tcpClient = New-Object System.Net.Sockets.TcpClient; $connect = $tcpClient.BeginConnect('127.0.0.1', %MASTER_PORT%, $null, $null); $wait = $connect.AsyncWaitHandle.WaitOne(200, $false); if (!$wait) { exit 1 } else { $tcpClient.EndConnect($connect); $tcpClient.Close(); exit 0 }"
if %ERRORLEVEL% == 1 (
    if not defined MASTER_STARTED (
        start wt -w 0 new-tab --title "Master - ComfyUI CLI" --startingDirectory "%CD%" cmd /k "title Master - ComfyUI CLI && call venv\Scripts\activate.bat && set CUDA_VISIBLE_DEVICES=%MASTER_CUDA% && comfy --here launch -- --port %MASTER_PORT%"
        set MASTER_STARTED=1
    )
    <nul set /p ="."
    powershell -NoProfile -Command "Start-Sleep -m %RETRY_DELAY%"
    goto MASTER
)
echo Master ready!

:: Process workers sequentially - wait for each to be ready before starting next
powershell -NoProfile -Command "try { $json = Get-Content '%CONFIG_PATH%' -Raw | ConvertFrom-Json; foreach ($worker in $json.workers) { if ($worker.enabled) { Write-Output $worker.id } } } catch { exit 1 }" > %TEMP%\worker_ids.txt

for /f %%w in (%TEMP%\worker_ids.txt) do (
    call :PROCESS_WORKER %%w
)

del %TEMP%\worker_ids.txt >nul 2>&1
goto :END_SCRIPT

:PROCESS_WORKER
set "WORKER_ID=%1"
call set "WORKER_NAME=%%WORKER_%WORKER_ID%_NAME%%"
call set "WORKER_PORT=%%WORKER_%WORKER_ID%_PORT%%"
call set "WORKER_CUDA=%%WORKER_%WORKER_ID%_CUDA%%"
call set "WORKER_EXTRA=%%WORKER_%WORKER_ID%_EXTRA%%"

echo Starting %WORKER_NAME%...

:WORKER_CHECK_LOOP
powershell -NoProfile -Command "$ErrorActionPreference = 'SilentlyContinue'; $tcpClient = New-Object System.Net.Sockets.TcpClient; $connect = $tcpClient.BeginConnect('127.0.0.1', %WORKER_PORT%, $null, $null); $wait = $connect.AsyncWaitHandle.WaitOne(200, $false); if (!$wait) { exit 1 } else { $tcpClient.EndConnect($connect); $tcpClient.Close(); exit 0 }"
if %ERRORLEVEL% == 1 (
    if not defined WORKER%WORKER_ID%_STARTED (
        start wt -w 0 new-tab --title "%WORKER_NAME% - ComfyUI CLI" --startingDirectory "%CD%" cmd /k "title %WORKER_NAME% - ComfyUI CLI && call venv\Scripts\activate.bat && set CUDA_VISIBLE_DEVICES=%WORKER_CUDA% && comfy --here launch -- --port %WORKER_PORT% %WORKER_EXTRA% --enable-cors-header"
        set WORKER%WORKER_ID%_STARTED=1
    )
    <nul set /p ="."
    powershell -NoProfile -Command "Start-Sleep -m %RETRY_DELAY%"
    goto WORKER_CHECK_LOOP
)
echo %WORKER_NAME% ready!
goto :eof

:END_SCRIPT

echo.
echo ======================================
echo All ComfyUI CLI instances are now running!
echo ======================================
echo.
echo Master: http://localhost:%MASTER_PORT%
powershell -NoProfile -Command "try { $json = Get-Content '%CONFIG_PATH%' -Raw | ConvertFrom-Json; foreach ($worker in $json.workers) { if ($worker.enabled) { Write-Output \"$($worker.name): http://localhost:$($worker.port)\" } } } catch { }"
echo.
echo Launcher will close in 5 seconds...
timeout /t 5 /nobreak >nul
exit