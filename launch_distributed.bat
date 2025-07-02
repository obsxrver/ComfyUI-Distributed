@echo off
title ComfyUI Unified Launcher
cd /d "%~dp0"
color 0B
cls

echo.
echo  ================================================================
echo                    ComfyUI Unified Launcher                      
echo  ================================================================
echo.

:: Check command line parameter
set "MODE=%1"
if "%MODE%"=="" goto :SHOW_MENU
if /i "%MODE%"=="master" goto :SETUP
if /i "%MODE%"=="all-slow" goto :SETUP
if /i "%MODE%"=="all-fast" goto :SETUP
if /i "%MODE%"=="venv" goto :ACTIVATE_VENV
if /i "%MODE%"=="update" goto :UPDATE_COMFYUI
if /i "%MODE%"=="update-nodes" goto :UPDATE_NODES

echo.
echo  ERROR: Invalid parameter: %MODE%
echo.
timeout /t 2 /nobreak >nul
goto :SHOW_MENU

:SHOW_MENU
color 0B
echo  ----------------------------------------------------------------
echo           Launch Mode Selection                         
echo  ----------------------------------------------------------------
echo.
echo   [1] Master only
echo       ^> Launch only the master instance
echo.
echo   [2] All instances (fast)
echo       ^> Launch all instances with 2-second delays
echo.
echo   [3] All instances (sequential)
echo       ^> Launch all instances, wait for each to be ready
echo.
echo   [4] Activate virtual environment
echo       ^> Open command prompt with venv activated
echo.
echo   [5] Update ComfyUI
echo       ^> Update ComfyUI to latest version
echo.
echo   [6] Update custom nodes
echo       ^> Update all custom nodes to latest versions
echo.
echo   [Q] Quit
echo.
echo  ----------------------------------------------------------------
echo   TIP: You can also launch directly from command line:
echo        launch_comfyui.bat master
echo        launch_comfyui.bat all-slow
echo        launch_comfyui.bat all-fast
echo        launch_comfyui.bat venv
echo        launch_comfyui.bat update
echo        launch_comfyui.bat update-nodes
echo  ----------------------------------------------------------------
echo.

:: Use CHOICE to wait for a single keypress without needing Enter
:: /C specifies the allowed characters. /N hides the default "[1,2,3,4,5,6,Q]?" prompt. /M sets a custom message.
choice /c 123456Q /n /m "  ^> Enter your choice: "
echo.

:: Check the ERRORLEVEL. IMPORTANT: Must check from highest to lowest!
:: CHOICE sets ERRORLEVEL to the index of the character in /C
:: 1=1, 2=2, 3=3, 4=4, 5=5, 6=6, Q=7

if errorlevel 7 goto :QUIT_CHOICE
if errorlevel 6 goto :UPDATE_NODES
if errorlevel 5 goto :UPDATE_COMFYUI
if errorlevel 4 goto :ACTIVATE_VENV
if errorlevel 3 (
    set "MODE=all-slow"
    goto :SETUP
)
if errorlevel 2 (
    set "MODE=all-fast"
    goto :SETUP
)
if errorlevel 1 (
    set "MODE=master"
    goto :SETUP
)

:: This part of the script should now be unreachable, but it's good practice to handle it.
echo.
echo  ERROR: Invalid choice.
echo.
timeout /t 2 /nobreak >nul
goto :SHOW_MENU

:QUIT_CHOICE
echo Exiting...
timeout /t 1 /nobreak >nul
exit /b 0

:SETUP
echo Mode: %MODE%
echo.

:: Check if gpu_config.json exists (relative path)
set "CONFIG_PATH=ComfyUI\custom_nodes\ComfyUI-Distributed\gpu_config.json"
if not exist "%CONFIG_PATH%" (
    color 0C
    echo Error: Configuration file not found!
    echo Expected location: %CONFIG_PATH%
    echo.
    echo Make sure ComfyUI-Distributed custom node is installed.
    pause
    exit /b 1
)

:: Check if venv exists (relative path)
set "VENV_PATH=venv"
if not exist "%VENV_PATH%\Scripts\activate.bat" (
    color 0C
    echo Error: Virtual environment not found!
    echo Expected location: %CD%\%VENV_PATH%
    echo Make sure this batch file is in the ComfyUI directory with venv folder
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

:: Jump to appropriate mode
if /i "%MODE%"=="master" goto :MODE_MASTER
if /i "%MODE%"=="all-slow" goto :MODE_ALL_SLOW
if /i "%MODE%"=="all-fast" goto :MODE_ALL_FAST

::=============================================================================
:: MASTER MODE - Launch only the master instance
::=============================================================================
:MODE_MASTER
echo Configuring memory management for large models...
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo Memory management configured.

echo Loading master configuration...

:: Read master configuration
powershell -NoProfile -Command "try { $json = Get-Content '%CONFIG_PATH%' -Raw | ConvertFrom-Json; $master = $json.master; Write-Output \"MASTER_PORT=$($master.port)\"; Write-Output \"MASTER_CUDA=$($master.cuda_device)\"; Write-Output \"MASTER_EXTRA=$($master.extra_args)\" } catch { Write-Output \"PowerShell Error: $_\"; pause; exit 1 }" > %TEMP%\master_config.txt

if %ERRORLEVEL% NEQ 0 (
    color 0C
    echo Error reading master configuration!
    pause
    exit /b 1
)

for /f "tokens=*" %%i in (%TEMP%\master_config.txt) do (
    set %%i
)

del %TEMP%\master_config.txt >nul 2>&1

:: Display master configuration
echo.
echo Master Configuration:
echo   Port: %MASTER_PORT%
echo   CUDA Device: %MASTER_CUDA%
echo   Extra Args: %MASTER_EXTRA%
echo.

:: Check if master is already running
echo Checking if master is already running...
powershell -NoProfile -Command "$ErrorActionPreference = 'SilentlyContinue'; $tcpClient = New-Object System.Net.Sockets.TcpClient; $connect = $tcpClient.BeginConnect('127.0.0.1', %MASTER_PORT%, $null, $null); $wait = $connect.AsyncWaitHandle.WaitOne(200, $false); if (!$wait) { exit 1 } else { $tcpClient.EndConnect($connect); $tcpClient.Close(); exit 0 }"

if %ERRORLEVEL% == 0 (
    color 0E
    echo Master is already running on port %MASTER_PORT%!
    echo URL: http://localhost:%MASTER_PORT%
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 0
)

:: Launch master
echo Launching Master ComfyUI instance...

:: Try Windows Terminal first, fall back to cmd
wt -w 0 new-tab --title "Master - ComfyUI CLI" --startingDirectory "%CD%" cmd /k "title Master - ComfyUI CLI && echo CUDA_VISIBLE_DEVICES=%MASTER_CUDA% && call venv\Scripts\activate.bat && set CUDA_VISIBLE_DEVICES=%MASTER_CUDA% && echo Starting ComfyUI Master... && comfy --here launch -- --port %MASTER_PORT% %MASTER_EXTRA%" 2>nul

if %ERRORLEVEL% NEQ 0 (
    echo Windows Terminal not available, using regular command prompt...
    start "Master - ComfyUI CLI" cmd /k "title Master - ComfyUI CLI && cd /d "%CD%" && echo CUDA_VISIBLE_DEVICES=%MASTER_CUDA% && call venv\Scripts\activate.bat && set CUDA_VISIBLE_DEVICES=%MASTER_CUDA% && echo Starting ComfyUI Master... && comfy --here launch -- --port %MASTER_PORT% %MASTER_EXTRA%"
)

echo.
echo ======================================
echo Master ComfyUI launcher started!
echo ======================================
echo.
echo Expected URL: http://localhost:%MASTER_PORT%
echo CUDA Device: %MASTER_CUDA%
echo.
echo The ComfyUI window will open shortly.
echo This launcher will close in 1 second...

timeout /t 1 /nobreak >nul
exit

::=============================================================================
:: ALL-SLOW MODE - Launch all instances sequentially
::=============================================================================
:MODE_ALL_SLOW
echo Configuring memory management for large models...
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo Memory management configured.

:: Read configuration with PowerShell and store in environment variables
echo Loading configuration...

powershell -NoProfile -Command "try { $json = Get-Content '%CONFIG_PATH%' -Raw | ConvertFrom-Json; $master = $json.master; $settings = $json.settings; $retryDelay = if ($settings.retry_delay_ms) { $settings.retry_delay_ms } else { 500 }; Write-Output \"RETRY_DELAY=$retryDelay\"; Write-Output \"MASTER_PORT=$($master.port)\"; Write-Output \"MASTER_CUDA=$($master.cuda_device)\"; Write-Output \"MASTER_EXTRA=$($master.extra_args)\"; foreach ($worker in $json.workers) { if ($worker.enabled -and -not $worker.host) { Write-Output \"WORKER_$($worker.id)_NAME=$($worker.name)\"; Write-Output \"WORKER_$($worker.id)_PORT=$($worker.port)\"; Write-Output \"WORKER_$($worker.id)_CUDA=$($worker.cuda_device)\"; Write-Output \"WORKER_$($worker.id)_EXTRA=$($worker.extra_args)\" } } } catch { Write-Output \"PowerShell Error: $_\"; pause; exit 1 }" > %TEMP%\config_vars.txt

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
:MASTER_SLOW
powershell -NoProfile -Command "$ErrorActionPreference = 'SilentlyContinue'; $tcpClient = New-Object System.Net.Sockets.TcpClient; $connect = $tcpClient.BeginConnect('127.0.0.1', %MASTER_PORT%, $null, $null); $wait = $connect.AsyncWaitHandle.WaitOne(200, $false); if (!$wait) { exit 1 } else { $tcpClient.EndConnect($connect); $tcpClient.Close(); exit 0 }"
if %ERRORLEVEL% == 1 (
    if not defined MASTER_STARTED (
        echo Starting Master with CUDA_VISIBLE_DEVICES=%MASTER_CUDA%
        start wt -w 0 new-tab --title "Master - ComfyUI CLI" --startingDirectory "%CD%" cmd /k "title Master - ComfyUI CLI && echo CUDA_VISIBLE_DEVICES=%MASTER_CUDA% && call venv\Scripts\activate.bat && set CUDA_VISIBLE_DEVICES=%MASTER_CUDA% && comfy --here launch -- --port %MASTER_PORT% %MASTER_EXTRA%"
        set MASTER_STARTED=1
    )
    <nul set /p ="."
    powershell -NoProfile -Command "Start-Sleep -m %RETRY_DELAY%"
    goto MASTER_SLOW
)
echo Master ready!

:: Process workers sequentially - wait for each to be ready before starting next
powershell -NoProfile -Command "try { $json = Get-Content '%CONFIG_PATH%' -Raw | ConvertFrom-Json; foreach ($worker in $json.workers) { if ($worker.enabled -and -not $worker.host) { Write-Output $worker.id } } } catch { exit 1 }" > %TEMP%\worker_ids.txt

for /f %%w in (%TEMP%\worker_ids.txt) do (
    call :PROCESS_WORKER_SLOW %%w
)

del %TEMP%\worker_ids.txt >nul 2>&1
goto :FINISH_ALL

:PROCESS_WORKER_SLOW
set "WORKER_ID=%1"
call set "WORKER_NAME=%%WORKER_%WORKER_ID%_NAME%%"
call set "WORKER_PORT=%%WORKER_%WORKER_ID%_PORT%%"
call set "WORKER_CUDA=%%WORKER_%WORKER_ID%_CUDA%%"
call set "WORKER_EXTRA=%%WORKER_%WORKER_ID%_EXTRA%%"

echo Starting %WORKER_NAME%...

:WORKER_CHECK_LOOP_SLOW
powershell -NoProfile -Command "$ErrorActionPreference = 'SilentlyContinue'; $tcpClient = New-Object System.Net.Sockets.TcpClient; $connect = $tcpClient.BeginConnect('127.0.0.1', %WORKER_PORT%, $null, $null); $wait = $connect.AsyncWaitHandle.WaitOne(200, $false); if (!$wait) { exit 1 } else { $tcpClient.EndConnect($connect); $tcpClient.Close(); exit 0 }"
if %ERRORLEVEL% == 1 (
    if not defined WORKER%WORKER_ID%_STARTED (
        echo Starting %WORKER_NAME% with CUDA_VISIBLE_DEVICES=%WORKER_CUDA%
        start wt -w 0 new-tab --title "%WORKER_NAME% - ComfyUI CLI" --startingDirectory "%CD%" cmd /k "title %WORKER_NAME% - ComfyUI CLI && echo CUDA_VISIBLE_DEVICES=%WORKER_CUDA% && call venv\Scripts\activate.bat && set CUDA_VISIBLE_DEVICES=%WORKER_CUDA% && comfy --here launch -- --port %WORKER_PORT% %WORKER_EXTRA% --enable-cors-header"
        set WORKER%WORKER_ID%_STARTED=1
    )
    <nul set /p ="."
    powershell -NoProfile -Command "Start-Sleep -m %RETRY_DELAY%"
    goto WORKER_CHECK_LOOP_SLOW
)
echo %WORKER_NAME% ready!
goto :eof

::=============================================================================
:: ALL-FAST MODE - Launch all instances with 2-second delays
::=============================================================================
:MODE_ALL_FAST
echo Configuring memory management for large models...
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo Memory management configured.

:: Read configuration with PowerShell and store in environment variables
echo Loading configuration...

powershell -NoProfile -Command "try { $json = Get-Content '%CONFIG_PATH%' -Raw | ConvertFrom-Json; $master = $json.master; Write-Output \"MASTER_PORT=$($master.port)\"; Write-Output \"MASTER_CUDA=$($master.cuda_device)\"; Write-Output \"MASTER_EXTRA=$($master.extra_args)\"; foreach ($worker in $json.workers) { if ($worker.enabled -and -not $worker.host) { Write-Output \"WORKER_$($worker.id)_NAME=$($worker.name)\"; Write-Output \"WORKER_$($worker.id)_PORT=$($worker.port)\"; Write-Output \"WORKER_$($worker.id)_CUDA=$($worker.cuda_device)\"; Write-Output \"WORKER_$($worker.id)_EXTRA=$($worker.extra_args)\" } } } catch { Write-Output \"PowerShell Error: $_\"; pause; exit 1 }" > %TEMP%\config_vars.txt

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

echo.
echo Checking running instances...
echo.

set "INSTANCES_STARTED=0"

:: Check and start Master if not running
echo Checking Master (port %MASTER_PORT%)...
powershell -NoProfile -Command "$ErrorActionPreference = 'SilentlyContinue'; $tcpClient = New-Object System.Net.Sockets.TcpClient; $connect = $tcpClient.BeginConnect('127.0.0.1', %MASTER_PORT%, $null, $null); $wait = $connect.AsyncWaitHandle.WaitOne(200, $false); if (!$wait) { exit 1 } else { $tcpClient.EndConnect($connect); $tcpClient.Close(); exit 0 }"
if %ERRORLEVEL% == 1 (
    echo Master NOT running - Starting with CUDA_VISIBLE_DEVICES=%MASTER_CUDA%
    start wt -w 0 new-tab --title "Master - ComfyUI CLI" --startingDirectory "%CD%" cmd /k "title Master - ComfyUI CLI && echo CUDA_VISIBLE_DEVICES=%MASTER_CUDA% && call venv\Scripts\activate.bat && set CUDA_VISIBLE_DEVICES=%MASTER_CUDA% && comfy --here launch -- --port %MASTER_PORT% %MASTER_EXTRA%"
    set /a INSTANCES_STARTED+=1
    echo Waiting 2 seconds before checking next instance...
    timeout /t 2 /nobreak >nul
) else (
    echo Master already running on port %MASTER_PORT%
)

:: Check and start workers
powershell -NoProfile -Command "try { $json = Get-Content '%CONFIG_PATH%' -Raw | ConvertFrom-Json; foreach ($worker in $json.workers) { if ($worker.enabled -and -not $worker.host) { Write-Output $worker.id } } } catch { exit 1 }" > %TEMP%\worker_ids.txt

for /f %%w in (%TEMP%\worker_ids.txt) do (
    call :CHECK_AND_START_WORKER_FAST %%w
)

del %TEMP%\worker_ids.txt >nul 2>&1

echo.
echo ======================================
if %INSTANCES_STARTED% == 0 (
    echo All instances were already running!
) else (
    echo Started %INSTANCES_STARTED% new instance^(s^)!
)
echo ======================================
goto :FINISH_ALL

:CHECK_AND_START_WORKER_FAST
set "WORKER_ID=%1"
call set "WORKER_NAME=%%WORKER_%WORKER_ID%_NAME%%"
call set "WORKER_PORT=%%WORKER_%WORKER_ID%_PORT%%"
call set "WORKER_CUDA=%%WORKER_%WORKER_ID%_CUDA%%"
call set "WORKER_EXTRA=%%WORKER_%WORKER_ID%_EXTRA%%"

echo Checking %WORKER_NAME% (port %WORKER_PORT%)...
powershell -NoProfile -Command "$ErrorActionPreference = 'SilentlyContinue'; $tcpClient = New-Object System.Net.Sockets.TcpClient; $connect = $tcpClient.BeginConnect('127.0.0.1', %WORKER_PORT%, $null, $null); $wait = $connect.AsyncWaitHandle.WaitOne(200, $false); if (!$wait) { exit 1 } else { $tcpClient.EndConnect($connect); $tcpClient.Close(); exit 0 }"
if %ERRORLEVEL% == 1 (
    echo %WORKER_NAME% NOT running - Starting with CUDA_VISIBLE_DEVICES=%WORKER_CUDA%
    start wt -w 0 new-tab --title "%WORKER_NAME% - ComfyUI CLI" --startingDirectory "%CD%" cmd /k "title %WORKER_NAME% - ComfyUI CLI && echo CUDA_VISIBLE_DEVICES=%WORKER_CUDA% && call venv\Scripts\activate.bat && set CUDA_VISIBLE_DEVICES=%WORKER_CUDA% && comfy --here launch -- --port %WORKER_PORT% %WORKER_EXTRA% --enable-cors-header"
    set /a INSTANCES_STARTED+=1
    echo Waiting 2 seconds before checking next instance...
    timeout /t 2 /nobreak >nul
) else (
    echo %WORKER_NAME% already running on port %WORKER_PORT%
)
goto :eof

::=============================================================================
:: FINISH - Common ending for all modes
::=============================================================================
:FINISH_ALL
echo.
echo Master: http://localhost:%MASTER_PORT%
powershell -NoProfile -Command "try { $json = Get-Content '%CONFIG_PATH%' -Raw | ConvertFrom-Json; foreach ($worker in $json.workers) { if ($worker.enabled -and -not $worker.host) { Write-Output \"$($worker.name): http://localhost:$($worker.port)\" } } } catch { }"
echo.
echo Launcher will close in 5 seconds...
timeout /t 5 /nobreak >nul
exit

::=============================================================================
:: ACTIVATE VIRTUAL ENVIRONMENT - Open command prompt with venv activated
::=============================================================================
:ACTIVATE_VENV
echo.
echo  ================================================================
echo                 Activating Virtual Environment
echo  ================================================================
echo.

:: Check if venv exists
set "VENV_PATH=venv"
if not exist "%VENV_PATH%\Scripts\activate.bat" (
    color 0C
    echo Error: Virtual environment not found!
    echo Expected location: %CD%\%VENV_PATH%
    pause
    exit /b 1
)

echo Activating virtual environment...
echo Project folder: %CD%
echo.
echo Opening new command prompt with activated environment...
echo (You can close this window)
echo.

:: Open new command prompt with venv activated
start "ComfyUI (Active venv)" cmd /k "call "%CD%\%VENV_PATH%\Scripts\activate.bat" & cd /d "%CD%""
exit

::=============================================================================
:: UPDATE COMFYUI - Update ComfyUI to latest version
::=============================================================================
:UPDATE_COMFYUI
echo.
echo  ================================================================
echo                       Updating ComfyUI
echo  ================================================================
echo.

:: Check if venv exists
set "VENV_PATH=venv"
if not exist "%VENV_PATH%\Scripts\activate.bat" (
    color 0C
    echo Error: Virtual environment not found!
    pause
    exit /b 1
)

:: Check if ComfyUI CLI is available
call "%VENV_PATH%\Scripts\activate.bat" && comfy --help >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    color 0C
    echo Error: ComfyUI CLI not found in virtual environment!
    pause
    exit /b 1
)

echo Starting ComfyUI update...
echo This may take a few minutes...
echo.

:: Activate venv and run update
call "%VENV_PATH%\Scripts\activate.bat" && comfy --here update

echo.
echo Update completed!
echo.
pause
exit

::=============================================================================
:: UPDATE CUSTOM NODES - Update all custom nodes
::=============================================================================
:UPDATE_NODES
echo.
echo  ================================================================
echo                     Updating Custom Nodes
echo  ================================================================
echo.

:: Check if venv exists
set "VENV_PATH=venv"
if not exist "%VENV_PATH%\Scripts\activate.bat" (
    color 0C
    echo Error: Virtual environment not found!
    pause
    exit /b 1
)

:: Check if ComfyUI CLI is available
call "%VENV_PATH%\Scripts\activate.bat" && comfy --help >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    color 0C
    echo Error: ComfyUI CLI not found in virtual environment!
    pause
    exit /b 1
)

echo Starting custom nodes update...
echo This may take several minutes...
echo.

:: Activate venv and run node update
call "%VENV_PATH%\Scripts\activate.bat" && comfy --here node update-all

echo.
echo Custom nodes update completed!
echo.
pause
exit
