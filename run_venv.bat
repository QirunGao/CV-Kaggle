@echo off
chcp 65001 >nul

REM 切换到脚本所在目录
cd /d "%~dp0"

REM 检查 venv 是否存在
IF NOT EXIST "venv\Scripts\activate.bat" (
    echo.
    echo venv 不存在，正在创建...
    python -m venv venv
    echo venv 创建完成
)

REM 弹出新窗口并激活虚拟环境
echo 正在启动新的虚拟环境终端窗口...
start "" cmd /k "cd /d \"%~dp0\" & call venv\Scripts\activate.bat & echo 已激活 venv"

echo.
echo 已启动新窗口，%~n0 将在 5 秒后自动退出...
timeout /t 5 >nul
exit /b
