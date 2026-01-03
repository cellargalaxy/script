@echo off
setlocal

REM ===== 将文件放在project文件夹下 =====
REM ===== project文件夹里有.conda文件夹 =====

REM ===== 获取当前 bat 所在目录（末尾自带反斜杠）=====
set PROJECT_DIR=%~dp0

REM ===== 去掉末尾反斜杠（可选，但更干净）=====
set PROJECT_DIR=%PROJECT_DIR:~0,-1%

REM ===== 将项目路径写入 workdir（不存在则创建，存在则覆盖）=====
echo %PROJECT_DIR% > "%PROJECT_DIR%\workdir"

REM ===== 激活项目内 conda 环境 =====
call conda activate "%PROJECT_DIR%\.conda"

REM ===== 切换到项目目录 =====
cd /d "%PROJECT_DIR%"

REM ===== 启动 WebUI =====
python launcher.py

pause