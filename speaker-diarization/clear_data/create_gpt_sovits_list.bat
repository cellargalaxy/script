@echo off
rem 设置文件编码为 UTF-8
chcp 65001 > nul

set "FOLDER_PATH="
set "LANGUAGE_TYPE="
set "FOLDER_NAME="
set "OUTPUT_FILE="
set "TEMP_FILE_LIST=%TEMP%\temp_list_%random%.tmp"

:INPUT_FOLDER
echo.
echo 请输入要处理的文件夹路径 (例如: d:\demo):
set /p "FOLDER_PATH="

rem 检查路径是否为空
if "%FOLDER_PATH%"=="" (
    echo.
    echo 错误：文件夹路径不能为空。请重新输入。
    goto INPUT_FOLDER
)

rem 检查路径是否存在
if not exist "%FOLDER_PATH%" (
    echo.
    echo 错误：指定的文件夹 "%FOLDER_PATH%" 不存在。请重新输入。
    goto INPUT_FOLDER
)

rem 移除路径末尾的反斜杠
if "%FOLDER_PATH:~-1%"=="\" set "FOLDER_PATH=%FOLDER_PATH:~0,-1%"

:SELECT_LANGUAGE
echo.
echo 请选择语言类型 (输入 JA 或 ZH):
set /p "LANGUAGE_TYPE="

rem 转换为大写并检查输入是否有效
set "LANGUAGE_TYPE=%LANGUAGE_TYPE:ja=JA%"
set "LANGUAGE_TYPE=%LANGUAGE_TYPE:zh=ZH%"

if /i "%LANGUAGE_TYPE%"=="JA" goto PROCESS
if /i "%LANGUAGE_TYPE%"=="ZH" goto PROCESS

echo.
echo 错误：语言类型输入无效。必须是 JA 或 ZH。
goto SELECT_LANGUAGE

:PROCESS
rem 提取文件夹名称
for %%F in ("%FOLDER_PATH%") do set "FOLDER_NAME=%%~nF"

rem 设置输出文件路径
set "OUTPUT_FILE=%FOLDER_PATH%.list"

echo.
echo 正在处理文件夹: %FOLDER_PATH%
echo 语言类型: %LANGUAGE_TYPE%
echo 输出文件: %OUTPUT_FILE%
echo.

rem *** 核心修改部分：使用临时文件累积内容，避免换行符错误 ***

rem 1. 清空临时文件
copy nul "%TEMP_FILE_LIST%" > nul

rem 开启延迟扩展
setlocal enabledelayedexpansion

rem 2. 遍历文件并将内容追加写入到临时文件
for %%F in ("%FOLDER_PATH%\*") do (
    rem 检查是否为文件
    echo %%~aF | find /i "d" > nul
    if errorlevel 1 (
        rem 提取绝对路径和文件名称（不含扩展名）
        set "FULL_PATH=%%~fF"
        set "FILE_NAME=%%~nF"

        rem 写入到临时文件，管道符 "|" 需要转义 "^|"
        echo !FULL_PATH!^|!FOLDER_NAME!^|!LANGUAGE_TYPE!^|!FILE_NAME! >> "%TEMP_FILE_LIST%"
    )
)

endlocal
rem 3. 将临时文件内容写入最终文件，并删除首尾空行
rem 这里的 /v /c "findstr /R . <..." 是一种安全的写入方法
cmd /v /c "findstr /R . < "!TEMP_FILE_LIST!"" > "%OUTPUT_FILE%"

rem 4. 清理临时文件
del "%TEMP_FILE_LIST%" 2>nul


echo.
echo ====================================
echo 文件列表已成功生成到: %OUTPUT_FILE%
echo ====================================
echo.
pause
exit