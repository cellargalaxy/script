@echo off
chcp 65001 > nul

rem 1. 执行 conda activate gen_subt_v7
call conda activate gen_subt_v7

rem 2. python调用当前文件夹下的main.py
if exist "main.py" (
    echo.
    echo ** 正在执行 Python 脚本 (main.py) ... **
    echo -----------------------------------------------------------------
    python main.py
    echo -----------------------------------------------------------------
) else (
    echo.
    echo ** 错误：未找到 main.py 文件！ **
    echo -----------------------------------------------------------------
)

rem 3. 无论是否正常退出，最后都要回车才退出命令行
echo.
echo ** 脚本执行完毕。请按任意键退出命令行窗口... **
pause > nul

rem 恢复默认的字符编码 (可选)
chcp 936 > nul