# run_script.ps1

# 步骤 1: 激活 Conda 环境
# 注意：Conda activate 通常是一个 shell 函数/脚本，直接在 .ps1 中调用可能失败。
# 最可靠的方法是使用 'conda run'，或者在一个子shell中执行activate。
# 我们使用 'conda run' 来执行后续的 python 命令。

# 步骤 2: 调用当前文件夹下的 main.py
# 步骤 3 & 4: 兼容中文日志和最后回车退出

# 使用 cmd /k 来执行一系列命令，确保最后等待用户按键，并且保留conda环境的激活状态。
# 注意：使用 -Command 而不是 -File，并用分号分隔命令。
# 'conda run' 是推荐的做法，它会在一个子shell中激活环境并执行命令。
$command = "conda activate gen_subt_v7 ; python .\main.py"

# 使用 cmd.exe /k 执行命令
# /C 是执行后关闭，/K 是执行后保留窗口
Write-Host "正在执行 Conda 环境激活和 Python 脚本..."

# 为了兼容中文输出，我们直接在 PowerShell 窗口中执行 Conda 激活和 Python
# 并使用 $LASTEXITCODE 来检查 Python 脚本的退出状态
try {
    # 尝试直接执行 conda activate 和 python
    # Conda环境的激活只在其执行的当前会话中有效。
    # 为了保证环境激活和脚本执行在同一个会话中，我们使用 call & 命令。
    
    # 尝试在当前 PowerShell 进程中执行命令（这对于 activate 可能不稳定，但兼容性最好）
    & conda activate gen_subt_v7
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Conda 环境激活失败！请确保 Conda 已正确添加到系统 PATH 中。"
    }
    
    Write-Host "--- Python 脚本开始执行 ---"
    python .\main.py
    $pythonExitCode = $LASTEXITCODE
    Write-Host "--- Python 脚本执行结束，退出码: $pythonExitCode ---"

} catch {
    Write-Error "脚本执行中发生错误: $($_.Exception.Message)"
} finally {
    # 无论前面是否正常退出，最后都等待用户按回车键
    Write-Host "按任意键退出命令行窗口..."
    
    # 暂停命令行，等待用户按键。在 PowerShell 中，这是通过 Read-Host 实现的。
    # 也可以使用 cmd /c "pause" 但为了保持原生性，我们用 Read-Host
    [void]$Host.UI