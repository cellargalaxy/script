# ================= 获取当前脚本所在目录 =================
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# ================= 激活 conda 环境 =================
& conda activate "$ProjectDir\.conda"

# ================= 定义输入文件数组（每行一个） =================
$InputFiles = @(
"D:/ripx.wav"
)

# ================= 定义模型文件数组（每行一个） =================
$ModelFiles = @(
"D:/model_50.pt"
"D:/model_100.pt"
)

# ================= 创建输出文件夹 =================
$OutputDir = Join-Path $ProjectDir "tmp\main_reflow"
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

# ================= 循环处理 =================
foreach ($i in $InputFiles) {
    foreach ($m in $ModelFiles) {

        # 获取文件名（不带路径和扩展名）
        $InputName = [System.IO.Path]::GetFileNameWithoutExtension($i)
        $ModelName = [System.IO.Path]::GetFileNameWithoutExtension($m)

        # 拼接输出文件路径
        $OutputFile = Join-Path $OutputDir "$InputName`_$ModelName.wav"

        # 执行主脚本
        Write-Host "Running: python -m ddspsvc_6_3.main_reflow -i $i -m $m -o $OutputFile"
        # method euler/rk4
        python -m ddspsvc_6_3.main_reflow -i $i -m $m -o $OutputFile -k 0 -id 1 -step 50 -method euler -ts 0.0
    }
}

Pause
