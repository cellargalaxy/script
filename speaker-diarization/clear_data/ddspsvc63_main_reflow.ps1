# ================= 需要在 ddspsvc_6_3.main_reflow.py 里添加下面这行代码 =================
# ================= import torch, fairseq; torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary]) =================


# ================= 获取当前脚本所在目录 =================
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# ================= 输出目录 =================
$OutputDir = Join-Path $ProjectDir "tmp\main_reflow"

# ================= 如果输出目录存在，删除里面的所有文件，否则创建目录 =================
if (Test-Path $OutputDir) {
    Write-Host "Cleaning existing output folder: $OutputDir"
    Remove-Item "$OutputDir\*" -Recurse -Force
} else {
    Write-Host "Creating output folder: $OutputDir"
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

# ================= 激活 conda 环境 =================
& conda activate "$ProjectDir\.conda"

# ================= 定义输入文件数组（每行一个） =================
$InputFiles = @(
"D:/东海/测试.wav"
)

# ================= 定义模型文件数组（每行一个） =================
$ModelFiles = @(
"D:/东海/model_400.pt"
)

# ================= 循环处理 =================
foreach ($i in $InputFiles) {
    foreach ($m in $ModelFiles) {

        # ===== 获取文件名（不带扩展名）=====
        $InputName = [System.IO.Path]::GetFileNameWithoutExtension($i)
        $ModelName = [System.IO.Path]::GetFileNameWithoutExtension($m)

        # ===== 模型所在文件夹名 =====
        # D:/模型/女声模型/model_50.pt -> 女声模型
        $ModelFolderName = Split-Path (Split-Path $m -Parent) -Leaf

        # ===== 输出目录：tmp/main_reflow/女声模型 =====
        $ModelOutputDir = Join-Path $OutputDir $ModelFolderName

        if (-not (Test-Path $ModelOutputDir)) {
            New-Item -ItemType Directory -Path $ModelOutputDir | Out-Null
        }

        # ===== 输出文件名（支持中文）=====
        $OutputFile = Join-Path $ModelOutputDir "$InputName`_$ModelName.wav"

        Write-Host "--------------------------------------------------"
        Write-Host "Input : $i"
        Write-Host "Model : $m"
        Write-Host "Output: $OutputFile"
        Write-Host "--------------------------------------------------"

        # method euler/rk4
        python -m ddspsvc_6_3.main_reflow `
            -i "$i" `
            -m "$m" `
            -o "$OutputFile" `
            -k 0 `
            -id 1 `
            -step 50 `
            -method euler `
            -ts 0.0
    }
}

Pause
