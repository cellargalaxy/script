# Force PowerShell output to UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Force Python to output UTF-8
$env:PYTHONIOENCODING = "UTF-8"
$env:PYTHONUTF8 = "1"

Write-Host "`n[INFO] Console encoding: $([Console]::OutputEncoding.WebName)"

# ---- Conda initialization ----
$condaHook = "D:\softdata\miniconda3\shell\condabin\conda-hook.ps1"

if (Test-Path $condaHook) {
    Write-Host "[INFO] Loading conda hook..."
    & $condaHook
} else {
    Write-Host "[ERROR] Cannot find conda hook file: $condaHook"
}

# ---- Activate environment ----
conda activate gen_subt_v7

# ---- Run Python script ----
try {
    python ".\main.py"
}
catch {
    Write-Host "`n[ERROR] Exception occurred while running main.py:"
    Write-Host $_
}
finally {
    Write-Host "`n[INFO] Execution finished. Press Enter to exit..."
    Read-Host
}
