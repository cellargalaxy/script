# Define variables
$EnvName = "gen_subt_v7"
$ScriptPath = "./main.py"
$MaxAttempts = 10
$Attempt = 1
$Success = $false

Write-Host "--- Starting Process ---"
Write-Host "Target Environment: $EnvName"

while ($Attempt -le $MaxAttempts) {
    Write-Host "Attempt $Attempt of $MaxAttempts..."

    # Execute the python script using conda run
    conda run -n $EnvName --no-capture-output python $ScriptPath

    # Check the exit code of the last command
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Success: Script finished successfully."
        $Success = $true
        break
    } else {
        Write-Host "Warning: Script exited with error code $LASTEXITCODE."
        if ($Attempt -lt $MaxAttempts) {
            Write-Host "Retrying in 2 seconds..."
            Start-Sleep -Seconds 2
        }
    }

    $Attempt++
}

if (-not $Success) {
    Write-Host "Error: Maximum retry attempts reached. Process failed." -ForegroundColor Red
}

Write-Host "--- Process Completed ---"
Write-Host "Press Enter to exit..."
Read-Host