# 1. Get folder path
$FolderPath = Read-Host "Enter the folder path (e.g., D:\demo)"

# Check if path exists
if (-not (Test-Path -Path $FolderPath -PathType Container)) {
    Write-Error "Error: Folder '$FolderPath' does not exist or is not a valid folder."
    return
}

# 2. Get folder name
$FolderName = (Split-Path -Path $FolderPath -Leaf)

# 3. Select language type
$LanguageOptions = @("JA", "ZH", "EN")
[int]$Choice = -1

do {
    Write-Host "Please select the language type:" -ForegroundColor Yellow
    for ($i = 0; $i -lt $LanguageOptions.Count; $i++) {
        Write-Host "[$i] $($LanguageOptions[$i])"
    }

    # Removed the Chinese prompt that was causing the parsing error
    $InputChoice = Read-Host "Enter the corresponding number (0, 1, or 2)"

    # Try to convert input to integer and check if it's within the valid range
    if ([int]::TryParse($InputChoice, [ref]$Choice) -and ($Choice -ge 0 -and $Choice -lt $LanguageOptions.Count)) {
        $LanguageType = $LanguageOptions[$Choice]
    } else {
        Write-Host "Invalid choice, please re-enter." -ForegroundColor Red
        $Choice = -1 # Reset to continue loop
    }
} while ($Choice -eq -1)

Write-Host "You selected language type: $LanguageType" -ForegroundColor Green

# 4. Set output file path
$OutputFilePath = Join-Path -Path (Split-Path -Path $FolderPath -Parent) -ChildPath "$FolderName.list"

# 5. Iterate files and write data
Write-Host "Generating list file..."

# Initialize an empty array to store all lines
$OutputContent = @()

# Get-ChildItem iterates through files in the specified path
Get-ChildItem -Path $FolderPath -File | ForEach-Object {

    # Get the full file path
    $FilePath = $_.FullName

    # Get the file name (without extension)
    $FileNameWithoutExt = $_.BaseName

    # Format the output line: {FilePath}|{FolderName}|{LanguageType}|{FileNameWithoutExt}
    $Line = "$FilePath|$FolderName|$LanguageType|$FileNameWithoutExt"

    # Add the line to the array
    $OutputContent += $Line
}

# 6. Write all content to the new text file (Using UTF8 to ensure compatibility)
$OutputContent | Set-Content -Path $OutputFilePath -Encoding UTF8

Write-Host "---"
Write-Host "Script execution completed!" -ForegroundColor Cyan
Write-Host "Results written to file: $OutputFilePath" -ForegroundColor Cyan
Write-Host "File content sample:" -ForegroundColor Green
# Display the first 2 lines of content
$OutputContent | Select-Object -First 2