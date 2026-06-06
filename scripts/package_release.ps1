param(
    [switch]$IncludeStorage
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$ReleaseRoot = Join-Path $Root "release"
$Stage = Join-Path $ReleaseRoot "MetalWearPlatform_v1_$Stamp"
$ZipPath = "$Stage.zip"

New-Item -ItemType Directory -Force -Path $ReleaseRoot | Out-Null
if (Test-Path $Stage) {
    Remove-Item -LiteralPath $Stage -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $Stage | Out-Null

$Dirs = @(
    "app",
    "web",
    "docs",
    "integrations",
    "manual_assets",
    "scripts",
    "checkpoints\baseline"
)

foreach ($Dir in $Dirs) {
    $Source = Join-Path $Root $Dir
    if (Test-Path $Source) {
        $Target = Join-Path $Stage $Dir
        New-Item -ItemType Directory -Force -Path (Split-Path $Target -Parent) | Out-Null
        Copy-Item -LiteralPath $Source -Destination $Target -Recurse -Force
    }
}

$Files = @(
    "README.md",
    "RELEASE_CN.md",
    "requirements.txt",
    "run.ps1",
    "checkpoints\registry.json"
)

foreach ($File in $Files) {
    $Source = Join-Path $Root $File
    if (Test-Path $Source) {
        $Target = Join-Path $Stage $File
        New-Item -ItemType Directory -Force -Path (Split-Path $Target -Parent) | Out-Null
        Copy-Item -LiteralPath $Source -Destination $Target -Force
    }
}

if ($IncludeStorage) {
    Copy-Item -LiteralPath (Join-Path $Root "storage") -Destination (Join-Path $Stage "storage") -Recurse -Force
} else {
    $StorageDirs = @(
        "storage\datasets",
        "storage\categories",
        "storage\exports",
        "storage\model_tests",
        "storage\dataset_archives",
        "storage\training_jobs",
        "storage\audit_logs",
        "checkpoints\runs"
    )
    foreach ($Dir in $StorageDirs) {
        New-Item -ItemType Directory -Force -Path (Join-Path $Stage $Dir) | Out-Null
    }
}

$CleanupDirs = @("__pycache__", ".pytest_cache")
foreach ($Name in $CleanupDirs) {
    Get-ChildItem -LiteralPath $Stage -Directory -Recurse -Force -Filter $Name |
        Remove-Item -Recurse -Force
}

$CleanupFiles = @("*.pyc", "*.pyo", "*.log")
foreach ($Pattern in $CleanupFiles) {
    Get-ChildItem -LiteralPath $Stage -File -Recurse -Force -Filter $Pattern |
        Remove-Item -Force
}

$ScratchModel = Join-Path $Stage "checkpoints\baseline\unet32_scratch.pt"
Get-ChildItem -LiteralPath (Join-Path $Stage "checkpoints") -File -Recurse -Force -Filter "*.pt" |
    Where-Object { $_.FullName -ne $ScratchModel } |
    Remove-Item -Force

if (Test-Path $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
}
Compress-Archive -Path (Join-Path $Stage "*") -DestinationPath $ZipPath -Force
Write-Host "Release package created:"
Write-Host $ZipPath
