param(
    [string]$Config = "configs/pipeline.json",
    [switch]$Pretrain
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    $stage = if ($Pretrain) { "pretrain" } else { "train" }
    python -m railvivid $stage --config $Config
    if ($LASTEXITCODE -ne 0) { throw "Rail-VIVID $stage failed with exit code $LASTEXITCODE. Inspect the reported diagnostic and preserved artifacts." }
} finally {
    Pop-Location
}
