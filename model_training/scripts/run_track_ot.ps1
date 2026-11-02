param(
    [ValidateSet('Doctor', 'Pilot', 'Plan', 'CampaignTrain', 'CampaignEvaluate', 'Summarize')]
    [string]$Stage = 'Doctor',
    [string]$Python = 'python',
    [string]$Config = 'configs/track_ot_plus_plus.json',
    [string]$PlanDirectory = 'artifacts/track_ot_plan',
    [switch]$OnTrainingDevice
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    if ($Stage -in @('Pilot', 'CampaignTrain') -and -not $OnTrainingDevice) {
        throw 'Fitting requires -OnTrainingDevice on the RTX training computer.'
    }
    switch ($Stage) {
        'Doctor' { & $Python -m track_ot doctor --config $Config }
        'Pilot' { & $Python -m track_ot train --config $Config --on-training-device }
        'Plan' { & $Python -m track_ot plan --config $Config --output $PlanDirectory }
        'CampaignTrain' { & $Python -m track_ot campaign-run --plan (Join-Path $PlanDirectory 'plan.json') --on-training-device }
        'CampaignEvaluate' { & $Python -m track_ot campaign-evaluate --plan (Join-Path $PlanDirectory 'plan.json') }
        'Summarize' { & $Python -m track_ot summarize --plan (Join-Path $PlanDirectory 'plan.json') --output reports/track_ot_results }
    }
    if ($LASTEXITCODE -ne 0) { throw "TRACK-OT command failed with exit code $LASTEXITCODE" }
}
finally { Pop-Location }
