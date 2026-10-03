<#
.SYNOPSIS
  Follow-up experiments after the MedMCQA train/eval ICL pilot (Windows A5000 server).

.DESCRIPTION
  Step "f0r1" (~20 min): evaluate F0 at FL round 1 on the pilot's test split, the
                         equal-budget partner of the existing C0 @ epoch 1 result.
  Step "main" (~4 days): three-seed pilot-size run of the baselines needed to judge
                         the method: B0 B1 L0 L1 F0 F1 FT0 FT1. Resumable: re-run the
                         same command after any interruption.
  Step "central"        : optional, adds C0 C1 CT0 CT1 to the three-seed run
                          (~+6 days; C0 and CT each train R epochs, ~22-27 h/seed).

  Run from the repo root:
    powershell -ExecutionPolicy Bypass -File scripts\run_followups.ps1 -Step f0r1
    powershell -ExecutionPolicy Bypass -File scripts\run_followups.ps1 -Step main -Gpu 1
  Watch progress:
    Get-Content outputs\a5000-medmcqa-train-icl-3seed\pipeline_state.yaml -Wait
#>
param(
    [ValidateSet("f0r1", "main", "central", "all")]
    [string]$Step = "all",
    [ValidateSet(0, 1)]
    [int]$Gpu = 1
)

$ErrorActionPreference = "Stop"
$PilotConfig = "configs\a5000-medmcqa-train-icl-pilot.yaml"
$MainConfig = "configs\a5000-medmcqa-train-icl-3seed.yaml"
$MainArms = @("B0", "B1", "L0", "L1", "F0", "F1", "FT0", "FT1")
$CentralArms = @("C0", "C1", "CT0", "CT1")

New-Item -ItemType Directory -Force -Path "outputs\logs" | Out-Null
$Log = "outputs\logs\followups-$(Get-Date -Format yyyyMMdd-HHmmss).log"
Start-Transcript -Path $Log -Append | Out-Null

function Invoke-Fedicl {
    param([string[]]$Arguments)
    Write-Host "`n>>> fedicl-mqa $($Arguments -join ' ')  [$(Get-Date -Format s)]" -ForegroundColor Cyan
    uv run --no-sync fedicl-mqa @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "fedicl-mqa $($Arguments[0]) exited with code $LASTEXITCODE"
    }
}

try {
    if ($Step -in @("f0r1", "all")) {
        # FL round 1 = one cumulative local epoch, so --epoch 1 selects checkpoint-round-0001.
        Invoke-Fedicl @("evaluate", "--config", $PilotConfig, "--arm", "F0",
            "--seed", "42", "--split", "test", "--epoch", "1", "--gpu", "$Gpu")
    }

    if ($Step -in @("main", "all")) {
        Invoke-Fedicl @("doctor", "--config", $MainConfig, "--gpu", "$Gpu")
        Invoke-Fedicl (@("pipeline", "--config", $MainConfig, "--gpu", "$Gpu", "--quiet", "--arms") + $MainArms)
    }

    if ($Step -eq "central") {
        # Same output root: finished F/FT/L/B work is detected and skipped.
        Invoke-Fedicl (@("pipeline", "--config", $MainConfig, "--gpu", "$Gpu", "--quiet", "--arms") + $MainArms + $CentralArms)
    }

    Write-Host "`nDone. Results:" -ForegroundColor Green
    if ($Step -in @("f0r1", "all")) {
        Write-Host "  outputs\a5000-medmcqa-train-icl-pilot\arms\medmcqa\F0\seed-42\test\epoch-1\summary.json"
    }
    if ($Step -ne "f0r1") {
        Write-Host "  outputs\a5000-medmcqa-train-icl-3seed\arms_comparison.md"
        Write-Host "  outputs\a5000-medmcqa-train-icl-3seed\reports\medmcqa\contrasts-*.json"
    }
}
finally {
    Stop-Transcript | Out-Null
    Write-Host "Log: $Log"
}
