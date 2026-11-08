<#
Train the Wiener residual MDN on the three million-scale datasets.

Run from anywhere; paths are resolved from the repository root.

    powershell -ExecutionPolicy Bypass -File code\scripts\run_experiment.ps1 -Python python

The datasets must be on disk first (see data/README.md):
    data/datasets/<arm>/data.npz + indices.npz

Artifacts land in:
    experiments/main_v3_all/seed_<seed>/        (arm v3_all)
    experiments/asai_holdout/<arm>/seed_<seed>/ (arms v2_no_asai, v3_no_asai)
#>
param(
    [string]$Python = 'python',
    [int[]]$Seeds = @(11, 42, 155),
    [string[]]$Arms = @('v3_all', 'v2_no_asai', 'v3_no_asai')
)

$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$root = (Resolve-Path (Join-Path $here '..\..')).Path
$trainer = Join-Path $root 'code\model\train_wiener_mdn.py'
$datasets = Join-Path $root 'data\datasets'
$experiments = Join-Path $root 'experiments'
$workbooks = Join-Path $root 'data\anchors\asai'
$runtime = Join-Path $root 'env\runtime'

if (Test-Path -LiteralPath $runtime) { $env:PYTHONPATH = $runtime }
$env:OPENBLAS_NUM_THREADS = '1'
$env:PYTHONUNBUFFERED = '1'

foreach ($seed in $Seeds) {
    foreach ($arm in $Arms) {
        $dataset = Join-Path $datasets $arm
        if (-not (Test-Path -LiteralPath (Join-Path $dataset 'data.npz'))) {
            Write-Output "Skip $arm seed=$seed - missing $dataset\data.npz (see data/README.md)"
            continue
        }
        $parent = if ($arm -eq 'v3_all') { Join-Path $experiments 'main_v3_all' } else { Join-Path $experiments "asai_holdout\$arm" }
        $result = Join-Path $parent "seed_$seed"
        if (Test-Path (Join-Path $result 'config.json')) {
            Write-Output "Already complete $arm seed=$seed"
            continue
        }
        New-Item -Path $result -ItemType Directory -Force | Out-Null
        Push-Location $result
        try {
            $argsForArm = @('--device', 'cuda', '--epochs', '100', '--seed', "$seed",
                            '--data-dir', $dataset, '--batch-size', '512')
            if ($arm -ne 'v3_all') {
                $argsForArm += @('--asai-dir', $workbooks)
            }
            Write-Output "$(Get-Date -Format s) Starting $arm seed=$seed"
            & $Python $trainer @argsForArm *> 'training.log'
            if ($LASTEXITCODE -ne 0) { throw "Failed $arm seed=$seed. See $result/training.log" }
            Write-Output "$(Get-Date -Format s) Finished $arm seed=$seed"
        } finally {
            Pop-Location
        }
    }
}
