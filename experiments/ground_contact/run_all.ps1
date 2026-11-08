# UPC155 bounce paired training - desktop run
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

if (Test-Path -LiteralPath 'results\_smoke') { Remove-Item -Recurse -Force 'results\_smoke' }
foreach ($d in 'results\airborne', 'results\bounce') {
    if (Test-Path -LiteralPath $d) { Remove-Item -Recurse -Force $d }
}

$t0 = Get-Date
Write-Output "=== airborne arm ==="
& $py train_bounce_control_mdn.py . results/airborne --arm airborne --epochs 100 2>&1 | Tee-Object -FilePath results\log_airborne.txt
$t1 = Get-Date
Write-Output "airborne seconds: $([math]::Round(($t1-$t0).TotalSeconds,1))"

Write-Output "=== bounce arm ==="
& $py train_bounce_control_mdn.py . results/bounce --arm bounce --epochs 100 2>&1 | Tee-Object -FilePath results\log_bounce.txt
$t2 = Get-Date
Write-Output "bounce seconds: $([math]::Round(($t2-$t1).TotalSeconds,1))"

Import-Csv results/airborne/test_scores.csv, results/bounce/test_scores.csv |
    Export-Csv results/paired_test_scores.csv -NoTypeInformation -Encoding utf8
Write-Output "merged rows: $((Import-Csv results/paired_test_scores.csv).Count)"
Write-Output "TOTAL_SECONDS=$([math]::Round(((Get-Date)-$t0).TotalSeconds,1))"
