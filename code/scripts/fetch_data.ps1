<#
Download the five simulated launch banks from the data-v1.0 release and place them
where the code expects them, verifying SHA-256 for every file.

    powershell -ExecutionPolicy Bypass -File code\scripts\fetch_data.ps1
    powershell -ExecutionPolicy Bypass -File code\scripts\fetch_data.ps1 -Force     # re-download

Files already present with the right hash are skipped. Size on disk is about 465 MB.
If GitHub is unreachable directly, set $env:HTTPS_PROXY first, e.g.
    $env:HTTPS_PROXY = 'http://127.0.0.1:10809'
#>
param(
    [switch]$Force,
    [string]$Repo = 'x1shang/MDN-soccer-prediction',
    [string]$Tag = 'data-v1.0'
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path

$assets = @(
    @{ Name = 'v3_all_1.225M_data.npz';    Dest = 'data\datasets\v3_all\data.npz';
       Sha = '8774a9fa746bcd1e326cff606fbcdf62fad8517f8038d1ef48299c7c44a9eba7' },
    @{ Name = 'v2_no_asai_1.225M_data.npz'; Dest = 'data\datasets\v2_no_asai\data.npz';
       Sha = 'e9b2662f85010383538d2fe2b0148e898b6dbf433bdb315d6f2d107310b15a16' },
    @{ Name = 'v3_no_asai_1.225M_data.npz'; Dest = 'data\datasets\v3_no_asai\data.npz';
       Sha = '93033ad5276b2b1f433b8ad03be00978481a0a261027acc093981bfa23c734c7' },
    @{ Name = 'v3_all_100k_data.npz';      Dest = 'experiments\ground_contact\data.npz';
       Sha = 'e16935d3d8cb004042e11cfd0669ba7b5ac853f4df8dc6936a98c008315b0b9f' },
    @{ Name = 'bounce_100k_data.npz';      Dest = 'experiments\ground_contact\bounce\data_bounce.npz';
       Sha = '46cf86a83d6749079f343a2b52440dc8cea7de0ee3c317f0ed7e6384c0fd5e10' }
)

function Test-Hash([string]$path, [string]$sha) {
    if (-not (Test-Path -LiteralPath $path)) { return $false }
    return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower() -eq $sha
}

$gh = Get-Command gh -ErrorAction SilentlyContinue
$tmp = Join-Path $env:TEMP ("mdn-data-" + [guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

try {
    foreach ($a in $assets) {
        $dest = Join-Path $root $a.Dest
        if (-not $Force -and (Test-Hash $dest $a.Sha)) {
            Write-Output "ok    $($a.Dest)"
            continue
        }
        $parent = Split-Path -Parent $dest
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
        $staged = Join-Path $tmp $a.Name

        if (-not (Test-Path -LiteralPath $staged)) {
            if ($gh) {
                Write-Output "fetch $($a.Name) via gh"
                & gh release download $Tag --repo $Repo --pattern $a.Name --dir $tmp --clobber
                if ($LASTEXITCODE -ne 0) { throw "gh release download failed for $($a.Name)" }
            } else {
                $url = "https://github.com/$Repo/releases/download/$Tag/$($a.Name)"
                Write-Output "fetch $($a.Name) via https"
                Invoke-WebRequest -Uri $url -OutFile $staged -UseBasicParsing
            }
        }

        if (-not (Test-Hash $staged $a.Sha)) {
            throw "SHA-256 mismatch for $($a.Name). Delete it and retry; the release asset may be corrupt."
        }
        Move-Item -LiteralPath $staged -Destination $dest -Force
        Write-Output "done  $($a.Dest)"
    }
} finally {
    if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue }
}

Write-Output 'All five datasets are in place.'
