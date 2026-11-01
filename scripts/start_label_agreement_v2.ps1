param(
    [ValidateSet('tune','full')][string]$Phase = 'full',
    [ValidateSet('initial','revision')][string]$Version = 'initial'
)
$ErrorActionPreference = 'Stop'
$repoPath = Split-Path -Parent $PSScriptRoot
$outputPath = Join-Path $repoPath 'data/labeling/v2'
if ($Phase -eq 'full' -and -not (Test-Path -LiteralPath (Join-Path $outputPath 'choice.json'))) {
    throw 'Choose and freeze a context after tuning first.'
}
if ($Phase -eq 'full') {
    $choice = Get-Content -LiteralPath (Join-Path $outputPath 'choice.json') -Raw | ConvertFrom-Json
    $Version = $choice.version
}
$pythonPath = (Get-Command python -ErrorAction Stop).Source
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fffffff'
$stdoutPath = Join-Path $outputPath "worker-$Phase-$Version-$stamp.stdout.log"
$stderrPath = Join-Path $outputPath "worker-$Phase-$Version-$stamp.stderr.log"
$worker = Start-Process -FilePath $pythonPath -ArgumentList @('-B', 'scripts/label_agreement_v2.py', $Phase, '--version', $Version) -WorkingDirectory $repoPath -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
$record = [ordered]@{pid=$worker.Id; phase=$Phase; version=$Version; launched_at=(Get-Date).ToUniversalTime().ToString('o'); stdout=$stdoutPath; stderr=$stderrPath; command="python -B scripts/label_agreement_v2.py $Phase --version $Version"}
$record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $outputPath "launch-$stamp.json") -Encoding UTF8
$record | ConvertTo-Json
