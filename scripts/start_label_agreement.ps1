$ErrorActionPreference = 'Stop'
$repoPath = Split-Path -Parent $PSScriptRoot
$outputPath = Join-Path $repoPath 'data/labeling'
$gatePath = Join-Path $outputPath 'pilot-gate.json'
if (-not (Test-Path -LiteralPath $gatePath)) { throw 'Run the pilot first.' }
$gate = Get-Content -LiteralPath $gatePath -Raw | ConvertFrom-Json
if (-not $gate.passed) { throw 'Pilot stop criterion triggered; full run forbidden.' }
$pythonPath = (Get-Command python -ErrorAction Stop).Source
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fffffff'
$stdoutPath = Join-Path $outputPath "worker-$stamp.stdout.log"
$stderrPath = Join-Path $outputPath "worker-$stamp.stderr.log"
# Start-Process is independent of this terminal; the Python worker owns an OS file
# lock, so duplicate starts cannot duplicate model work or corrupt the JSONL log.
$worker = Start-Process -FilePath $pythonPath -ArgumentList @('-B', 'scripts/label_agreement.py', 'full') -WorkingDirectory $repoPath -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
$record = [ordered]@{pid=$worker.Id; launched_at=(Get-Date).ToUniversalTime().ToString('o'); stdout=$stdoutPath; stderr=$stderrPath; command='python -B scripts/label_agreement.py full'}
$record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $outputPath "launch-$stamp.json") -Encoding UTF8
$record | ConvertTo-Json
