$ErrorActionPreference = 'Stop'
$repoPath = Split-Path -Parent $PSScriptRoot
$outputPath = Join-Path $repoPath 'data/labeling/v3'
if ((Get-TimeZone).Id -ne 'Eastern Standard Time') { throw 'Daily cap requires America/New_York local timezone.' }
$pythonPath = (Get-Command python -ErrorAction Stop).Source
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fffffff'
$worker = Start-Process -FilePath $pythonPath -ArgumentList @('-B', 'scripts/label_agreement_v3.py', 'run') -WorkingDirectory $repoPath -WindowStyle Hidden -RedirectStandardOutput (Join-Path $outputPath "worker-$stamp.stdout.log") -RedirectStandardError (Join-Path $outputPath "worker-$stamp.stderr.log") -PassThru
$record = [ordered]@{pid=$worker.Id; launched_at=(Get-Date).ToUniversalTime().ToString('o'); command='python -B scripts/label_agreement_v3.py run'}
$record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $outputPath "launch-$stamp.json") -Encoding UTF8
$record | ConvertTo-Json
