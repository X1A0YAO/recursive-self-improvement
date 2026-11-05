# DS-120 V0 baseline + stage-B queue runner (ASCII only for GBK safety)
# Runs after matrix_screen finishes: DS-120 baseline eval, then stage-B confirm cells.

$ErrorActionPreference = "Stop"
Set-Location D:\Proj\weight-space-rsi
$env:PYTHONIOENCODING = "utf-8"

# Wait for matrix_screen to finish (poll every 60s, max 6h)
Write-Output "[queue] waiting for matrix_screen to finish..."
$deadline = (Get-Date).AddHours(6)
while ((Get-Date) -lt $deadline) {
    $proc = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*matrix_screen*" -and $_.ProcessId -ne $PID }
    if (-not $proc) { break }
    Start-Sleep -Seconds 60
}
Write-Output "[queue] matrix_screen done (or timeout). Starting DS-120 V0 baseline..."

# DS-120 V0 baseline (needed for any stage-B McNemar)
& .venv\Scripts\python.exe -X utf8 scripts\evaluate.py --model-dir checkpoints\V0 --splits ds120 --output results\ds120_V0.json --max-new-tokens 768 --batch-size 4
if ($LASTEXITCODE -ne 0) { Write-Output "[queue] DS-120 baseline FAILED"; exit 1 }
Write-Output "[queue] DS-120 V0 baseline done -> results\ds120_V0.json"

# Read screen results, select eligible cells (logprob <= -3.0 OR quick20_worse >= 3)
& .venv\Scripts\python.exe -X utf8 -c "
import json
screen = json.loads(open('results/matrix_screen_full.json', encoding='utf-8').read())
elig = [cid for cid, c in screen['cells'].items() if c['logprob_delta'] <= -3.0 or c['quick20_worse'] >= 3]
print('ELIGIBLE:' + ','.join(elig))
" | Tee-Object -Variable eligLine | Write-Output
$cells = ($eligLine | Select-String "ELIGIBLE:(.+)").Matches[0].Groups[1].Value
Write-Output "[queue] eligible cells: $cells"

if ($cells) {
    & .venv\Scripts\python.exe -X utf8 scripts\matrix_confirm.py --cells $cells
    Write-Output "[queue] stage-B confirm done"
} else {
    Write-Output "[queue] NO eligible cells - preregistration honest-negative clause applies"
}
