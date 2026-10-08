# Milestone 1 — backend foundation, real inference.
# Starts kokon-service (debug build) with isolated test data, which launches sd-server,
# then runs text-to-image and an edit, printing truthful progress from the backend.
#
#   powershell -ExecutionPolicy Bypass -File scripts\dev\m1-backend-test.ps1
#
# Takes ~3-5 minutes (model load + two Fast generations).

$ErrorActionPreference = 'Stop'
$root = Resolve-Path "$PSScriptRoot\..\.."
$test = Join-Path $root '.test-data'
$env:QLS_DATA_DIR = Join-Path $test 'data'
$env:QLS_CONFIG_DIR = Join-Path $test 'config'
$out = Join-Path $test 'output'
New-Item -ItemType Directory -Force $env:QLS_CONFIG_DIR, $out | Out-Null
$port = 39171
# Test config: real models, isolated output folder and ports.
@{ outputDir = $out; serverPort = 39170; servicePort = $port } | ConvertTo-Json | Set-Content (Join-Path $env:QLS_CONFIG_DIR 'config.json') -Encoding utf8

$exe = Join-Path $root 'target\debug\kokon-service.exe'
if (-not (Test-Path $exe)) { throw "Build first: cargo build -p qls-service" }
$svc = Start-Process $exe -ArgumentList '--port', $port -PassThru -WindowStyle Hidden
Start-Sleep 2
$info = Get-Content (Join-Path $env:QLS_DATA_DIR 'service.json') | ConvertFrom-Json
$H = @{ Authorization = "Bearer $($info.token)" }
$B = "http://127.0.0.1:$port/api/v1"

function Invoke-Job($path, $body) {
    $json = $body | ConvertTo-Json -Depth 6
    $job = Invoke-RestMethod -Method Post -Uri "$B/$path`?wait=false" -Headers $H -ContentType 'application/json' -Body ([Text.Encoding]::UTF8.GetBytes($json))
    $sw = [Diagnostics.Stopwatch]::StartNew(); $last = ''
    while ($true) {
        $j = Invoke-RestMethod -Uri "$B/jobs/$($job.id)" -Headers $H
        $step = if ($j.step) { " $($j.step[0])/$($j.step[1])" } else { '' }
        $line = "$($j.status) $($j.stage)$step"
        if ($line -ne $last) { Write-Host ("  [{0,5:N1}s] {1}" -f $sw.Elapsed.TotalSeconds, $line); $last = $line }
        if ($j.status -in 'done', 'failed', 'cancelled') { break }
        Start-Sleep -Milliseconds 400
    }
    if ($j.status -ne 'done') { Write-Host "  ERROR: $($j.error | ConvertTo-Json -Depth 4)" -ForegroundColor Red; return $null }
    $r = $j.results[0]
    Write-Host "  plan: $($j.plan.mode) working $($j.plan.working.width)x$($j.plan.working.height) -> output $($r.width)x$($r.height), steps $($j.plan.steps), cfg $($j.plan.cfg), seed $($r.seed)"
    if ($j.plan.notes) { Write-Host "  notes: $($j.plan.notes -join ' | ')" }
    Write-Host "  file: $($r.file_path)" -ForegroundColor Green
    return $r
}

try {
    Write-Host "1) Text to image (16:9 from the prompt, Fast)" -ForegroundColor Cyan
    $t2i = Invoke-Job 'generate' @{ prompt = 'A hero image for a coffee roastery landing page, warm morning light, 16:9'; quality = 'fast' }

    if ($t2i) {
        Write-Host "2) Edit the result (dimensions must be preserved)" -ForegroundColor Cyan
        $ed = Invoke-Job 'edit' @{ image = $t2i.file_path; prompt = 'make it nighttime with warm window lights'; quality = 'fast' }
        if ($ed) {
            $same = ($ed.width -eq $t2i.width) -and ($ed.height -eq $t2i.height)
            Write-Host ("  dimensions preserved: {0}" -f $same) -ForegroundColor ($(if ($same) { 'Green' } else { 'Red' }))
        }
    }
    $st = Invoke-RestMethod -Uri "$B/status" -Headers $H
    Write-Host "Server: $($st.server.state), owned=$($st.server.owned), pid=$($st.server.pid), model=$($st.server.model)"
}
finally {
    Write-Host "3) Shutdown (sd-server must stop)" -ForegroundColor Cyan
    try { Invoke-RestMethod -Method Post -Uri "$B/shutdown" -Headers $H | Out-Null } catch {}
    Start-Sleep 4
    $left = Get-Process sd-server -ErrorAction SilentlyContinue
    if ($left) { Write-Host "  sd-server still running: $($left.Id -join ', ')" -ForegroundColor Red } else { Write-Host "  sd-server stopped" -ForegroundColor Green }
    Write-Host "Service log: $env:QLS_DATA_DIR\logs\service.log"
    Write-Host "Server log:  $env:QLS_DATA_DIR\logs\sd-server.log"
}
