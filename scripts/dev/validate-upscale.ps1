# Speed & upscaling validation through the real app engine (planner, two-pass refine,
# ESRGAN enlargement, Upscale action). Starts qls-service (debug build) with isolated
# test data and ports, which launches sd-server.
#
#   cargo build -p qls-service
#   powershell -ExecutionPolicy Bypass -File scripts\dev\validate-upscale.ps1 [-Only fast,best,4k,edit,upscale,logo] [-EditImage C:\path\photo.png]
#
# Close Qwen Local Studio first (GPU). Takes ~12-15 minutes for everything.
# Without -EditImage, a 5120x2880 test photo is made by upscaling the 2K result.
param([string[]]$Only, [string]$EditImage)

$ErrorActionPreference = 'Stop'
$Only = @($Only | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($Only.Count -eq 0) { $Only = $null }
function Want($n) { return (-not $Only) -or ($n -in $Only) }

$root = Resolve-Path "$PSScriptRoot\..\.."
$test = Join-Path $root '.test-data'
$env:QLS_DATA_DIR = Join-Path $test 'data'
$env:QLS_CONFIG_DIR = Join-Path $test 'config'
$out = Join-Path $test 'output'
New-Item -ItemType Directory -Force $env:QLS_CONFIG_DIR, $out | Out-Null
$port = 39181
# Test config: real models, defaults for everything new (refineMode fast, ESRGAN from
# <sdcpp>\models\upscalers), isolated output folder and ports.
@{ outputDir = $out; serverPort = 39180; servicePort = $port; setupComplete = $true } | ConvertTo-Json | Set-Content (Join-Path $env:QLS_CONFIG_DIR 'config.json') -Encoding utf8

$exe = Join-Path $root 'target\debug\qls-service.exe'
if (-not (Test-Path $exe)) { throw "Build first: cargo build -p qls-service" }
$svc = Start-Process $exe -ArgumentList '--port', $port -PassThru -WindowStyle Hidden
Start-Sleep 2
$info = Get-Content (Join-Path $env:QLS_DATA_DIR 'service.json') | ConvertFrom-Json
$H = @{ Authorization = "Bearer $($info.token)" }
$B = "http://127.0.0.1:$port/api/v1"
$results = @()

function Dims($s) { if ($s) { return "$($s.width)x$($s.height)" } return '?' }
function Pipeline($g) {
    $p = $g.pipeline
    if (-not $p) { return 'single pass' }
    $parts = @()
    if ($p.base) { $parts += "$(Dims $p.base) -> refine $($g.working_width)x$($g.working_height) ($($p.refine), $($p.refine_steps) steps, $($p.hires_upscaler))" }
    if ($p.enlarged_with) {
        if (-not $parts) { $parts += (Dims $p.enlarged_from) }
        $parts += "$($g.width)x$($g.height) ($($p.enlarged_with))"
    }
    return ($parts -join ' -> ')
}

function Run($name, $body) {
    $json = $body | ConvertTo-Json -Depth 6
    $job = Invoke-RestMethod -Method Post -Uri "$B/jobs" -Headers $H -ContentType 'application/json' -Body ([Text.Encoding]::UTF8.GetBytes($json))
    Write-Host "== $name (plan: estimate $([math]::Round($job.plan.estimated_seconds))s)" -ForegroundColor Cyan
    $sw = [Diagnostics.Stopwatch]::StartNew(); $last = ''
    while ($true) {
        $j = Invoke-RestMethod -Uri "$B/jobs/$($job.id)" -Headers $H
        $step = if ($j.step) { " $($j.step[0])/$($j.step[1])" } else { '' }
        $line = "$($j.status) $($j.stage)$step"
        if ($line -ne $last) { Write-Host ("  [{0,5:N1}s] {1}" -f $sw.Elapsed.TotalSeconds, $line); $last = $line }
        if ($j.status -in 'done', 'failed', 'cancelled') { break }
        Start-Sleep -Milliseconds 400
    }
    $secs = [math]::Round($sw.Elapsed.TotalSeconds, 1)
    if ($j.status -ne 'done') {
        Write-Host "  ERROR: $($j.error | ConvertTo-Json -Depth 4)" -ForegroundColor Red
        $script:results += [pscustomobject]@{ Test = $name; Seconds = $secs; Estimate = [math]::Round($j.plan.estimated_seconds); Size = ''; Pipeline = "FAILED: $($j.error.message)"; File = '' }
        return $null
    }
    $g = $j.results[0]
    Write-Host "  $($g.width)x$($g.height) in $secs s - $(Pipeline $g)" -ForegroundColor Green
    if ($j.notes) { Write-Host "  notes: $($j.notes -join ' ')" -ForegroundColor DarkGray }
    $script:results += [pscustomobject]@{ Test = $name; Seconds = $secs; Estimate = [math]::Round($j.plan.estimated_seconds); Size = "$($g.width)x$($g.height)"; Pipeline = (Pipeline $g); File = $g.file_path }
    return $g
}

$prompt = 'a cozy mountain cabin at golden hour, snowy peaks, warm light in the windows, highly detailed photo'
# 20 base steps like the 2K bench (C:\AI\sdcpp\bench\bench-2k.ps1) so timings compare.
$gen2k = @{ prompt = $prompt; mode = 'generate'; images = @(); aspect_ratio = '16:9'; quality = 'quality'; resolution = 'auto'; seed = 42; advanced = @{ steps = 20 } }

try {
    Write-Host 'Loading the model (first job includes ~1 min of model load)...' -ForegroundColor Cyan
    Invoke-RestMethod -Method Post -Uri "$B/backend/start" -Headers $H -TimeoutSec 900 | Out-Null
    $st = Invoke-RestMethod -Uri "$B/status" -Headers $H
    Write-Host "Upscaler: $($st.upscaler | ConvertTo-Json -Compress); server image upscale: $($st.server.upscale)"

    $fast = $null
    if (Want 'fast') { $fast = Run 'gen2k_fast' $gen2k }
    if (Want 'best') {
        $b = $gen2k.Clone(); $b.advanced = @{ steps = 20; refine = 'best' }
        Run 'gen2k_best' $b | Out-Null
    }
    if (Want '4k') {
        $k = $gen2k.Clone(); $k.resolution = 'custom'; $k.width = 3840; $k.height = 2160; $k.Remove('aspect_ratio')
        Run 'gen4k_fast' $k | Out-Null
    }
    if (Want 'edit') {
        $src = $EditImage
        if (-not $src) {
            if (-not $fast) { $fast = Run 'gen2k_fast' $gen2k }
            $big = Run 'make_5120' @{ prompt = ''; mode = 'upscale'; images = @("gen:$($fast.id)"); upscale = @{ method = 'quick'; width = 5120; height = 2880 }; advanced = @{} }
            $src = "gen:$($big.id)"
        }
        $edit = @{ prompt = 'make it a dark version: night, moonlight, keep everything else unchanged'; mode = 'edit'; images = @($src); quality = 'balanced'; resolution = 'auto'; seed = 3; advanced = @{} }
        Run 'edit_balanced' $edit | Out-Null     # ~1 MP, ESRGAN back to the source size
        $e2 = $edit.Clone(); $e2.resolution = 'source'
        Run 'edit_source_2pass' $e2 | Out-Null   # base ~1 MP + refine 2048x1152, ESRGAN to the source size
    }
    if (Want 'upscale') {
        if (-not $fast) { $fast = Run 'gen2k_fast' $gen2k }
        Run 'upscale_quick_2x' @{ prompt = ''; mode = 'upscale'; images = @("gen:$($fast.id)"); upscale = @{ method = 'quick'; scale = 2 }; advanced = @{} } | Out-Null
        $small = Run 'gen1mp' @{ prompt = $prompt; mode = 'generate'; images = @(); aspect_ratio = '16:9'; quality = 'fast'; resolution = 'auto'; seed = 5; advanced = @{} }
        if ($small) {
            Run 'upscale_detailed_4k' @{ prompt = ''; mode = 'upscale'; images = @("gen:$($small.id)"); upscale = @{ method = 'detailed'; width = 3840 }; advanced = @{} } | Out-Null
            Run 'upscale_quick_4k' @{ prompt = ''; mode = 'upscale'; images = @("gen:$($small.id)"); upscale = @{ method = 'quick'; width = 3840 }; advanced = @{} } | Out-Null
        }
    }
    if (Want 'logo') {
        $logo = Run 'logo_1mp' @{ prompt = "flat vector logo for a coffee shop called 'BEAN THERE', thin black line art and lettering on a plain white background"; mode = 'generate'; images = @(); aspect_ratio = '1:1'; quality = 'fast'; resolution = 'auto'; seed = 11; advanced = @{} }
        if ($logo) { Run 'logo_quick_4x' @{ prompt = ''; mode = 'upscale'; images = @("gen:$($logo.id)"); upscale = @{ method = 'quick'; scale = 4 }; advanced = @{} } | Out-Null }
    }
} finally {
    Write-Host ''
    $results | Format-Table Test, Seconds, Estimate, Size, Pipeline -AutoSize -Wrap
    $results | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 (Join-Path $out 'validate-results.json')
    Write-Host "Images in $out (results: validate-results.json)."
    Invoke-RestMethod -Method Post -Uri "$B/shutdown" -Headers $H -ErrorAction SilentlyContinue | Out-Null
    Start-Sleep 2
    if (-not $svc.HasExited) { Stop-Process -Id $svc.Id -Force }
}
