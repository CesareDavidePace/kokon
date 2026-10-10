# 12 GB verification on a bigger card. Holds VRAM with vram-ballast.py so the GPU really has
# what a 12 GB card has, tells the planner the card has 12 GB (vramOverrideMib), then runs
# every kind of job through the real engine and records time, sizes, VRAM peaks, spills to
# shared memory and out-of-memory retries.
#
#   cargo build -p qls-service
#   powershell -ExecutionPolicy Bypass -File scripts\dev\test-12gb.ps1 [-Only t2i-fast,edit-balanced] [-Image C:\path\photo.jpg]
#
# Close Kokon first (the GPU must be free). About 20-30 minutes for everything.
# Results: docs\test-results\12gb-<date>.md (paste it back in the chat).
#
# -CardMib    total VRAM of the card to emulate, as nvidia-smi reports it (12 GB: 12282)
# -SdcppDir   stable-diffusion.cpp folder with the models (official Q4_K_M preferred)
# -Image      a large photo to edit (optional; otherwise only generated images are edited)
param([string[]]$Only, [string]$Image, [int]$CardMib = 12282, [string]$SdcppDir = 'C:\AI\sdcpp', [switch]$NoBallast)

$ErrorActionPreference = 'Stop'
$Only = @($Only | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($Only.Count -eq 0) { $Only = $null }
function Want($n) { return (-not $Only) -or ($n -in $Only) }

$root = Resolve-Path "$PSScriptRoot\..\.."
$test = Join-Path $root '.test-data\12gb'
$env:QLS_DATA_DIR = Join-Path $test 'data'
$env:QLS_CONFIG_DIR = Join-Path $test 'config'
$out = Join-Path $test 'output'
New-Item -ItemType Directory -Force $env:QLS_CONFIG_DIR, $out, $env:QLS_DATA_DIR | Out-Null

$exe = Join-Path $root 'target\debug\kokon-service.exe'
if (-not (Test-Path $exe)) { throw "Build first: cargo build -p qls-service" }
if (Get-Process sd-server -ErrorAction SilentlyContinue) { throw "sd-server is running. Close Kokon (and any sd-server) first." }

# The model the setup wizard installs (Qwen-Image 2.1 Turbo, Q4_K).
$m = Join-Path $SdcppDir 'models\qwen21'
$diffusion = Join-Path $m 'qwen-image-2.1-turbo-Q4_K.gguf'
if (-not (Test-Path $diffusion)) { throw "Model not found: $diffusion. Run Settings -> Models -> Set up models first." }
$prof = @{
    id = 'qwen-image-2.1'; displayName = 'test'; diffusionModel = $diffusion
    vae = (Join-Path $m 'qwen_image_2.1_vae_bf16.safetensors')
    textEncoder = (Join-Path $m 'Qwen3VL-8B-Instruct-Q4_K_M.gguf')
    visionProjector = (Join-Path $m 'mmproj-Qwen3VL-8B-Instruct-F16.gguf')
}
@{
    sdcppDir = $SdcppDir; serverPort = 39190; servicePort = 39191; outputDir = $out; setupComplete = $true
    hardwarePreset = '12gb'; vramOverrideMib = $CardMib; activeProfile = 'qwen-image-2.1'; profiles = @($prof)
} | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $env:QLS_CONFIG_DIR 'config.json') -Encoding utf8

# 1. Ballast: the memory a 12 GB card does not have.
$ballast = $null
if (-not $NoBallast) {
    $blog = Join-Path $test 'ballast.log'
    Remove-Item $blog -ErrorAction SilentlyContinue
    $ballast = Start-Process python -ArgumentList '-I', "`"$root\scripts\dev\vram-ballast.py`"", '--card-mib', $CardMib, '--sdcpp', "`"$SdcppDir`"" `
        -RedirectStandardOutput $blog -RedirectStandardError "$blog.err" -PassThru -WindowStyle Hidden
    for ($i = 0; $i -lt 40 -and -not (Select-String -Path $blog -Pattern 'Holding' -Quiet -ErrorAction SilentlyContinue); $i++) { Start-Sleep -Milliseconds 250 }
    if ($ballast.HasExited -or -not (Select-String -Path $blog -Pattern 'Holding' -Quiet)) { throw "Ballast failed: $(Get-Content "$blog.err" -Raw)" }
    Write-Host (Get-Content $blog | Select-Object -First 1) -ForegroundColor Cyan
}

# 2. Service (isolated data, ports and output).
$svc = Start-Process $exe -ArgumentList '--port', 39191 -PassThru -WindowStyle Hidden
Start-Sleep 2
$info = Get-Content (Join-Path $env:QLS_DATA_DIR 'service.json') | ConvertFrom-Json
$H = @{ Authorization = "Bearer $($info.token)" }
$B = "http://127.0.0.1:39191/api/v1"

# 3. VRAM sampler: whole-GPU dedicated use (nvidia-smi) and sd-server's shared (spilled) memory.
$samples = Join-Path $test 'vram.csv'
"t,used_mib,shared_mib" | Set-Content $samples
$sampler = Start-Job -ArgumentList $samples -ScriptBlock {
    param($file)
    while ($true) {
        $used = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1).Trim()
        $shared = 0
        $p = Get-Process sd-server -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($p) {
            try {
                $c = Get-Counter "\GPU Process Memory(pid_$($p.Id)*)\Shared Usage" -ErrorAction Stop
                $shared = [math]::Round((($c.CounterSamples | Measure-Object CookedValue -Sum).Sum) / 1MB)
            } catch {}
        }
        "$([DateTimeOffset]::Now.ToUnixTimeMilliseconds()),$used,$shared" | Add-Content $file
        Start-Sleep -Milliseconds 500
    }
}
function Peaks($fromMs, $toMs) {
    $rows = Import-Csv $samples | Where-Object { [int64]$_.t -ge $fromMs -and [int64]$_.t -le $toMs }
    if (-not $rows) { return @('?', '?') }
    $u = ($rows | ForEach-Object { [int]$_.used_mib } | Measure-Object -Maximum).Maximum
    $s = ($rows | ForEach-Object { [int]$_.shared_mib } | Measure-Object -Maximum).Maximum
    return @($u, $s)
}

$results = @()
$gens = @{}
$sw = [Diagnostics.Stopwatch]::StartNew()
try {
    Write-Host "== Loading the model" -ForegroundColor Cyan
    $t0 = [DateTimeOffset]::Now.ToUnixTimeMilliseconds()
    Invoke-RestMethod -Method Post "$B/backend/start" -Headers $H -TimeoutSec 900 | Out-Null
    $pk = Peaks $t0 ([DateTimeOffset]::Now.ToUnixTimeMilliseconds())
    $results += [pscustomobject]@{ Case = 'model load'; Status = 'done'; Seconds = [math]::Round(([DateTimeOffset]::Now.ToUnixTimeMilliseconds() - $t0) / 1000); Working = ''; Output = ''; SecPerStep = ''; PeakVramMiB = $pk[0]; PeakSharedMiB = $pk[1]; Notes = '' }

    function Run($name, $body) {
        if (-not (Want $name)) { return $null }
        $body.seed = 42
        $body.client = 'test-12gb'
        $json = $body | ConvertTo-Json -Depth 6
        $logBefore = (Invoke-RestMethod "$B/logs?lines=4000" -Headers $H).lines.Count
        $t0 = [DateTimeOffset]::Now.ToUnixTimeMilliseconds()
        $job = Invoke-RestMethod -Method Post -Uri "$B/jobs" -Headers $H -ContentType 'application/json' -Body ([Text.Encoding]::UTF8.GetBytes($json))
        Write-Host "== $name" -ForegroundColor Cyan
        $last = ''; $sps = 0
        while ($true) {
            $j = Invoke-RestMethod -Uri "$B/jobs/$($job.id)/wait?timeout=2" -Headers $H
            if ($j.seconds_per_step -and $j.seconds_per_step -gt $sps) { $sps = $j.seconds_per_step }
            $step = if ($j.step) { " $($j.step[0])/$($j.step[1])" } else { '' }
            $line = "$($j.status) $($j.stage)$step"
            if ($line -ne $last) { Write-Host ("  [{0,6:N1}s] {1}" -f (([DateTimeOffset]::Now.ToUnixTimeMilliseconds() - $t0) / 1000), $line); $last = $line }
            if ($j.status -in 'done', 'failed', 'cancelled') { break }
        }
        $t1 = [DateTimeOffset]::Now.ToUnixTimeMilliseconds()
        $pk = Peaks $t0 $t1
        $log = (Invoke-RestMethod "$B/logs?lines=4000" -Headers $H).lines | Select-Object -Skip ([math]::Max(0, $logBefore - 50))
        $memLines = @($log | Where-Object { $_ -match 'out of memory|insufficient memory|retrying without|failed to allocate|OOM' } | Select-Object -Unique)
        $notes = @($j.notes) + @($memLines | ForEach-Object { "log: $($_.Trim())" })
        if ($j.error) { $notes += "error: $($j.error.message)" }
        $g = $j.results | Select-Object -First 1
        if ($g) { $gens[$name] = $g }
        $p = $j.plan
        $script:results += [pscustomobject]@{
            Case = $name; Status = $j.status; Seconds = [math]::Round(($t1 - $t0) / 1000)
            Working = if ($p) { "$($p.working.width)x$($p.working.height)" } else { '' }
            Output = if ($g) { "$($g.width)x$($g.height)" } else { '' }
            SecPerStep = if ($sps) { [math]::Round($sps, 2) } else { '' }
            PeakVramMiB = $pk[0]; PeakSharedMiB = $pk[1]
            Notes = ($notes | Where-Object { $_ } | Select-Object -Unique) -join ' | '
        }
        Write-Host ("  {0} in {1}s, working {2}, peak {3} MiB, shared {4} MiB" -f $j.status, [math]::Round(($t1 - $t0) / 1000), $script:results[-1].Working, $pk[0], $pk[1])
        return $g
    }

    $scene = 'a red vintage bicycle leaning against a sunlit brick wall in a quiet street, photo'
    Run 't2i-fast' @{ prompt = $scene; mode = 'generate'; quality = 'fast' } | Out-Null
    Run 't2i-balanced' @{ prompt = $scene; mode = 'generate'; quality = 'balanced'; aspect_ratio = '16:9' } | Out-Null
    Run 't2i-quality' @{ prompt = $scene; mode = 'generate'; quality = 'quality' } | Out-Null
    $src = if ($gens['t2i-balanced']) { "gen:$($gens['t2i-balanced'].id)" } elseif ($gens['t2i-fast']) { "gen:$($gens['t2i-fast'].id)" } else { $null }
    if ($Image) { $photo = (Resolve-Path $Image).Path } else { $photo = $null }
    if ($src) {
        Run 'edit-balanced' @{ prompt = 'make it night, with warm street lights and wet pavement'; mode = 'edit'; images = @($src); quality = 'balanced' } | Out-Null
        Run 'edit-quality' @{ prompt = 'make it night, with warm street lights and wet pavement'; mode = 'edit'; images = @($src); quality = 'quality' } | Out-Null
        # Inpaint (Remove): a mask over the middle third of the image.
        if (Want 'inpaint-remove') {
            Add-Type -AssemblyName System.Drawing
            $g0 = $gens['t2i-balanced']; if (-not $g0) { $g0 = $gens['t2i-fast'] }
            $bmp = New-Object System.Drawing.Bitmap $g0.width, $g0.height
            $gr = [System.Drawing.Graphics]::FromImage($bmp)
            $gr.Clear([System.Drawing.Color]::Black)
            $gr.FillEllipse([System.Drawing.Brushes]::White, [int]($g0.width / 3), [int]($g0.height / 4), [int]($g0.width / 3), [int]($g0.height / 2))
            $mask = Join-Path $test 'mask.png'
            $bmp.Save($mask, [System.Drawing.Imaging.ImageFormat]::Png); $gr.Dispose(); $bmp.Dispose()
            Run 'inpaint-remove' @{ prompt = ''; mode = 'inpaint'; images = @($src); mask = $mask; inpaint_action = 'remove'; quality = 'balanced' } | Out-Null
        }
        Run 'outpaint' @{ prompt = 'continue the street'; mode = 'outpaint'; images = @($src); quality = 'balanced'; outpaint = @{ left = 25; right = 25; top = 0; bottom = 0; units = 'percent' } } | Out-Null
        if ($gens['t2i-fast']) {
            Run 'multi-ref' @{ prompt = 'put the bicycle from image 1 into the street of image 2, at night'; mode = 'multi_reference'; images = @("gen:$($gens['t2i-fast'].id)", $src); quality = 'balanced' } | Out-Null
        }
        Run 'upscale-detailed' @{ prompt = ''; mode = 'upscale'; images = @($src); upscale = @{ method = 'detailed'; scale = 2 } } | Out-Null
    }
    if ($photo) {
        Run 'photo-edit-balanced' @{ prompt = 'replace the sky with a dramatic sunset'; mode = 'edit'; images = @($photo); quality = 'balanced' } | Out-Null
        Run 'photo-edit-quality' @{ prompt = 'replace the sky with a dramatic sunset'; mode = 'edit'; images = @($photo); quality = 'quality' } | Out-Null
    }
}
finally {
    $status = $null
    try { $status = Invoke-RestMethod "$B/status" -Headers $H } catch {}
    try { Invoke-RestMethod -Method Post "$B/backend/stop" -Headers $H | Out-Null } catch {}
    try { Invoke-RestMethod -Method Post "$B/shutdown" -Headers $H | Out-Null } catch {}
    Stop-Job $sampler -ErrorAction SilentlyContinue; Remove-Job $sampler -Force -ErrorAction SilentlyContinue
    if ($ballast -and -not $ballast.HasExited) { Stop-Process -Id $ballast.Id -Force }
    Start-Sleep 1
    if (-not $svc.HasExited) { Stop-Process -Id $svc.Id -Force -ErrorAction SilentlyContinue }

    $gpu = (& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | Select-Object -First 1)
    $policy = 'unknown (NVIDIA Control Panel > Manage 3D settings > CUDA - Sysmem Fallback Policy)'
    $md = @(
        "# 12 GB verification - $(Get-Date -Format 'yyyy-MM-dd HH:mm')", '',
        "- GPU: $gpu; emulated card: $CardMib MiB (ballast: $(if ($NoBallast) { 'off' } else { 'on' }), planner override: $CardMib MiB)",
        "- Model: $diffusion", "- Sysmem fallback policy: $policy",
        "- Total wall time: $([math]::Round($sw.Elapsed.TotalMinutes, 1)) min", '',
        '| Case | Status | Seconds | Working | Output | s/step | Peak VRAM MiB (whole GPU, incl. ballast) | Peak shared MiB (sd-server) | Notes |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |'
    )
    foreach ($r in $results) { $md += "| $($r.Case) | $($r.Status) | $($r.Seconds) | $($r.Working) | $($r.Output) | $($r.SecPerStep) | $($r.PeakVramMiB) | $($r.PeakSharedMiB) | $($r.Notes -replace '\|', '/') |" }
    $md += '', "Peak VRAM includes the ballast; a real 12 GB card has $CardMib MiB in total. Shared MiB above a few hundred means sd-server spilled into system memory (slow, but no crash with the default fallback policy)."
    $md += "Images: $out"
    $dir = Join-Path $root 'docs\test-results'
    New-Item -ItemType Directory -Force $dir | Out-Null
    $file = Join-Path $dir "12gb-$(Get-Date -Format 'yyyy-MM-dd-HHmm').md"
    $md | Set-Content $file -Encoding utf8
    $results | Format-Table Case, Status, Seconds, Working, Output, SecPerStep, PeakVramMiB, PeakSharedMiB -AutoSize
    Write-Host "Results: $file" -ForegroundColor Green
}
