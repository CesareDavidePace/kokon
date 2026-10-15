# Manual test of "Use model files I already have" (#14) in an isolated Kokon: its own config,
# data, ports and stable-diffusion.cpp folder under .test-data\own-models. The installed Kokon,
# %APPDATA%\Kokon\config.json and the real model folders are never written to.
#
# A fake ComfyUI folder is made of renamed hardlinks to the official files of your installed
# Kokon (no extra disk space; deleting or renaming a link never touches the original).
#
#   powershell -ExecutionPolicy Bypass -File scripts\dev\test-own-models.ps1 [-Action <action>] [options]
#
# Actions
#   setup     (default) Build the test environment and start Kokon on it (npm run tauri dev).
#             -Scenario fresh    first run, nothing configured (the setup wizard)
#             -Scenario migrate  a 0.9.2 config whose official profile was pointed at the
#                                ComfyUI files by hand (badraymen's config): 0.9.3 must keep them
#             -DownloadEngine    do not copy your engine: Setup downloads it (~0.9 GB with CUDA)
#             -Reset             delete the previous test environment first
#   move      Move the text encoder to another sub-folder (a file the user moved).
#   restore   Move it back.
#   unplug    Remove drive -Drive (default K:), made with "plug" (a disconnected drive).
#   plug      Map -Drive to the fake ComfyUI folder (choose K:\models in Setup to use it).
#   status    What the test Kokon sees: active profile, missing files (GET /models).
#   stop      Stop the test service and its sd-server (they outlive the window). Only processes
#             started from this repository or the test folder; the installed Kokon keeps running.
#
# Close the installed Kokon first: both would load a model on the same GPU.
param(
    [ValidateSet('setup', 'move', 'restore', 'unplug', 'plug', 'status', 'stop')] [string]$Action = 'setup',
    [ValidateSet('fresh', 'migrate')] [string]$Scenario = 'fresh',
    [string]$From = (Join-Path $env:LOCALAPPDATA 'KokonData\sdcpp'),
    [string]$Drive = 'K:',
    [switch]$DownloadEngine,
    [switch]$Reset
)

$ErrorActionPreference = 'Stop'
$root = Resolve-Path "$PSScriptRoot\..\.."
$test = Join-Path $root '.test-data\own-models'
$data = Join-Path $test 'data'
$config = Join-Path $test 'config'
$sdcpp = Join-Path $test 'sdcpp'
$comfy = Join-Path $test 'ComfyUI'
$models = Join-Path $comfy 'models'
$te = Join-Path $models 'text_encoders\qwen3vl_8b_instruct_q4km.gguf'
$teMoved = Join-Path $models 'text_encoders\old\qwen3vl_8b_instruct_q4km.gguf'
$realConfig = Join-Path $env:APPDATA 'Kokon\config.json'

function RealConfigHash {
    if (Test-Path $realConfig) { (Get-FileHash $realConfig -Algorithm SHA256).Hash.Substring(0, 16) } else { 'none' }
}

function StopTestProcesses {
    $mine = Get-Process kokon-service, sd-server -ErrorAction SilentlyContinue | Where-Object { $_.Path -and ($_.Path.StartsWith("$root\target", 'OrdinalIgnoreCase') -or $_.Path.StartsWith("$test", 'OrdinalIgnoreCase')) }
    foreach ($p in $mine) {
        Write-Host "Stopping $($p.Name) ($($p.Path))"
        Stop-Process -Id $p.Id -Force
    }
    if (-not $mine) { Write-Host 'No test processes running.' }
}

switch ($Action) {
    'stop' {
        StopTestProcesses
        return
    }
    'move' {
        New-Item -ItemType Directory -Force (Split-Path $teMoved) | Out-Null
        Move-Item $te $teMoved
        Write-Host "Moved the text encoder to $teMoved"
        return
    }
    'restore' {
        Move-Item $teMoved $te
        Write-Host "Moved the text encoder back to $te"
        return
    }
    'plug' {
        subst $Drive $comfy
        Write-Host "$Drive is now the fake ComfyUI folder. In Setup choose $Drive\models."
        return
    }
    'unplug' {
        subst $Drive /d
        Write-Host "$Drive removed (like a disconnected drive)."
        return
    }
    'status' {
        $info = Get-Content (Join-Path $data 'service.json') -Raw | ConvertFrom-Json
        $h = @{ Authorization = "Bearer $($info.token)" }
        $m = Invoke-RestMethod -Uri "http://127.0.0.1:$($info.port)/api/v1/models" -Headers $h
        Write-Host "Active: $($m.active.display_name) ($($m.active.source))"
        foreach ($s in $m.slots) {
            $what = if ($s.info) { " [$($s.info.weights), $([math]::Round($s.info.size / 1e9, 2)) GB]" } else { '' }
            Write-Host ("  {0,-26} {1} {2}{3}" -f $s.label, $(if ($s.exists) { 'ok     ' } else { 'MISSING' }), $s.path, $what)
        }
        Write-Host "Engine installed: $($m.engine_installed); official files installed: $($m.official_installed); scan folder: $($m.scan_dir)"
        Write-Host "Real config.json (must not change): $(RealConfigHash)"
        return
    }
}

# --- setup ---
$off = Join-Path $From 'models\qwen21'
$official = @{
    diffusion = Join-Path $off 'qwen-image-2.1-turbo-Q4_K.gguf'
    vae       = Join-Path $off 'qwen_image_2.1_vae_bf16.safetensors'
    te        = Join-Path $off 'Qwen3VL-8B-Instruct-Q4_K_M.gguf'
    mmproj    = Join-Path $off 'mmproj-Qwen3VL-8B-Instruct-F16.gguf'
}
foreach ($f in $official.Values) { if (-not (Test-Path $f)) { throw "Not found: $f. Pass -From with the folder of your installed Kokon models." } }
if ([IO.Path]::GetPathRoot($off) -ne [IO.Path]::GetPathRoot("$test")) { throw "Hardlinks need $test on the same drive as $off." }
if (Get-Process sd-server -ErrorAction SilentlyContinue) { Write-Warning "sd-server is running: close Kokon first, or both load a model on the GPU." }

# A test service left running would keep the old build (and lock its exe).
StopTestProcesses
if ($Reset -and (Test-Path $test)) {
    # Only the test folder; the model files in it are hardlinks (the originals stay).
    Remove-Item -Recurse -Force $test
}
New-Item -ItemType Directory -Force $data, $config, $sdcpp, (Join-Path $test 'output') | Out-Null

# The fake ComfyUI folder: official files under ComfyUI-style names (recognized by their
# contents, not as the official files), the VAE under its own name (as ComfyUI keeps it), and
# files of other kinds a scan must skip.
$links = @(
    @{ path = Join-Path $models 'diffusion_models\qwen_image_2.1_turbo_q4k.gguf'; target = $official.diffusion },
    @{ path = Join-Path $models 'vae\qwen_image_2.1_vae_bf16.safetensors'; target = $official.vae },
    @{ path = $te; target = $official.te },
    @{ path = Join-Path $models 'clip_vision\mmproj_qwen3vl_8b_f16.gguf'; target = $official.mmproj }
)
foreach ($l in $links) {
    if ((Test-Path $l.path) -or ($l.path -eq $te -and (Test-Path $teMoved))) { continue }
    New-Item -ItemType Directory -Force (Split-Path $l.path) | Out-Null
    New-Item -ItemType HardLink -Path $l.path -Target $l.target | Out-Null
}
New-Item -ItemType Directory -Force (Join-Path $models 'loras'), (Join-Path $comfy 'custom_nodes') | Out-Null
Set-Content (Join-Path $models 'loras\readme.txt') 'not a model'

# The engine (and upscalers) of your installed Kokon, unless Setup should download it.
if (-not $DownloadEngine -and -not (Test-Path (Join-Path $sdcpp 'sd-server.exe'))) {
    Get-ChildItem $From -File | Where-Object { $_.Extension -in '.exe', '.dll' -or $_.Name -eq 'kokon-engine.txt' } | Copy-Item -Destination $sdcpp
    $up = Join-Path $From 'models\upscalers'
    if (Test-Path $up) {
        New-Item -ItemType Directory -Force (Join-Path $sdcpp 'models\upscalers') | Out-Null
        Get-ChildItem $up -File | Copy-Item -Destination (Join-Path $sdcpp 'models\upscalers')
    }
    Write-Host "Copied the engine from $From"
}

$cfg = @{ sdcppDir = "$sdcpp"; serverPort = 39190; servicePort = 39191; outputDir = (Join-Path $test 'output'); idleUnloadMinutes = 5 }
if ($Scenario -eq 'migrate') {
    # 0.9.2: own files set by editing the official profile.
    $cfg.schemaVersion = 8
    $cfg.setupComplete = $true
    $cfg.activeProfile = 'qwen-image-2.1'
    $cfg.profiles = @(@{
            id = 'qwen-image-2.1'; displayName = 'Qwen-Image 2.1 Turbo Q4_K (official)'
            diffusionModel = (Join-Path $models 'diffusion_models\qwen_image_2.1_turbo_q4k.gguf')
            vae = (Join-Path $models 'vae\qwen_image_2.1_vae_bf16.safetensors')
            textEncoder = $te
            visionProjector = (Join-Path $models 'clip_vision\mmproj_qwen3vl_8b_f16.gguf')
        })
} else {
    $cfg.setupComplete = $false
}
# UTF-8 without BOM (Set-Content -Encoding utf8 in PowerShell 5.1 writes one; Kokon accepts both).
[IO.File]::WriteAllText((Join-Path $config 'config.json'), ($cfg | ConvertTo-Json -Depth 5))

Write-Host "Test environment: $test" -ForegroundColor Cyan
Write-Host "Fake ComfyUI models folder: $models"
Write-Host "Real config.json (must not change): $(RealConfigHash)"
Write-Host "Starting Kokon on it (close the window to stop)..." -ForegroundColor Cyan
$env:QLS_DATA_DIR = $data
$env:QLS_CONFIG_DIR = $config
Push-Location $root
try { npm run tauri dev } finally { Pop-Location }
