# Manual test of the 0.9.3 VAE update (#21) in an isolated Kokon: its own config, data, ports and
# stable-diffusion.cpp folder under .test-data\vae-update. The installed Kokon,
# %APPDATA%\Kokon\config.json and the real model folders are never written to.
#
# The test folder holds hardlinks to the official files of your installed Kokon (no extra disk
# space; when the update deletes the old VAE it deletes the link, never the original).
#
#   powershell -ExecutionPolicy Bypass -File scripts\dev\test-vae-update.ps1 [-Action <action>] [options]
#
# Actions
#   setup     (default) Build a 0.9.2 install and start Kokon on it (npm run tauri dev). What's new
#             opens with the "New image decoder" offer: Update now downloads 675 MB.
#             -Scenario official  the official files, original VAE (most users)
#             -Scenario own       the user's own files (#14) with their own VAE: accepting must
#                                 point the profile at Kokon's copy and keep their file
#             -Reset              start again as a fresh 0.9.2 install (deletes the test folder)
#             Without -Reset a second run is a restart: the config the app saved is kept.
#   status    What the test Kokon sees (GET /models) and which VAE files are on disk.
#   stop      Stop the test service and its sd-server (they outlive the window).
#
# Close the installed Kokon first: both would load a model on the same GPU.
param(
    [ValidateSet('setup', 'status', 'stop')] [string]$Action = 'setup',
    [ValidateSet('official', 'own')] [string]$Scenario = 'official',
    [string]$From = (Join-Path $env:LOCALAPPDATA 'KokonData\sdcpp'),
    [switch]$Reset
)

$ErrorActionPreference = 'Stop'
$root = Resolve-Path "$PSScriptRoot\..\.."
$test = Join-Path $root '.test-data\vae-update'
$data = Join-Path $test 'data'
$config = Join-Path $test 'config'
$sdcpp = Join-Path $test 'sdcpp'
$qwen = Join-Path $sdcpp 'models\qwen21'
$own = Join-Path $test 'own-vae\qwen_image_2.1_vae_bf16.safetensors'
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
    'status' {
        $info = Get-Content (Join-Path $data 'service.json') -Raw | ConvertFrom-Json
        $h = @{ Authorization = "Bearer $($info.token)" }
        $m = Invoke-RestMethod -Uri "http://127.0.0.1:$($info.port)/api/v1/models" -Headers $h
        Write-Host "Active: $($m.active.display_name) ($($m.active.source))"
        Write-Host "New VAE in use: $($m.vae.improved); download: $([math]::Round($m.vae.download_bytes / 1e6)) MB"
        $vae = ($m.slots | Where-Object { $_.role -eq 'vae' }).path
        Write-Host "VAE path: $vae"
        foreach ($f in @((Join-Path $qwen 'qwen_image_2.1_vae_bf16.safetensors'), (Join-Path $qwen 'texture_fix_vae_for_qwen_image_2.1_bf16.safetensors'), $own)) {
            Write-Host ("  {0} {1}" -f $(if (Test-Path $f) { 'present' } else { 'absent ' }), $f)
        }
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

StopTestProcesses
if ($Reset -and (Test-Path $test)) {
    # Only the test folder; the model files in it are hardlinks (the originals stay).
    Remove-Item -Recurse -Force $test
}
New-Item -ItemType Directory -Force $data, $config, $qwen, (Join-Path $test 'output') | Out-Null

# A 0.9.2 install: the official files with the original VAE (hardlinks).
foreach ($f in $official.Values) {
    $link = Join-Path $qwen (Split-Path $f -Leaf)
    if (-not (Test-Path $link)) { New-Item -ItemType HardLink -Path $link -Target $f | Out-Null }
}
if ($Scenario -eq 'own' -and -not (Test-Path $own)) {
    New-Item -ItemType Directory -Force (Split-Path $own) | Out-Null
    New-Item -ItemType HardLink -Path $own -Target $official.vae | Out-Null
}

# The engine and upscalers of your installed Kokon.
if (-not (Test-Path (Join-Path $sdcpp 'sd-server.exe'))) {
    Get-ChildItem $From -File | Where-Object { $_.Extension -in '.exe', '.dll' -or $_.Name -eq 'kokon-engine.txt' } | Copy-Item -Destination $sdcpp
    $up = Join-Path $From 'models\upscalers'
    if (Test-Path $up) {
        New-Item -ItemType Directory -Force (Join-Path $sdcpp 'models\upscalers') | Out-Null
        Get-ChildItem $up -File | Copy-Item -Destination (Join-Path $sdcpp 'models\upscalers')
    }
    Write-Host "Copied the engine from $From"
}

$cfg = @{
    schemaVersion = 9; setupComplete = $true; sdcppDir = "$sdcpp"; serverPort = 39192; servicePort = 39193
    outputDir = (Join-Path $test 'output'); idleUnloadMinutes = 5; activeProfile = 'qwen-image-2.1'
    # Updating from 0.9.2: the welcome was seen, the 0.9.2 notes too.
    ui = @{ welcomeSeen = $true; tourSeen = $true; whatsNewSeen = '0.9.2' }
}
if ($Scenario -eq 'own') {
    $cfg.activeProfile = 'qwen-image-2.1-user'
    $cfg.profiles = @(@{
            id = 'qwen-image-2.1-user'; displayName = 'Qwen-Image 2.1 (your files)'; model = 'qwen-image-2.1'; source = 'user'
            diffusionModel = (Join-Path $qwen 'qwen-image-2.1-turbo-Q4_K.gguf'); vae = $own
            textEncoder = (Join-Path $qwen 'Qwen3VL-8B-Instruct-Q4_K_M.gguf'); visionProjector = (Join-Path $qwen 'mmproj-Qwen3VL-8B-Instruct-F16.gguf')
        })
}
# Written once: a second run (without -Reset) continues from what the app saved, like a restart.
$cfgFile = Join-Path $config 'config.json'
if (-not (Test-Path $cfgFile)) {
    # UTF-8 without BOM (Set-Content -Encoding utf8 in PowerShell 5.1 writes one; Kokon accepts both).
    [IO.File]::WriteAllText($cfgFile, ($cfg | ConvertTo-Json -Depth 5))
} else {
    Write-Host "Keeping the saved test config (use -Reset to start again as a 0.9.2 install)."
}

Write-Host "Test environment: $test ($Scenario)" -ForegroundColor Cyan
Write-Host "Real config.json (must not change): $(RealConfigHash)"
Write-Host "Starting Kokon on it (close the window to stop)..." -ForegroundColor Cyan
$env:QLS_DATA_DIR = $data
$env:QLS_CONFIG_DIR = $config
Push-Location $root
try { npm run tauri dev } finally { Pop-Location }
