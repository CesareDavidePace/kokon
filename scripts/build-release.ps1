# Builds the Windows release: qls-service.exe, qls-mcp.exe, the desktop app and the
# NSIS installer ("Qwen Local Studio_<version>_x64-setup.exe").
#
#   powershell -ExecutionPolicy Bypass -File scripts\build-release.ps1
#
# Takes ~5-15 minutes on first run (optimized Rust build). CPU only.

$ErrorActionPreference = 'Stop'
$root = Resolve-Path "$PSScriptRoot\.."
Set-Location $root
$env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"

Write-Host "1/5 Tests (Rust core + frontend + MCP)" -ForegroundColor Cyan
cargo test -p qls-core --release --quiet
if ($LASTEXITCODE) { throw "Rust tests failed" }
npx vitest run
if ($LASTEXITCODE) { throw "Frontend tests failed" }
Push-Location mcp; npx vitest run; $rc = $LASTEXITCODE; Pop-Location
if ($rc) { throw "MCP tests failed" }

Write-Host "2/5 qls-service (release)" -ForegroundColor Cyan
cargo build -p qls-service --release
if ($LASTEXITCODE) { throw "service build failed" }
Copy-Item target\release\qls-service.exe src-tauri\binaries\qls-service-x86_64-pc-windows-msvc.exe -Force

Write-Host "3/5 qls-mcp.exe" -ForegroundColor Cyan
Push-Location mcp; npm run build:exe; $rc = $LASTEXITCODE; Pop-Location
if ($rc) { throw "MCP build failed" }

Write-Host "4/5 Desktop app + NSIS installer" -ForegroundColor Cyan
npx tauri build
if ($LASTEXITCODE) { throw "tauri build failed" }

Write-Host "5/5 Artifacts" -ForegroundColor Cyan
$exe = Get-Item "target\release\qwen-local-studio.exe"
$setup = Get-ChildItem "target\release\bundle\nsis\*.exe" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
New-Item -ItemType Directory -Force dist-release | Out-Null
Copy-Item $exe.FullName "dist-release\Qwen Local Studio.exe" -Force
Copy-Item $setup.FullName "dist-release\Qwen Local Studio Setup.exe" -Force
Copy-Item target\release\qls-service.exe dist-release\ -Force
Copy-Item mcp\build\qls-mcp.exe dist-release\ -Force
Get-ChildItem dist-release | Format-Table Name, @{n = 'MB'; e = { [math]::Round($_.Length / 1MB, 1) } } -AutoSize
Write-Host "Installer: $root\dist-release\Qwen Local Studio Setup.exe" -ForegroundColor Green
