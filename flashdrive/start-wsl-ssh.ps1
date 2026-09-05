# start-wsl-ssh.ps1 - make WSL sshd reachable from LAN
# Usage (GPU machine, every time after boot/restart):
#   Right-click PowerShell -> Run as Administrator, then:
#   powershell -ExecutionPolicy Bypass -File .\start-wsl-ssh.ps1
#
# What it does: starts sshd inside WSL, forwards Windows port 22 to the
# current WSL IP (WSL IP changes on every restart), opens the firewall.
# Result: from other machines on the LAN,  ssh user@<this-pc-wifi-ip>

#Requires -RunAsAdministrator

$ErrorActionPreference = "Stop"

# -- 1) make sure sshd is running inside WSL (skip if already up) -----
Write-Host "[1/4] checking sshd inside WSL ..." -ForegroundColor Cyan
$sshUp = wsl -u root pgrep -x sshd
if ($sshUp) {
    Write-Host "      sshd already running (pid $sshUp) - skipping start"
} else {
    wsl -u root service ssh start
    if ($LASTEXITCODE -ne 0) {
        wsl -u root /usr/sbin/sshd
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: cannot start sshd inside WSL" -ForegroundColor Red
            exit 1
        }
    }
    Write-Host "      sshd started"
}

# -- 2) read the CURRENT WSL IP --------------------------------------
Write-Host "[2/4] reading WSL IP ..." -ForegroundColor Cyan
$wslIp = (wsl hostname -I).Trim().Split(" ")[0]
if (-not $wslIp) {
    Write-Host "ERROR: no WSL IP - check that 'wsl hostname -I' works on this PC" -ForegroundColor Red
    exit 1
}
Write-Host "      WSL IP = $wslIp"

# -- 3) refresh the port forward (old rule removed first) --------------
Write-Host "[3/4] updating port forwarding (22 -> WSL) ..." -ForegroundColor Cyan
netsh interface portproxy delete v4tov4 listenport=22 listenaddress=0.0.0.0 2>$null | Out-Null
netsh interface portproxy add v4tov4 listenport=22 listenaddress=0.0.0.0 connectport=22 connectaddress=$wslIp | Out-Null

# -- 4) firewall rule (idempotent) ------------------------------------
Write-Host "[4/4] firewall rule ..." -ForegroundColor Cyan
if (-not (Get-NetFirewallRule -DisplayName "SSH-to-WSL" -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -DisplayName "SSH-to-WSL" -Direction Inbound -Protocol TCP -LocalPort 22 -Action Allow | Out-Null
    Write-Host "      firewall rule created"
} else {
    Write-Host "      firewall rule already exists"
}

# -- summary: pick the REAL connected LAN IP ---------------------------
# Only adapters that are actually connected (skip Hyper-V vEthernet,
# WSL, loopback, link-local). Prefer Wi-Fi, then Ethernet.
$lanIp = $null
$adapters = Get-NetAdapter | Where-Object { $_.Status -eq "Up" -and $_.Name -notlike "vEthernet*" }
foreach ($ad in $adapters) {
    $ip = (Get-NetIPAddress -InterfaceIndex $ad.ifIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" -and $_.IPAddress -notlike "172.*" } |
        Select-Object -First 1).IPAddress
    if ($ip) {
        if ($ad.Name -like "*Wi-Fi*" -or -not $lanIp) { $lanIp = $ip }
        if ($ad.Name -like "*Wi-Fi*") { break }
    }
}
Write-Host ""
Write-Host "=============================================" -ForegroundColor Green
if ($lanIp) {
    Write-Host " SSH is ready" -ForegroundColor Green
    Write-Host " From other machines on the LAN use:" -ForegroundColor Green
    Write-Host "   ssh user@$lanIp" -ForegroundColor Yellow
    Write-Host " ('user' = WSL username, '$lanIp' = this PC's current network IP)" -ForegroundColor Green
} else {
    Write-Host " WARNING: this PC has NO connected network adapter" -ForegroundColor Red
    Write-Host " (only Hyper-V/WSL virtual switches are up)" -ForegroundColor Red
    Write-Host " Connect to Wi-Fi or Ethernet first, then run this script again." -ForegroundColor Red
}
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""
Write-Host "NOTE: after any reboot of this PC, run this script again (as Admin)."
