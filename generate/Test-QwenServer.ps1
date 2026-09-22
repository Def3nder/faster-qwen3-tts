#requires -Version 5.1
<#
.SYNOPSIS
Prueft TCP-Verbindung und Qwen-HTTP-Vertrag, ohne Audio zu erzeugen.
.EXAMPLE
.\Test-QwenServer.ps1 -BaseUrl 'http://192.168.1.65:8765'
#>
[CmdletBinding()]
param(
    [string]$BaseUrl = 'http://192.168.1.65:8765',
    [System.Security.SecureString]$Token
)

$ErrorActionPreference = 'Stop'
$uri = [Uri]$BaseUrl
if (-not $uri.IsAbsoluteUri -or $uri.Scheme -notin @('http', 'https') -or
    $uri.UserInfo -or $uri.Query -or $uri.Fragment -or $uri.AbsolutePath -ne '/') {
    throw 'BaseUrl muss eine HTTP(S)-Adresse ohne Pfad oder Zugangsdaten sein.'
}

Write-Host "Pruefe Verbindung zu $($uri.Host):$($uri.Port) ..."
$client = New-Object System.Net.Sockets.TcpClient
try {
    $connection = $client.ConnectAsync($uri.Host, $uri.Port)
    if (-not $connection.Wait(5000)) { throw 'Zeitlimit erreicht.' }
    $connection.GetAwaiter().GetResult()
} catch {
    throw 'Server nicht erreichbar. Adresse, Port, Firewall und Start mit --host 0.0.0.0 pruefen.'
} finally {
    $client.Dispose()
}

if ($null -eq $Token) { $Token = Read-Host 'Qwen-Token (wie auf dem Qwen-Server)' -AsSecureString }
if ($Token.Length -eq 0) { throw 'Ein Token ist erforderlich.' }
$pointer = [IntPtr]::Zero
$plainToken = $null
$headers = $null
try {
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Token)
    $plainToken = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer).Trim()
    $headers = @{ Authorization = "Bearer $plainToken" }
    try {
        $health = Invoke-RestMethod -Uri ($uri.GetLeftPart([UriPartial]::Authority) + '/v1/health') `
            -Headers $headers -Method Get -TimeoutSec 15 -MaximumRedirection 0
    } catch {
        $status = $_.Exception.Response.StatusCode
        if ($null -ne $status) {
            throw "Health-Pruefung: HTTP $([int]$status). Bei 401 Token pruefen, bei 404 Dienst und Port pruefen."
        }
        throw 'Health-Pruefung fehlgeschlagen. Verbindung, TLS und HTTP-Dienst pruefen.'
    }
    if ($health.protocol -ne 'webarchiv-qwen-v2' -or $health.single_chunk -ne $true -or $health.audio_format -ne 'wav') {
        throw 'HTTP-Bruecke und qwen_chunk_worker.py auf dem Modellserver aktualisieren: webarchiv-qwen-v2 wird benoetigt.'
    }
    Write-Host 'Erfolgreich: Qwen-HTTP-Dienst erreichbar und Token akzeptiert.' -ForegroundColor Green
    Write-Host 'Es wurde keine Synthese gestartet. Modell und Stimme sind damit noch nicht getestet.'
    $health
} finally {
    if ($pointer -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer) }
    if ($null -ne $headers) { $headers.Clear() }
    $plainToken = $null
}
