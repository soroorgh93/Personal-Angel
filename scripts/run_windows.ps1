param([string]$Profile = "", [int]$Port = 8600, [string]$OllamaModel = "", [switch]$Web)
# Start PersonalAngel from PowerShell (shows the logs). Desktop mode by default; -Web serves the
# browser UI at http://127.0.0.1:<Port> instead.
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
if (-not $Profile) { $Profile = if (Test-Path "profile.txt") { (Get-Content "profile.txt" -Raw).Trim() } else { "pc_cpu" } }
if ($OllamaModel) { $env:ANGEL_LLM__MODEL = $OllamaModel }
if ($Web) {
    Write-Host "PersonalAngel UI -> http://127.0.0.1:$Port  (profile $Profile)" -ForegroundColor Green
    & .\.venv-win\Scripts\python.exe -m personal_angel serve --profile $Profile --port $Port
} else {
    Write-Host "PersonalAngel desktop app (profile $Profile) - close the window to quit" -ForegroundColor Green
    & .\.venv-win\Scripts\python.exe -m personal_angel desktop --profile $Profile --port $Port
}
