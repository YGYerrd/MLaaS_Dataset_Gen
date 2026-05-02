param(
    [string]$VenvPath = ".venv",
    [switch]$SkipTensorFlow
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvFullPath = Join-Path $repoRoot $VenvPath
$requirementsPath = Join-Path $repoRoot "requirements.txt"
$pythonLauncher = "py"

$rocmSdkArtifacts = @(
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_core-7.2.0.dev0-py3-none-win_amd64.whl",
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_devel-7.2.0.dev0-py3-none-win_amd64.whl",
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_libraries_custom-7.2.0.dev0-py3-none-win_amd64.whl",
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm-7.2.0.dev0.tar.gz"
)

$rocmTorchArtifacts = @(
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torch-2.9.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl",
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torchaudio-2.9.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl",
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torchvision-0.24.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl"
)

Write-Host "Creating Python 3.12 virtual environment at $venvFullPath"
& $pythonLauncher -3.12 -m venv $venvFullPath

$venvPython = Join-Path $venvFullPath "Scripts\python.exe"
$activationHelper = Join-Path $repoRoot "scripts\Activate-ROCm-Venv.ps1"

Write-Host "Upgrading pip/setuptools/wheel"
& $venvPython -m pip install --upgrade pip setuptools wheel

Write-Host "Installing AMD ROCm SDK packages"
& $venvPython -m pip install --no-cache-dir $rocmSdkArtifacts

Write-Host "Installing AMD ROCm PyTorch wheels"
& $venvPython -m pip install --no-cache-dir $rocmTorchArtifacts

$excludedPackages = @("torch", "torchvision", "torchaudio")
if ($SkipTensorFlow) {
    $excludedPackages += @("tensorflow", "keras")
}

$filteredRequirements = New-TemporaryFile
try {
    Get-Content $requirementsPath |
        Where-Object {
            $line = $_.Trim()
            if (-not $line -or $line.StartsWith("#")) {
                return $true
            }
            $packageName = ($line -split '[<>=!~\[]', 2)[0].Trim().ToLowerInvariant()
            return $packageName -notin $excludedPackages
        } |
        Set-Content $filteredRequirements.FullName

    Write-Host "Installing remaining project requirements"
    & $venvPython -m pip install -r $filteredRequirements.FullName
}
finally {
    Remove-Item $filteredRequirements.FullName -ErrorAction SilentlyContinue
}

@"
`$env:ROCM_SDK_TARGET_FAMILY = "custom"
. "$venvFullPath\Scripts\Activate.ps1"
"@ | Set-Content $activationHelper

Write-Host "Verifying ROCm-backed PyTorch device visibility"
$env:ROCM_SDK_TARGET_FAMILY = "custom"
& $venvPython -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
