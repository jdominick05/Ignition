# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: AGPL-3.0-or-later
<#
.SYNOPSIS
    Install or update Ignition, ignite-xdna, the NPU compiler toolchain and the model-building tools on Windows 11
    with an AMD Phoenix NPU.

.DESCRIPTION
    Paste into PowerShell:

        irm https://raw.githubusercontent.com/jdominick05/Ignition/main/install.ps1 | iex

    With options:

        & ([scriptblock]::Create((irm https://raw.githubusercontent.com/jdominick05/Ignition/main/install.ps1))) -InstallDriver

    Run it again at any time to update. It
      1. checks for AMD's NPU driver and, with -InstallDriver, installs AMD's production driver,
      2. finds 64-bit CPython 3.13, or installs it for the current user with winget,
      3. finds 64-bit CPython 3.12 for the model tools, or installs it for the current user with winget,
      4. finds Git, or installs it with winget,
      5. installs the XRT SDK 2.21.75 into C:\Xilinx\XRT\xrt_sdk, where ignite-xdna loads pyxrt from,
      6. clones ignite-xdna and Ignition under <InstallRoot>\src, or fast-forwards them,
      7. creates <InstallRoot>\venv (Python 3.13) with mlir-aie 1.4.2, llvm-aie (Peano) and both projects,
      8. creates <InstallRoot>\venv-models (Python 3.12) with AMD Quark, Ultralytics and the Hugging Face CLI,
      9. writes <InstallRoot>\ignition-models.ps1 and ignition-env.ps1, which set up a PowerShell session to build
         models, and to compile and run them,
     10. checks that both environments import, xclbinutil is found and Ignition sees the NPU.
    Steps 3 and 8 are skipped with -SkipModelTools. It writes nothing outside <InstallRoot> and
    C:\Xilinx\XRT\xrt_sdk, apart from what winget and AMD's driver installer do.

.PARAMETER InstallRoot
    Where the sources, the Python environments and downloads go. Default: %LOCALAPPDATA%\Ignition.

.PARAMETER Python
    A 64-bit CPython 3.13 python.exe to build the NPU environment from, instead of searching for or installing one.

.PARAMETER ModelPython
    A 64-bit CPython 3.12 python.exe to build the model-building environment from. AMD Quark 0.11.2 needs Python 3.12
    or older.

.PARAMETER SkipModelTools
    Do not install Python 3.12 or the model-building environment (for running and compiling ready-made models only).

.PARAMETER InstallDriver
    When the NPU driver is missing or older than 32.0.20101.3760, download AMD's production driver package and run
    its installer with --AcceptAmdEula. That accepts AMD's Software End User License Agreement on your behalf, so
    read it first; Windows asks for administrator approval.

.PARAMETER ReinstallXrt
    Replace an existing C:\Xilinx\XRT\xrt_sdk with XRT SDK 2.21.75.

.PARAMETER IgniteXdnaRepo
    Git URL or path to clone ignite-xdna from. Default: its GitHub repository.

.PARAMETER IgniteXdnaBranch
    Branch of ignite-xdna to install. Default: main.

.PARAMETER IgnitionRepo
    Git URL or path to clone Ignition from. Default: its GitHub repository.

.PARAMETER IgnitionBranch
    Branch of Ignition to install. Default: main.
#>
[CmdletBinding()]
param(
    [string]$InstallRoot = (Join-Path $env:LOCALAPPDATA 'Ignition'),
    [string]$Python = '',
    [string]$ModelPython = '',
    [switch]$SkipModelTools,
    [switch]$InstallDriver,
    [switch]$ReinstallXrt,
    [string]$IgniteXdnaRepo = 'https://github.com/jdominick05/ignite-xdna.git',
    [string]$IgniteXdnaBranch = 'main',
    [string]$IgnitionRepo = 'https://github.com/jdominick05/Ignition.git',
    [string]$IgnitionBranch = 'main'
)

# Everything runs inside this script block so that, pasted through iex, no helper or variable is left in the
# caller's session, and failures are thrown rather than exiting (exit would close the window).
& {
    # Native programs are judged by their exit codes. With 'Stop', Windows PowerShell 5.1 turns Git's and pip's
    # progress on stderr into terminating errors whenever the output is redirected, so cmdlets that must succeed
    # carry -ErrorAction Stop instead.
    $ErrorActionPreference = 'Continue'
    $ProgressPreference = 'SilentlyContinue'  # Windows PowerShell 5.1 downloads crawl while drawing progress
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

    # Verified together on a Ryzen 7 8700G (Phoenix NPU), Windows 11 build 26200, 2026-09-15.
    $XrtSdkUrl = 'https://github.com/Xilinx/XRT/releases/download/2.21.75/xrt_windows_sdk.zip'
    $XrtSdkSha256 = 'CCC244C2C423588972ADE76142CDC01049477AAA39A35BE97E782B97EB7C5295'
    $XrtParent = 'C:\Xilinx\XRT'
    $XrtRoot = 'C:\Xilinx\XRT\xrt_sdk\xrt'  # ignite-xdna's runtime/driver.py loads pyxrt from here
    $DriverUrl = 'https://download.amd.com/opendownload/RyzenAI/Driver/NPU_RAI_376_WHQL.zip'
    $DriverSha256 = 'AA836CBFCAD5D0782C79B58F197AA50624AF37E7CB8311C5F94D85B0DC3CCAAD'
    $DriverVersion = [version]'32.0.20101.3760'
    $MlirAie = @('mlir_aie==1.4.2', 'https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.4.2')
    $LlvmAie = @('llvm-aie==22.0.0.2026090201+a36c62b9', 'https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly')
    $Eudsl = @('eudsl-python-extras==0.1.0.20260801.905+68a0d7a', 'https://llvm.github.io/eudsl')
    # Export (Ultralytics), quantization (AMD Quark's XINT8) and the Hugging Face CLI. Quark 0.11.2 only warns when it
    # cannot build its C++ custom ops, which XINT8 does not use; Quark 0.12 stops at import without a C++ compiler.
    $ModelTools = @('amd-quark==0.11.2', 'ultralytics==8.4.153', 'torch==2.14.0', 'onnxruntime==1.30.0', 'huggingface_hub==1.31.0')
    $StepCount = if ($SkipModelTools) { 8 } else { 10 }
    $state = @{ Step = 0 }

    function Write-Step([string]$Text) {
        $state.Step++
        Write-Host ''
        Write-Host ('[{0}/{1}] {2}' -f $state.Step, $StepCount, $Text) -ForegroundColor Cyan
    }

    function Write-Note([string]$Text) { Write-Host "      $Text" }

    function Write-Warn([string]$Text) { Write-Host "      $Text" -ForegroundColor Yellow }

    function Assert-Exit([string]$What) {
        if ($LASTEXITCODE -ne 0) { throw "$What failed with exit code $LASTEXITCODE." }
    }

    function ConvertTo-Quoted([string]$Text) { return "'" + $Text.Replace("'", "''") + "'" }

    function Get-VerifiedDownload([string]$Url, [string]$Sha256, [string]$Destination) {
        if ((Test-Path $Destination) -and ((Get-FileHash -Path $Destination -Algorithm SHA256 -ErrorAction Stop).Hash -eq $Sha256)) {
            Write-Note "using $Destination"
            return
        }
        New-Item -ItemType Directory -Force -Path (Split-Path $Destination) -ErrorAction Stop | Out-Null
        Write-Note "downloading $Url"
        Invoke-WebRequest -Uri $Url -OutFile $Destination -UseBasicParsing -ErrorAction Stop
        $hash = (Get-FileHash -Path $Destination -Algorithm SHA256 -ErrorAction Stop).Hash
        if ($hash -ne $Sha256) {
            Remove-Item -Force $Destination -ErrorAction SilentlyContinue
            throw "$Url downloaded with SHA-256 $hash, expected $Sha256."
        }
    }

    function Get-NpuDriverVersion {
        $device = Get-CimInstance -ClassName Win32_PnPSignedDriver -ErrorAction Stop |
            Where-Object { $_.DeviceName -eq 'NPU Compute Accelerator Device' } | Select-Object -First 1
        if ($device -and $device.DriverVersion) { return [version]$device.DriverVersion }
        return $null
    }

    function Test-Python([string]$Path, [string]$Minor) {
        if (-not $Path -or -not (Test-Path $Path)) { return $false }
        $version = & $Path -c "import struct, sys; print('%d.%d-%d' % (sys.version_info[0], sys.version_info[1], struct.calcsize('P') * 8))"
        return ($LASTEXITCODE -eq 0 -and "$version".Trim() -eq "3.$Minor-64")
    }

    function Find-Python([string]$Explicit, [string]$Minor, [string]$Option) {
        if ($Explicit) {
            if (Test-Python $Explicit $Minor) { return (Resolve-Path $Explicit).Path }
            throw "$Option $Explicit is not a 64-bit CPython 3.$Minor."
        }
        $candidates = @()
        $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
        if ($launcher) {
            foreach ($line in @(& $launcher.Source -0p)) {
                if ("$line" -match "3\.$Minor\S*\s+\*?\s*(\S.*python\.exe)\s*$") { $candidates += $Matches[1] }
            }
        }
        $candidates += (Join-Path $env:LOCALAPPDATA "Programs\Python\Python3$Minor\python.exe")
        $candidates += (Join-Path $env:ProgramFiles "Python3$Minor\python.exe")
        foreach ($candidate in $candidates) {
            if (Test-Python $candidate $Minor) { return $candidate }
        }
        return $null
    }

    function Get-Python([string]$Explicit, [string]$Minor, [string]$Option) {
        $found = Find-Python $Explicit $Minor $Option
        if (-not $found) {
            Install-WithWinget "Python.Python.3.$Minor" @('--scope', 'user')
            $found = Find-Python '' $Minor $Option
            if (-not $found) {
                throw "Python 3.$Minor was not found after winget ran. Install 64-bit Python 3.$Minor from python.org, then run this again with $Option <path to python.exe>."
            }
        }
        Write-Note "using $found"
        return $found
    }

    function Find-Git {
        $command = Get-Command git.exe -ErrorAction SilentlyContinue
        if ($command) { return $command.Source }
        foreach ($path in @((Join-Path $env:ProgramFiles 'Git\cmd\git.exe'), (Join-Path $env:LOCALAPPDATA 'Programs\Git\cmd\git.exe'))) {
            if (Test-Path $path) { return $path }
        }
        return $null
    }

    function Install-WithWinget([string]$Id, [string[]]$Extra) {
        $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
        if (-not $winget) {
            throw "winget is not available to install $Id. Install App Installer from the Microsoft Store, or install $Id yourself, then run this again."
        }
        Write-Note "installing $Id with winget"
        & $winget.Source install --id $Id --exact --silent --accept-package-agreements --accept-source-agreements --disable-interactivity @Extra
        # winget also exits non-zero when the package is already there; the caller looks for the program again.
    }

    function Test-XrtSdk {
        $pyxrt = Get-ChildItem -Path (Join-Path $XrtRoot 'python') -Filter 'pyxrt*.pyd' -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $pyxrt) { return 'no pyxrt' }
        $bytes = [IO.File]::ReadAllBytes($pyxrt.FullName)
        if ([Text.Encoding]::ASCII.GetString($bytes) -notmatch '(?i)python313\.dll') { return 'pyxrt is not built for Python 3.13' }
        if (-not (Test-Path (Join-Path $XrtRoot 'xclbinutil.exe'))) { return 'no xclbinutil.exe' }
        return 'ok'
    }

    function Sync-Repository([string]$Git, [string]$Url, [string]$Branch, [string]$Directory) {
        if (Test-Path (Join-Path $Directory '.git')) {
            Write-Note "updating $Directory"
            & $Git -C $Directory fetch origin $Branch
            Assert-Exit "git fetch in $Directory"
            & $Git -C $Directory checkout $Branch
            Assert-Exit "git checkout $Branch in $Directory"
            & $Git -C $Directory merge --ff-only "origin/$Branch"
            Assert-Exit "Fast-forwarding $Directory (local changes or local commits stop an update)"
        }
        elseif (Test-Path $Directory) {
            throw "$Directory exists but is not a Git checkout. Move it away, or pass another -InstallRoot."
        }
        else {
            Write-Note "cloning $Url"
            & $Git clone --branch $Branch $Url $Directory
            Assert-Exit "git clone $Url"
        }
        $head = & $Git -C $Directory log -1 '--format=%h %s'
        Write-Note "$(Split-Path $Directory -Leaf) at $head"
    }

    function New-Venv([string]$Base, [string]$Venv, [string]$Minor) {
        $venvPython = Join-Path $Venv 'Scripts\python.exe'
        if ((Test-Path $venvPython) -and -not (Test-Python $venvPython $Minor)) {
            Write-Note "recreating $Venv, which is not Python 3.$Minor"
            Remove-Item -Recurse -Force $Venv -ErrorAction Stop
        }
        if (-not (Test-Path $venvPython)) {
            & $Base -m venv $Venv
            Assert-Exit "Creating $Venv"
        }
        return $venvPython
    }

    function Repair-LlvmAie([string]$Venv) {
        # The published Windows llvm-aie wheel needs what mlir-aie's utils/iron_setup.py does after installing it:
        # GNU-style libc.a and libm.a names, and no .deplibs section in crt1.o. Both steps are idempotent.
        $peano = Join-Path $Venv 'Lib\site-packages\llvm-aie'
        $objcopy = Join-Path $peano 'bin\llvm-objcopy.exe'
        if (-not (Test-Path $objcopy)) { $objcopy = Join-Path $Venv 'Lib\site-packages\mlir_aie\bin\llvm-objcopy.exe' }
        $toolchains = @(Get-ChildItem -Path (Join-Path $peano 'lib') -Directory -Filter '*-none-unknown-elf' -ErrorAction SilentlyContinue)
        if ($toolchains.Count -eq 0) { throw "No llvm-aie toolchains under $peano\lib." }
        foreach ($toolchain in $toolchains) {
            foreach ($pair in @(@('c.lib', 'libc.a'), @('m.lib', 'libm.a'))) {
                $from = Join-Path $toolchain.FullName $pair[0]
                $to = Join-Path $toolchain.FullName $pair[1]
                if ((Test-Path $from) -and -not (Test-Path $to)) { Copy-Item -Path $from -Destination $to -ErrorAction Stop }
            }
            $crt1 = Join-Path $toolchain.FullName 'crt1.o'
            if (Test-Path $crt1) {
                & $objcopy --remove-section=.deplibs $crt1
                Assert-Exit "Preparing $crt1"
            }
        }
        Write-Note "prepared $($toolchains.Count) llvm-aie toolchains"
    }

    function Write-SessionScript([string]$Path, [string]$Purpose, [string]$Venv, [string]$SessionPath, [string]$Location) {
        $lines = @(
            '# Written by Ignition''s install.ps1, which rewrites it on every run.',
            "# Sets up this PowerShell session to $Purpose.",
            ('$env:IGNITION_HOME = ' + (ConvertTo-Quoted $InstallRoot)),
            ('$env:VIRTUAL_ENV = ' + (ConvertTo-Quoted $Venv)),
            ('$env:PATH = ' + (ConvertTo-Quoted "$SessionPath;") + ' + $env:PATH'),
            ('Set-Location ' + (ConvertTo-Quoted $Location)),
            ('Write-Host "Ready to ' + $Purpose + ' in $(Get-Location)"')
        )
        Set-Content -Path $Path -Value $lines -Encoding UTF8 -ErrorAction Stop
        Write-Note "wrote $Path"
    }

    Write-Host "Ignition installer -> $InstallRoot" -ForegroundColor Green
    $build = [Environment]::OSVersion.Version.Build
    if ($build -lt 22621) { throw "Windows 11 (build 22621 or newer) is required; this is build $build." }
    if (-not [Environment]::Is64BitOperatingSystem) { throw '64-bit Windows is required.' }
    New-Item -ItemType Directory -Force -Path $InstallRoot -ErrorAction Stop | Out-Null
    $downloads = Join-Path $InstallRoot 'downloads'

    Write-Step 'AMD NPU driver'
    $driver = Get-NpuDriverVersion
    if ($driver -and $driver -ge $DriverVersion) {
        Write-Note "NPU driver $driver"
    }
    elseif ($InstallDriver) {
        $zip = Join-Path $downloads 'NPU_RAI_376_WHQL.zip'
        Get-VerifiedDownload $DriverUrl $DriverSha256 $zip
        $unpacked = Join-Path $downloads 'NPU_RAI_376_WHQL'
        if (Test-Path $unpacked) { Remove-Item -Recurse -Force $unpacked -ErrorAction Stop }
        Expand-Archive -Path $zip -DestinationPath $unpacked -ErrorAction Stop
        Write-Warn "Running AMD's npu_sw_installer.exe --AcceptAmdEula, which accepts AMD's Software End User License Agreement."
        Write-Warn 'Windows asks for administrator approval.'
        $process = Start-Process -FilePath (Join-Path $unpacked 'npu_sw_installer.exe') -ArgumentList '--AcceptAmdEula' -WorkingDirectory $unpacked -Verb RunAs -Wait -PassThru -ErrorAction Stop
        $driver = Get-NpuDriverVersion
        if ($driver -and $driver -ge $DriverVersion) { Write-Note "NPU driver $driver" }
        else { Write-Warn "The driver installer exited with code $($process.ExitCode) and the NPU driver reads '$driver'. Restart Windows, then run this again." }
    }
    else {
        if ($driver) { Write-Warn "NPU driver $driver is older than $DriverVersion, the version Ignition was verified with." }
        else { Write-Warn "No AMD NPU driver found (no 'NPU Compute Accelerator Device')." }
        Write-Warn 'Compiling works without it; running on the NPU does not. Install AMD''s production driver from'
        Write-Warn "$DriverUrl, or run this installer again with -InstallDriver."
    }

    Write-Step 'Python 3.13'
    $pythonExe = Get-Python $Python '13' '-Python'

    if (-not $SkipModelTools) {
        Write-Step 'Python 3.12 for the model tools'
        $modelBase = Get-Python $ModelPython '12' '-ModelPython'
    }

    Write-Step 'Git'
    $git = Find-Git
    if (-not $git) {
        Install-WithWinget 'Git.Git' @()
        $git = Find-Git
        if (-not $git) { throw 'Git was not found after winget ran. Install Git for Windows, then run this again.' }
    }
    Write-Note "using $git"

    Write-Step 'XRT SDK 2.21.75'
    $xrt = Test-XrtSdk
    if ($xrt -eq 'ok' -and -not $ReinstallXrt) {
        Write-Note "found $XrtRoot"
    }
    else {
        if ($xrt -ne 'no pyxrt' -and -not $ReinstallXrt) {
            throw "$XrtRoot is there but unusable ($xrt). Run this again with -ReinstallXrt to replace C:\Xilinx\XRT\xrt_sdk."
        }
        $zip = Join-Path $downloads 'xrt_windows_sdk.zip'
        Get-VerifiedDownload $XrtSdkUrl $XrtSdkSha256 $zip
        $sdk = Join-Path $XrtParent 'xrt_sdk'
        try {
            if (Test-Path $sdk) { Remove-Item -Recurse -Force $sdk -ErrorAction Stop }
            New-Item -ItemType Directory -Force -Path $XrtParent -ErrorAction Stop | Out-Null
        }
        catch {
            throw "Cannot write $XrtParent ($($_.Exception.Message)). Run this once from an administrator PowerShell."
        }
        & tar.exe -xf $zip -C $XrtParent
        Assert-Exit 'Unpacking the XRT SDK'
        $xrt = Test-XrtSdk
        if ($xrt -ne 'ok') { throw "The XRT SDK did not unpack as expected into $XrtRoot ($xrt)." }
        Write-Note "installed $XrtRoot"
    }

    Write-Step 'ignite-xdna and Ignition'
    $sources = Join-Path $InstallRoot 'src'
    New-Item -ItemType Directory -Force -Path $sources -ErrorAction Stop | Out-Null
    $igniteXdna = Join-Path $sources 'ignite-xdna'
    $ignition = Join-Path $sources 'Ignition'
    Sync-Repository $git $IgniteXdnaRepo $IgniteXdnaBranch $igniteXdna
    Sync-Repository $git $IgnitionRepo $IgnitionBranch $ignition

    $pip = @('-m', 'pip', 'install', '--disable-pip-version-check')

    Write-Step 'Python 3.13 environment and NPU compiler toolchain'
    $venv = Join-Path $InstallRoot 'venv'
    $venvPython = New-Venv $pythonExe $venv '13'
    & $venvPython @pip --upgrade pip
    Assert-Exit 'Upgrading pip'
    & $venvPython @pip aiofiles rich 'ml_dtypes>=0.5.4' cloudpickle 'numpy>=2.5.1,<3.0'
    Assert-Exit "Installing mlir-aie's Python requirements"
    & $venvPython @pip $Eudsl[0] -f $Eudsl[1] '--config-settings=EUDSL_PYTHON_EXTRAS_HOST_PACKAGE_PREFIX=aie'
    Assert-Exit 'Installing eudsl-python-extras'
    & $venvPython @pip $MlirAie[0] -f $MlirAie[1]
    Assert-Exit 'Installing mlir-aie'
    & $venvPython @pip $LlvmAie[0] -f $LlvmAie[1]
    Assert-Exit 'Installing llvm-aie (Peano)'
    Repair-LlvmAie $venv
    & $venvPython @pip -e $igniteXdna -e $ignition
    Assert-Exit 'Installing ignite-xdna and Ignition'

    $modelVenv = Join-Path $InstallRoot 'venv-models'
    if (-not $SkipModelTools) {
        Write-Step 'Python 3.12 environment for exporting and quantizing models'
        $modelPythonExe = New-Venv $modelBase $modelVenv '12'
        & $modelPythonExe @pip --upgrade pip
        Assert-Exit 'Upgrading pip in the model environment'
        & $modelPythonExe @pip @ModelTools
        Assert-Exit 'Installing AMD Quark, Ultralytics and the Hugging Face CLI'
    }

    Write-Step 'Session setup scripts'
    # The NPU session: the Python 3.13 environment's scripts, and the XRT SDK root for xclbinutil, which aiecc runs to
    # write the xclbin. ignite-xdna finds pyxrt and the XRT DLLs itself, so running a container needs nothing more.
    $sessionPath = "$venv\Scripts;$XrtRoot"
    $envScript = Join-Path $InstallRoot 'ignition-env.ps1'
    Write-SessionScript $envScript 'compile and run models with Ignition' $venv $sessionPath $ignition
    # The model session: Python 3.12 with Quark (which also needs its ninja.exe on PATH), in ignite-xdna, whose
    # pipeline scripts read and write models\ and data\ under the checkout.
    $modelsScript = Join-Path $InstallRoot 'ignition-models.ps1'
    if (-not $SkipModelTools) {
        Write-SessionScript $modelsScript 'download, export and quantize models' $modelVenv "$modelVenv\Scripts" $igniteXdna
    }

    Write-Step 'Checks'
    $savedPath = $env:PATH
    $npuSeen = $false
    try {
        $env:PATH = "$sessionPath;$env:PATH"
        & $venvPython -c "import aie.iron, ignite_xdna, ignition; print('      aie.iron, ignite_xdna and ignition import')"
        Assert-Exit 'Importing the toolchain'
        $xclbinutil = Get-Command xclbinutil.exe -ErrorAction SilentlyContinue
        if (-not $xclbinutil) { throw 'xclbinutil.exe is not on the session PATH, so ignite-compile cannot write an xclbin.' }
        Write-Note "xclbinutil: $($xclbinutil.Source)"
        $devices = @(& (Join-Path $venv 'Scripts\ignition.exe') devices)
        $devicesExit = $LASTEXITCODE
        foreach ($line in $devices) { Write-Note "$line" }
        $npuSeen = ($devicesExit -eq 0) -and (($devices -join "`n") -match 'NPU Phoenix')
        if (-not $SkipModelTools) {
            & $modelPythonExe -c "import quark, ultralytics, onnxruntime, huggingface_hub; print('      quark', quark.__version__, 'and ultralytics', ultralytics.__version__, 'import')"
            Assert-Exit 'Importing the model tools'
        }
    }
    finally {
        $env:PATH = $savedPath
    }
    if (-not $npuSeen) { Write-Warn 'Ignition did not list a Phoenix NPU: check the driver step above. Models still compile.' }

    Write-Host ''
    if ($SkipModelTools) {
        Write-Host 'Ignition is installed. Follow the README from "Compile and run". Each PowerShell window starts with:' -ForegroundColor Green
    } else {
        Write-Host 'Ignition is installed. Follow the README from "Get a model". Each PowerShell window starts with one of:' -ForegroundColor Green
        Write-Host '  to download, export and quantize a model:'
        Write-Host "    Set-ExecutionPolicy Bypass -Scope Process -Force; . $(ConvertTo-Quoted $modelsScript)"
    }
    Write-Host '  to compile and run a model:'
    Write-Host "    Set-ExecutionPolicy Bypass -Scope Process -Force; . $(ConvertTo-Quoted $envScript)"
}
