# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

<#
.SYNOPSIS
Set up a Windows ARM64 runner to build scikit-plots wheels: LLVM (clang-cl
and flang), the Visual Studio C++ tools, and pkg-config.

.DESCRIPTION
Dot-source this file, then call one function per step:

    . "$env:GITHUB_ACTION_PATH/windows_arm64.ps1"
    Install-Llvm -Version 20.1.8 -Sha256 <hash>
    Initialize-Msvc -Mode detect
    Install-PkgConf

Each function stops with a message that says what was looked for, where,
and what to change. Nothing is guessed: Visual Studio is located with
vswhere, vcpkg through the variable the runner image defines.

.NOTES
USER NOTES
  See README.md in this directory for the inputs, the sources of every
  path used here, and what to do when a step fails.

DEVELOPER NOTES
  * No path to Visual Studio is written in this file. The path contains the
    release ("2022", then "18" for Visual Studio 2026) and the edition, and
    GitHub replaces the image behind a runner label without notice to the
    workflow. That is what broke this action in September 2026.
  * Everything that touches the machine goes through three small functions
    (Invoke-Native, Invoke-Download, Start-Installer). The tests replace
    them, so the logic runs on any system that has PowerShell 7.
  * Values reach later steps only through the files GitHub names in
    GITHUB_ENV, GITHUB_PATH and GITHUB_OUTPUT. A variable set in this
    process ends with the step.
  * A batch file cannot change the environment of the PowerShell that calls
    it. `& vcvarsall.bat arm64` therefore changes nothing; see
    Get-VcVarsEnvironment for the way that does.

.LINK
https://github.com/microsoft/vswhere/wiki/Find-VC
.LINK
https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line
.LINK
https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-commands
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# The component whose presence means "this Visual Studio can build ARM64 C++".
# https://learn.microsoft.com/en-us/visualstudio/install/workload-component-id-vs-enterprise
$script:VcToolsArm64Component = 'Microsoft.VisualStudio.Component.VC.Tools.ARM64'


# ---------------------------------------------------------------------------
# The three doors to the machine. Tests replace these.
# ---------------------------------------------------------------------------

function Invoke-Native {
    <#
    .SYNOPSIS
    Run a program and return its exit code and output; never throws on a
    non-zero exit code.
    .OUTPUTS
    PSCustomObject with ExitCode (int) and Output (string, stdout and stderr).
    #>
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$ArgumentList = @()
    )
    $text = & $FilePath @ArgumentList 2>&1 | Out-String
    return [pscustomobject]@{ ExitCode = $LASTEXITCODE; Output = $text }
}

function Invoke-Download {
    <#
    .SYNOPSIS
    Download a file over HTTPS, trying again after a network error.
    #>
    param(
        [Parameter(Mandatory)][string]$Uri,
        [Parameter(Mandatory)][string]$OutFile,
        [ValidateRange(1, 10)][int]$Attempts = 4,
        [ValidateRange(0, 120)][int]$DelaySeconds = 10
    )
    if ($Uri -notmatch '^https://') {
        throw "Refusing to download over anything but https: $Uri"
    }
    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        try {
            Invoke-WebRequest -Uri $Uri -OutFile $OutFile -UseBasicParsing
            return
        }
        catch {
            if ($attempt -eq $Attempts) {
                throw "Download failed after $Attempts attempts: $Uri`n$($_.Exception.Message)"
            }
            Write-Warning "Download attempt $attempt of $Attempts failed: $($_.Exception.Message)"
            Start-Sleep -Seconds ($DelaySeconds * $attempt)
        }
    }
}

function Start-Installer {
    <#
    .SYNOPSIS
    Run a silent installer, wait for it, and return its exit code.
    #>
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string[]]$ArgumentList
    )
    $process = Start-Process -FilePath $FilePath -ArgumentList $ArgumentList -Wait -PassThru
    return $process.ExitCode
}


# ---------------------------------------------------------------------------
# Passing values to later steps
# ---------------------------------------------------------------------------

function Get-GitHubFile {
    <#
    .SYNOPSIS
    Return the path GitHub Actions gave for GITHUB_ENV, GITHUB_PATH or
    GITHUB_OUTPUT; stop if this is not a workflow run.
    #>
    param([Parameter(Mandatory)][ValidateSet('GITHUB_ENV', 'GITHUB_PATH', 'GITHUB_OUTPUT')][string]$Variable)
    $path = [Environment]::GetEnvironmentVariable($Variable)
    if ([string]::IsNullOrWhiteSpace($path)) {
        throw "$Variable is not set: this script passes values to later steps and must run inside GitHub Actions."
    }
    return $path
}

function Assert-SingleLine {
    param([string]$What, [AllowEmptyString()][string]$Value)
    if ($Value -match "[`r`n]") {
        throw "$What contains a line break and cannot be passed to later steps: $($Value.Split("`n")[0])..."
    }
}

function Add-GitHubEnv {
    <#
    .SYNOPSIS
    Define an environment variable for the steps that follow this one.
    #>
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][AllowEmptyString()][string]$Value
    )
    if ($Name -notmatch '^[A-Za-z_][A-Za-z0-9_]*$') {
        throw "Not a valid environment variable name: '$Name'"
    }
    Assert-SingleLine -What "The value of $Name" -Value $Value
    Add-Content -LiteralPath (Get-GitHubFile GITHUB_ENV) -Value "$Name=$Value" -Encoding utf8
}

function Add-GitHubPath {
    <#
    .SYNOPSIS
    Put a directory at the front of PATH for the steps that follow this one.
    .NOTES
    GitHub puts each new entry before the earlier ones. To keep several
    directories in a given order, add the last one first.
    #>
    param([Parameter(Mandatory)][string]$Directory)
    Assert-SingleLine -What 'A PATH entry' -Value $Directory
    Add-Content -LiteralPath (Get-GitHubFile GITHUB_PATH) -Value $Directory -Encoding utf8
}

function Add-GitHubOutput {
    <#
    .SYNOPSIS
    Set an output of the current step.
    #>
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][AllowEmptyString()][string]$Value
    )
    Assert-SingleLine -What "The output $Name" -Value $Value
    Add-Content -LiteralPath (Get-GitHubFile GITHUB_OUTPUT) -Value "$Name=$Value" -Encoding utf8
}


# ---------------------------------------------------------------------------
# Step 1: LLVM
# ---------------------------------------------------------------------------

function Assert-FileSha256 {
    <#
    .SYNOPSIS
    Stop unless a file has the expected SHA-256.
    #>
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Expected
    )
    if ($Expected -notmatch '^[0-9A-Fa-f]{64}$') {
        throw "Not a SHA-256 (64 hexadecimal digits): '$Expected'"
    }
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
    if ($actual -ine $Expected) {
        throw ("Checksum mismatch for $Path`n  expected: $($Expected.ToLowerInvariant())`n  actual:   $($actual.ToLowerInvariant())`n" +
            'The download is corrupt, or the version and the checksum given to this action do not belong together.')
    }
}

function Install-Llvm {
    <#
    .SYNOPSIS
    Install the official LLVM release for Windows on ARM64 and select
    clang-cl and flang as the compilers of the following steps.

    .PARAMETER Version
    The LLVM release, for example 20.1.8.

    .PARAMETER Sha256
    SHA-256 of LLVM-<Version>-woa64.exe, as published with the release.

    .PARAMETER InstallDirectory
    Where the installer puts LLVM: its own default, which is not changed
    here. Give another directory only to look for an LLVM installed there.

    .PARAMETER WorkDirectory
    Where the installer is downloaded to.

    .NOTES
    DEVELOPER NOTES
      * clang-cl is clang with the command line and the ABI of MSVC, which is
        what CPython on Windows is built with. It needs the headers and
        libraries of Visual Studio; see Initialize-Msvc.
      * The Fortran compiler was `flang-new` until LLVM 19 and is `flang`
        from LLVM 20, which still ships the old name. Whichever exists is
        used, the old name first, so a newer LLVM needs no change here.
      * The standard library of Visual Studio refuses a clang older than it
        supports (error STL1000). After a Visual Studio upgrade on the
        runner image, raise Version and Sha256 together.

    .LINK
    https://github.com/llvm/llvm-project/releases
    .LINK
    https://clang.llvm.org/docs/MSVCCompatibility.html
    .LINK
    https://flang.llvm.org/docs/FlangDriver.html
    #>
    param(
        [Parameter(Mandatory)][string]$Version,
        [Parameter(Mandatory)][string]$Sha256,
        [string]$InstallDirectory = (Join-Path $env:ProgramFiles 'LLVM'),
        [string]$WorkDirectory = $(if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { [IO.Path]::GetTempPath() })
    )
    if ($Version -notmatch '^\d+\.\d+\.\d+$') {
        throw "Not an LLVM release number (expected for example 20.1.8): '$Version'"
    }
    $uri = "https://github.com/llvm/llvm-project/releases/download/llvmorg-$Version/LLVM-$Version-woa64.exe"
    $installer = Join-Path $WorkDirectory "LLVM-$Version-woa64.exe"

    Write-Host "Downloading $uri"
    Invoke-Download -Uri $uri -OutFile $installer
    Assert-FileSha256 -Path $installer -Expected $Sha256

    Write-Host "Installing LLVM $Version (expected in $InstallDirectory)"
    # /S: the silent switch of the NSIS installer LLVM ships for Windows.
    $exitCode = Start-Installer -FilePath $installer -ArgumentList @('/S')
    if ($exitCode -ne 0) {
        throw "The LLVM installer ended with exit code $exitCode."
    }

    $bin = Join-Path $InstallDirectory 'bin'
    $clang = Join-Path $bin 'clang-cl.exe'
    if (-not (Test-Path -LiteralPath $clang -PathType Leaf)) {
        throw "clang-cl.exe is not in $bin after the installation."
    }
    $fortran = @('flang-new', 'flang') |
        Where-Object { Test-Path -LiteralPath (Join-Path $bin "$_.exe") -PathType Leaf } |
        Select-Object -First 1
    if (-not $fortran) {
        throw "Neither flang-new.exe nor flang.exe is in $bin; LLVM $Version for Windows on ARM64 has no Fortran compiler."
    }
    foreach ($tool in @($clang, (Join-Path $bin "$fortran.exe"))) {
        $probe = Invoke-Native -FilePath $tool -ArgumentList @('--version')
        if ($probe.ExitCode -ne 0) {
            throw "$tool does not run (exit code $($probe.ExitCode)):`n$($probe.Output)"
        }
        Write-Host ($probe.Output.Trim().Split("`n")[0])
    }

    Add-GitHubPath -Directory $bin
    Add-GitHubEnv -Name 'CC' -Value 'clang-cl'
    Add-GitHubEnv -Name 'CXX' -Value 'clang-cl'
    Add-GitHubEnv -Name 'FC' -Value $fortran
    Add-GitHubEnv -Name 'TARGET_ARCH' -Value 'ARM64'
    Add-GitHubOutput -Name 'llvm-bin' -Value $bin
    Add-GitHubOutput -Name 'fortran-compiler' -Value $fortran
}


# ---------------------------------------------------------------------------
# Step 2: Visual Studio
# ---------------------------------------------------------------------------

function Resolve-VsWhere {
    <#
    .SYNOPSIS
    Return the path of vswhere.exe.

    .NOTES
    The Visual Studio installer has put vswhere at this fixed location since
    Visual Studio 2017 version 15.2, so that it can be found without knowing
    which Visual Studio is installed.

    .LINK
    https://github.com/microsoft/vswhere/wiki/Installing
    #>
    param([string[]]$Candidates)
    if (-not $Candidates) {
        $Candidates = @(${env:ProgramFiles(x86)}, $env:ProgramFiles) |
            Where-Object { $_ } |
            ForEach-Object { Join-Path $_ 'Microsoft Visual Studio' 'Installer' 'vswhere.exe' }
    }
    foreach ($candidate in $Candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }
    $onPath = Get-Command -Name 'vswhere' -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($onPath) {
        return $onPath.Source
    }
    throw ("vswhere.exe was not found. Looked in:`n  " + ($Candidates -join "`n  ") + "`n  and on PATH.`n" +
        'It is installed with Visual Studio 2017 or later; this runner has no Visual Studio, or an image that this action does not know.')
}

function ConvertFrom-VsWhereJson {
    <#
    .SYNOPSIS
    Return the installations in the JSON that vswhere printed; none for `[]`.
    #>
    param([AllowEmptyString()][string]$Text)
    if ([string]::IsNullOrWhiteSpace($Text)) {
        return @()
    }
    try {
        $parsed = $Text | ConvertFrom-Json
    }
    catch {
        throw "vswhere did not print JSON:`n$Text"
    }
    return @($parsed | Where-Object { $null -ne $_ })
}

function Get-InstanceProperty {
    <#
    .SYNOPSIS
    Return a property of a vswhere installation as text, or an empty string.
    #>
    param($Instance, [string]$Name)
    $property = $Instance.PSObject.Properties[$Name]
    if ($null -eq $property -or $null -eq $property.Value) {
        return ''
    }
    return [string]$property.Value
}

function Find-VisualStudio {
    <#
    .SYNOPSIS
    Find the newest Visual Studio that has the ARM64 C++ build tools.

    .PARAMETER VsWhere
    Path of vswhere.exe; found by Resolve-VsWhere when omitted.

    .PARAMETER Component
    The Visual Studio component that must be installed.

    .OUTPUTS
    PSCustomObject with InstallationPath, Version, DisplayName, VcVarsAll
    and ToolsVersion (empty if Visual Studio does not state a default).

    .NOTES
    vswhere is asked for JSON, not for one property, so that an installation
    that lacks the component can be told from no installation at all.

    .LINK
    https://github.com/microsoft/vswhere/wiki/Find-VC
    #>
    param(
        [string]$VsWhere,
        [string]$Component = $script:VcToolsArm64Component
    )
    if (-not $VsWhere) {
        $VsWhere = Resolve-VsWhere
    }
    $common = @('-products', '*', '-format', 'json', '-utf8', '-nologo')
    $found = Invoke-Native -FilePath $VsWhere -ArgumentList (@('-latest', '-requires', $Component) + $common)
    if ($found.ExitCode -ne 0) {
        throw "vswhere failed (exit code $($found.ExitCode)):`n$($found.Output)"
    }
    $instances = @(ConvertFrom-VsWhereJson -Text $found.Output)
    if ($instances.Count -eq 0) {
        $all = Invoke-Native -FilePath $VsWhere -ArgumentList (@('-all') + $common)
        $names = @()
        if ($all.ExitCode -eq 0) {
            $names = @(ConvertFrom-VsWhereJson -Text $all.Output) | ForEach-Object {
                "$(Get-InstanceProperty $_ 'displayName') $(Get-InstanceProperty $_ 'installationVersion') at $(Get-InstanceProperty $_ 'installationPath')"
            }
        }
        $installed = if ($names) { "Installed:`n  " + ($names -join "`n  ") } else { 'No Visual Studio is installed.' }
        throw "No Visual Studio installation has the component $Component (MSVC build tools for ARM64).`n$installed"
    }
    $instance = $instances[0]
    $root = Get-InstanceProperty $instance 'installationPath'
    if (-not $root) {
        throw "vswhere returned an installation without installationPath:`n$($found.Output)"
    }
    $vcvarsall = Join-Path $root 'VC' 'Auxiliary' 'Build' 'vcvarsall.bat'
    if (-not (Test-Path -LiteralPath $vcvarsall -PathType Leaf)) {
        throw "Visual Studio at $root reports the C++ tools but has no VC\Auxiliary\Build\vcvarsall.bat."
    }
    $toolsVersion = ''
    $default = Join-Path $root 'VC' 'Auxiliary' 'Build' 'Microsoft.VCToolsVersion.default.txt'
    if (Test-Path -LiteralPath $default -PathType Leaf) {
        $toolsVersion = (Get-Content -LiteralPath $default -Raw).Trim()
    }
    return [pscustomobject]@{
        InstallationPath = $root
        Version          = Get-InstanceProperty $instance 'installationVersion'
        DisplayName      = Get-InstanceProperty $instance 'displayName'
        VcVarsAll        = $vcvarsall
        ToolsVersion     = $toolsVersion
    }
}

function ConvertFrom-SetOutput {
    <#
    .SYNOPSIS
    Turn the output of the `set` command of cmd.exe into a dictionary.

    .NOTES
    Lines without `NAME=`, and the hidden `=C:` variables cmd.exe keeps for
    the current directory of each drive, are not variables and are skipped.
    #>
    param([Parameter(Mandatory)][AllowEmptyString()][string]$Text)
    $environment = [ordered]@{}
    foreach ($line in ($Text -split "`r?`n")) {
        $at = $line.IndexOf('=')
        if ($at -lt 1) {
            continue
        }
        $environment[$line.Substring(0, $at)] = $line.Substring($at + 1)
    }
    return $environment
}

function Get-VcVarsEnvironment {
    <#
    .SYNOPSIS
    Return the environment that vcvarsall.bat sets up for an architecture.

    .NOTES
    DEVELOPER NOTES
      vcvarsall.bat is a batch file. It can only change the environment of
      the cmd.exe that runs it, and that cmd.exe ends when the batch file
      does. So it is called from a small command file that prints the
      resulting environment with `set`, and the caller reads that back.
      The command file avoids quoting a path with spaces on a command line.

    .LINK
    https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line#vcvarsall-syntax
    #>
    param(
        [Parameter(Mandatory)][string]$VcVarsAll,
        [ValidateSet('arm64', 'x64', 'x86', 'x64_arm64', 'arm64_x64')][string]$Architecture = 'arm64',
        [string]$WorkDirectory = $(if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { [IO.Path]::GetTempPath() })
    )
    $commandFile = Join-Path $WorkDirectory "vcvars-$Architecture-$PID.cmd"
    $lines = @(
        '@echo off',
        "call `"$VcVarsAll`" $Architecture",
        'if errorlevel 1 exit /b 1',
        'echo ===ENVIRONMENT===',
        'set'
    )
    Set-Content -LiteralPath $commandFile -Value $lines -Encoding ascii
    try {
        $shell = if ($env:ComSpec) { $env:ComSpec } else { 'cmd.exe' }
        $result = Invoke-Native -FilePath $shell -ArgumentList @('/d', '/c', $commandFile)
    }
    finally {
        Remove-Item -LiteralPath $commandFile -Force -ErrorAction SilentlyContinue
    }
    $marker = $result.Output.IndexOf('===ENVIRONMENT===')
    if ($result.ExitCode -ne 0 -or $marker -lt 0) {
        throw "vcvarsall.bat $Architecture failed (exit code $($result.ExitCode)):`n$($result.Output)"
    }
    $environment = ConvertFrom-SetOutput -Text $result.Output.Substring($marker)
    if (-not $environment.Contains('VCToolsInstallDir')) {
        throw "vcvarsall.bat $Architecture ran but did not set VCToolsInstallDir; the C++ tools for $Architecture are not usable.`n$($result.Output.Substring(0, $marker))"
    }
    return $environment
}

function Export-EnvironmentChange {
    <#
    .SYNOPSIS
    Pass to later steps every variable that differs between two environments.

    .PARAMETER Before
    The environment as it is now.

    .PARAMETER After
    The environment to reach.

    .OUTPUTS
    PSCustomObject with Variables (names written) and PathEntries (new PATH
    entries, in the order they will have at the front of PATH).

    .NOTES
    PATH is not written as a variable: its new entries go to GITHUB_PATH, so
    entries added by earlier and later steps keep their place.
    #>
    param(
        [Parameter(Mandatory)][System.Collections.IDictionary]$Before,
        [Parameter(Mandatory)][System.Collections.IDictionary]$After
    )
    $written = @()
    foreach ($name in $After.Keys) {
        if ($name -ieq 'PATH' -or $name -notmatch '^[A-Za-z_][A-Za-z0-9_]*$') {
            continue
        }
        if (-not $Before.Contains($name) -or [string]$Before[$name] -cne [string]$After[$name]) {
            Add-GitHubEnv -Name $name -Value ([string]$After[$name])
            $written += $name
        }
    }
    $normalize = { param($entry) $entry.Trim().TrimEnd('\', '/').ToLowerInvariant() }
    $known = @{}
    if ($Before.Contains('PATH')) {
        foreach ($entry in ([string]$Before['PATH'] -split ';')) {
            if ($entry.Trim()) { $known[(& $normalize $entry)] = $true }
        }
    }
    $added = @()
    if ($After.Contains('PATH')) {
        foreach ($entry in ([string]$After['PATH'] -split ';')) {
            if (-not $entry.Trim()) { continue }
            $key = & $normalize $entry
            if (-not $known.ContainsKey($key)) {
                $known[$key] = $true
                $added += $entry.Trim()
            }
        }
    }
    # GitHub puts each entry before the earlier ones: last written, first on PATH.
    for ($index = $added.Count - 1; $index -ge 0; $index--) {
        Add-GitHubPath -Directory $added[$index]
    }
    return [pscustomobject]@{ Variables = $written; PathEntries = $added }
}

function Initialize-Msvc {
    <#
    .SYNOPSIS
    Locate the Visual Studio C++ tools for ARM64 and, on request, activate
    them for the following steps.

    .PARAMETER Mode
    detect  Find Visual Studio, check it, report it. The environment of the
            following steps is not changed; clang-cl and vcpkg find Visual
            Studio by themselves. This is the default and is what every
            build that passed before October 2026 ran with.
    import  Also run vcvarsall.bat and pass the environment it sets up
            (INCLUDE, LIB, LIBPATH, the tools on PATH, ...) to the
            following steps. Use it if the build cannot find the headers
            or libraries of MSVC or of the Windows SDK.

    .PARAMETER Architecture
    The vcvarsall.bat argument used by `import`.

    .LINK
    https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line
    #>
    param(
        [ValidateSet('detect', 'import')][string]$Mode = 'detect',
        [string]$Architecture = 'arm64',
        [string]$VsWhere
    )
    $studio = Find-VisualStudio -VsWhere $VsWhere
    Write-Host "Visual Studio : $($studio.DisplayName) $($studio.Version)"
    Write-Host "Location      : $($studio.InstallationPath)"
    Write-Host "MSVC tools    : $(if ($studio.ToolsVersion) { $studio.ToolsVersion } else { '(no default version file)' })"
    Write-Host "vcvarsall.bat : $($studio.VcVarsAll)"
    Add-GitHubOutput -Name 'vs-installation-path' -Value $studio.InstallationPath
    Add-GitHubOutput -Name 'vs-version' -Value $studio.Version
    Add-GitHubOutput -Name 'vcvarsall' -Value $studio.VcVarsAll
    Add-GitHubOutput -Name 'msvc-tools-version' -Value $studio.ToolsVersion

    if ($Mode -eq 'detect') {
        Write-Host 'Mode detect: the environment of the following steps is unchanged.'
        return
    }

    $before = [ordered]@{}
    foreach ($entry in [Environment]::GetEnvironmentVariables().GetEnumerator()) {
        $before[[string]$entry.Key] = [string]$entry.Value
    }
    $after = Get-VcVarsEnvironment -VcVarsAll $studio.VcVarsAll -Architecture $Architecture
    $change = Export-EnvironmentChange -Before $before -After $after
    Write-Host "Mode import: $($change.Variables.Count) variable(s) and $($change.PathEntries.Count) PATH entr$(if ($change.PathEntries.Count -eq 1) { 'y' } else { 'ies' }) passed to the following steps."
    Write-Host ("Variables     : " + ($change.Variables -join ', '))
}


# ---------------------------------------------------------------------------
# Step 3: pkg-config
# ---------------------------------------------------------------------------

function Resolve-VcpkgRoot {
    <#
    .SYNOPSIS
    Return the directory of the vcpkg that the runner image provides.

    .NOTES
    The image defines VCPKG_INSTALLATION_ROOT. VCPKG_ROOT is what vcpkg
    itself reads and is honoured if a workflow sets it. C:\vcpkg is where
    both Windows ARM64 images keep it and is tried last.

    .LINK
    https://github.com/actions/runner-images/blob/main/images/windows/Windows11-VS2026-Arm64-Readme.md
    #>
    param([string[]]$Candidates)
    if (-not $Candidates) {
        $Candidates = @($env:VCPKG_INSTALLATION_ROOT, $env:VCPKG_ROOT, 'C:\vcpkg') | Where-Object { $_ }
    }
    foreach ($candidate in $Candidates) {
        if (Test-Path -LiteralPath (Join-Path $candidate 'vcpkg.exe') -PathType Leaf) {
            return $candidate
        }
    }
    throw ("vcpkg.exe was not found. Looked in:`n  " + ($Candidates -join "`n  ") + "`n" +
        'Set VCPKG_INSTALLATION_ROOT to the directory that holds vcpkg.exe.')
}

function Install-PkgConf {
    <#
    .SYNOPSIS
    Install pkgconf for ARM64 with vcpkg and make it the pkg-config that
    Meson uses.

    .PARAMETER Triplet
    The vcpkg triplet to install for.

    .PARAMETER Attempts
    How many times to try the installation; vcpkg downloads sources.

    .NOTES
    DEVELOPER NOTES
      * The pkg-config that is first on PATH of a Windows runner belongs to
        Strawberry Perl. It is an x64 program with Perl's own search paths
        and does not find the .pc files of this build. Chocolatey has no
        ARM64 package. PKG_CONFIG tells Meson exactly which program to run.
      * vcpkg installs `pkgconf.exe`; tools look for `pkg-config.exe`, so
        a copy with that name is put beside it.
      * vcpkg finds Visual Studio by itself, with the same vswhere query.

    .LINK
    https://mesonbuild.com/Machine-files.html
    .LINK
    https://learn.microsoft.com/en-us/vcpkg/commands/install
    #>
    param(
        [string]$Triplet = 'arm64-windows',
        [ValidateRange(1, 5)][int]$Attempts = 3,
        [string]$VcpkgRoot
    )
    if (-not $VcpkgRoot) {
        $VcpkgRoot = Resolve-VcpkgRoot
    }
    $vcpkg = Join-Path $VcpkgRoot 'vcpkg.exe'
    $package = "pkgconf:$Triplet"

    # From its own directory vcpkg runs in classic mode whatever the
    # repository contains.
    Push-Location -LiteralPath $VcpkgRoot
    try {
        for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
            $result = Invoke-Native -FilePath $vcpkg -ArgumentList @('install', $package)
            Write-Host $result.Output
            if ($result.ExitCode -eq 0) {
                break
            }
            if ($attempt -eq $Attempts) {
                throw "vcpkg install $package failed $Attempts times (last exit code $($result.ExitCode)); the output is above."
            }
            Write-Warning "vcpkg install $package failed (attempt $attempt of $Attempts); trying again."
            Start-Sleep -Seconds (5 * $attempt)
        }
    }
    finally {
        Pop-Location
    }

    $tools = Join-Path $VcpkgRoot 'installed' $Triplet 'tools' 'pkgconf'
    $pkgconf = Join-Path $tools 'pkgconf.exe'
    if (-not (Test-Path -LiteralPath $pkgconf -PathType Leaf)) {
        throw "vcpkg reported success but $pkgconf does not exist."
    }
    $pkgConfig = Join-Path $tools 'pkg-config.exe'
    Copy-Item -LiteralPath $pkgconf -Destination $pkgConfig -Force

    $probe = Invoke-Native -FilePath $pkgConfig -ArgumentList @('--version')
    if ($probe.ExitCode -ne 0) {
        throw "$pkgConfig does not run (exit code $($probe.ExitCode)):`n$($probe.Output)"
    }
    Write-Host "pkg-config $($probe.Output.Trim()) at $pkgConfig"

    Add-GitHubPath -Directory $tools
    Add-GitHubEnv -Name 'PKG_CONFIG' -Value $pkgConfig
    Add-GitHubOutput -Name 'pkg-config' -Value $pkgConfig
}
