# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

<#
.SYNOPSIS
Tests for ../windows_arm64.ps1. They run on any system with PowerShell 7:

    pwsh -NoProfile -File .github/windows_arm64_steps/tests/test_windows_arm64.ps1

.NOTES
DEVELOPER NOTES
  Nothing here needs Windows, Visual Studio or a network. The script under
  test reaches the machine through Invoke-Native, Invoke-Download and
  Start-Installer only; each test replaces those with a stand-in and builds
  the directories it needs under a temporary folder. What stays untested is
  therefore exactly those three functions and the real programs behind
  them; README.md says how that part was checked.

  No test framework is used, so the file runs on a fresh runner.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot '..' 'windows_arm64.ps1')

$script:Failures = @()
$script:Passed = 0
$script:Root = Join-Path ([IO.Path]::GetTempPath()) "windows-arm64-tests-$PID"

function Assert-Equal {
    param($Actual, $Expected, [string]$Because = '')
    # Wrapped in an array on both sides, so that "no lines" compares as empty.
    $a = ConvertTo-Json -InputObject @($Actual) -Depth 6 -Compress
    $e = ConvertTo-Json -InputObject @($Expected) -Depth 6 -Compress
    if ($a -cne $e) { throw "expected $e but got $a. $Because" }
}

function Assert-True {
    param($Condition, [string]$Because = 'condition is false')
    if (-not $Condition) { throw $Because }
}

function Assert-Throws {
    param([string]$Pattern, [scriptblock]$Action)
    try { & $Action | Out-Null }
    catch {
        if ($_.Exception.Message -notmatch $Pattern) {
            throw "threw, but the message does not match '$Pattern': $($_.Exception.Message)"
        }
        return
    }
    throw "did not throw (expected a message matching '$Pattern')"
}

function New-Workspace {
    # A fresh folder with the three files GitHub Actions would provide.
    $folder = Join-Path $script:Root ([guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $folder -Force | Out-Null
    foreach ($name in 'GITHUB_ENV', 'GITHUB_PATH', 'GITHUB_OUTPUT') {
        $file = Join-Path $folder $name
        New-Item -ItemType File -Path $file -Force | Out-Null
        [Environment]::SetEnvironmentVariable($name, $file)
    }
    $env:RUNNER_TEMP = $folder
    return $folder
}

function Get-Lines {
    param([string]$Variable)
    # The comma keeps an empty file an empty array instead of nothing.
    return , @(Get-Content -LiteralPath ([Environment]::GetEnvironmentVariable($Variable)))
}

function New-File {
    param([string]$Path, [string]$Text = '')
    New-Item -ItemType Directory -Path (Split-Path -Parent $Path) -Force | Out-Null
    Set-Content -LiteralPath $Path -Value $Text -NoNewline
}

function Test {
    param([string]$Name, [scriptblock]$Body)
    $here = Get-Location
    try {
        & $Body
        $script:Passed++
        Write-Host "  ok    $Name"
    }
    catch {
        $script:Failures += $Name
        Write-Host "  FAIL  $Name`n        $($_.Exception.Message)"
    }
    finally {
        Set-Location $here
    }
}

# Waiting between attempts has no place in a test.
function Start-Sleep { param($Seconds) }


Write-Host 'Passing values to later steps'

Test 'a variable is written as NAME=value' {
    New-Workspace | Out-Null
    Add-GitHubEnv -Name 'CC' -Value 'clang-cl'
    Add-GitHubEnv -Name 'EMPTY' -Value ''
    Assert-Equal (Get-Lines GITHUB_ENV) @('CC=clang-cl', 'EMPTY=')
}

Test 'a name that is not a variable name is refused' {
    New-Workspace | Out-Null
    Assert-Throws 'Not a valid environment variable name' { Add-GitHubEnv -Name 'A B' -Value 'x' }
    Assert-Throws 'Not a valid environment variable name' { Add-GitHubEnv -Name 'ProgramFiles(x86)' -Value 'x' }
    Assert-Equal (Get-Lines GITHUB_ENV) @()
}

Test 'a value with a line break is refused, for all three files' {
    New-Workspace | Out-Null
    Assert-Throws 'line break' { Add-GitHubEnv -Name 'A' -Value "x`nINJECTED=1" }
    Assert-Throws 'line break' { Add-GitHubPath -Directory "C:\a`r`nC:\b" }
    Assert-Throws 'line break' { Add-GitHubOutput -Name 'o' -Value "x`ny" }
}

Test 'outside GitHub Actions the reason is stated' {
    New-Workspace | Out-Null
    [Environment]::SetEnvironmentVariable('GITHUB_ENV', $null)
    Assert-Throws 'GITHUB_ENV is not set' { Add-GitHubEnv -Name 'A' -Value 'b' }
}


Write-Host 'LLVM'

function Use-LlvmStandIns {
    # Download writes a known file; the installer creates the tools named.
    param([string]$Folder, [string[]]$Tools, [int]$ExitCode = 0)
    $script:Llvm = @{ Uri = ''; Installed = $false; Install = Join-Path $Folder 'LLVM'; Tools = $Tools; ExitCode = $ExitCode }
    function script:Invoke-Download {
        param($Uri, $OutFile)
        $script:Llvm.Uri = $Uri
        Set-Content -LiteralPath $OutFile -Value 'installer' -NoNewline
    }
    function script:Start-Installer {
        param($FilePath, $ArgumentList)
        $script:Llvm.Installed = $true
        $script:Llvm.Arguments = $ArgumentList
        foreach ($tool in $script:Llvm.Tools) { New-File (Join-Path $script:Llvm.Install 'bin' $tool) }
        return $script:Llvm.ExitCode
    }
    function script:Invoke-Native {
        param($FilePath, $ArgumentList)
        return [pscustomobject]@{ ExitCode = 0; Output = "$(Split-Path -Leaf $FilePath) version 20.1.8`nTarget: aarch64-pc-windows-msvc" }
    }
}

# SHA-256 of the seven bytes "installer".
$InstallerHash = [BitConverter]::ToString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::ASCII.GetBytes('installer'))).Replace('-', '')

Test 'the release is downloaded, verified, installed and selected' {
    $folder = New-Workspace
    Use-LlvmStandIns -Folder $folder -Tools 'clang-cl.exe', 'flang-new.exe', 'flang.exe'
    Install-Llvm -Version '20.1.8' -Sha256 $InstallerHash.ToLowerInvariant() -InstallDirectory $script:Llvm.Install
    Assert-Equal $script:Llvm.Uri 'https://github.com/llvm/llvm-project/releases/download/llvmorg-20.1.8/LLVM-20.1.8-woa64.exe'
    Assert-Equal @($script:Llvm.Arguments) @('/S')
    Assert-Equal (Get-Lines GITHUB_ENV) @('CC=clang-cl', 'CXX=clang-cl', 'FC=flang-new', 'TARGET_ARCH=ARM64')
    Assert-Equal (Get-Lines GITHUB_PATH) @((Join-Path $script:Llvm.Install 'bin'))
    Assert-True ((Get-Lines GITHUB_OUTPUT) -contains 'fortran-compiler=flang-new')
}

Test 'an LLVM that only has the new Fortran name is accepted' {
    $folder = New-Workspace
    Use-LlvmStandIns -Folder $folder -Tools 'clang-cl.exe', 'flang.exe'
    Install-Llvm -Version '22.1.0' -Sha256 $InstallerHash -InstallDirectory $script:Llvm.Install
    Assert-True ((Get-Lines GITHUB_ENV) -contains 'FC=flang')
}

Test 'a wrong checksum stops before the installer runs' {
    $folder = New-Workspace
    Use-LlvmStandIns -Folder $folder -Tools 'clang-cl.exe', 'flang.exe'
    Assert-Throws 'Checksum mismatch' { Install-Llvm -Version '20.1.8' -Sha256 ('0' * 64) -InstallDirectory $script:Llvm.Install }
    Assert-Equal $script:Llvm.Installed $false
    Assert-Equal (Get-Lines GITHUB_ENV) @()
}

Test 'a version or checksum of the wrong shape is refused' {
    $folder = New-Workspace
    Use-LlvmStandIns -Folder $folder -Tools 'clang-cl.exe', 'flang.exe'
    Assert-Throws 'Not an LLVM release number' { Install-Llvm -Version '20.1' -Sha256 $InstallerHash -InstallDirectory $script:Llvm.Install }
    Assert-Throws 'Not an LLVM release number' { Install-Llvm -Version '20.1.8/../x' -Sha256 $InstallerHash -InstallDirectory $script:Llvm.Install }
    Assert-Throws 'Not a SHA-256' { Install-Llvm -Version '20.1.8' -Sha256 'abc' -InstallDirectory $script:Llvm.Install }
}

Test 'an installer that fails, or leaves a compiler out, stops the step' {
    $folder = New-Workspace
    Use-LlvmStandIns -Folder $folder -Tools 'clang-cl.exe', 'flang.exe' -ExitCode 3
    Assert-Throws 'exit code 3' { Install-Llvm -Version '20.1.8' -Sha256 $InstallerHash -InstallDirectory $script:Llvm.Install }
    $folder = New-Workspace
    Use-LlvmStandIns -Folder $folder -Tools 'flang.exe'
    Assert-Throws 'clang-cl.exe is not in' { Install-Llvm -Version '20.1.8' -Sha256 $InstallerHash -InstallDirectory $script:Llvm.Install }
    $folder = New-Workspace
    Use-LlvmStandIns -Folder $folder -Tools 'clang-cl.exe'
    Assert-Throws 'no Fortran compiler' { Install-Llvm -Version '20.1.8' -Sha256 $InstallerHash -InstallDirectory $script:Llvm.Install }
    Assert-Equal (Get-Lines GITHUB_ENV) @()
}

Test 'a compiler that does not run stops the step' {
    $folder = New-Workspace
    Use-LlvmStandIns -Folder $folder -Tools 'clang-cl.exe', 'flang.exe'
    function script:Invoke-Native { param($FilePath, $ArgumentList) [pscustomobject]@{ ExitCode = 127; Output = 'cannot execute' } }
    Assert-Throws 'does not run \(exit code 127\)' { Install-Llvm -Version '20.1.8' -Sha256 $InstallerHash -InstallDirectory $script:Llvm.Install }
}

Test 'only https is downloaded' {
    . (Join-Path $PSScriptRoot '..' 'windows_arm64.ps1')   # the real Invoke-Download again
    Assert-Throws 'anything but https' { Invoke-Download -Uri 'http://example.invalid/x.exe' -OutFile 'x.exe' }
}


Write-Host 'Visual Studio'

function New-VisualStudio {
    # A directory laid out like an installation; returns its vswhere record.
    param([string]$Folder, [string]$Release = '18', [switch]$NoVcVars, [switch]$NoDefault)
    $root = Join-Path $Folder 'Microsoft Visual Studio' $Release 'Enterprise'
    New-Item -ItemType Directory -Path $root -Force | Out-Null
    if (-not $NoVcVars) { New-File (Join-Path $root 'VC' 'Auxiliary' 'Build' 'vcvarsall.bat') '@echo off' }
    if (-not $NoDefault) { New-File (Join-Path $root 'VC' 'Auxiliary' 'Build' 'Microsoft.VCToolsVersion.default.txt') "14.50.35717`r`n" }
    return [ordered]@{ installationPath = $root; installationVersion = '18.8.12023.21'; displayName = 'Visual Studio Enterprise 2026' }
}

function Use-VsWhere {
    # -requires answers with $Matching, -all with $All; arguments are kept.
    param($Matching, $All, [int]$ExitCode = 0)
    $script:VsWhere = @{ Matching = $Matching; All = $All; ExitCode = $ExitCode; Calls = @() }
    function script:Invoke-Native {
        param($FilePath, $ArgumentList)
        $script:VsWhere.Calls += , @($ArgumentList)
        $answer = if ($ArgumentList -contains '-all') { $script:VsWhere.All } else { $script:VsWhere.Matching }
        $json = if (@($answer).Count -eq 0) { '[]' } else { ConvertTo-Json -InputObject @($answer) -Depth 4 }
        return [pscustomobject]@{ ExitCode = $script:VsWhere.ExitCode; Output = $json }
    }
}

Test 'vswhere output: none, empty, one, and not JSON' {
    Assert-Equal @(ConvertFrom-VsWhereJson -Text '[]').Count 0
    Assert-Equal @(ConvertFrom-VsWhereJson -Text '').Count 0
    Assert-Equal @(ConvertFrom-VsWhereJson -Text '[{"installationPath":"C:\\VS"}]').Count 1
    Assert-Throws 'did not print JSON' { ConvertFrom-VsWhereJson -Text 'Error 0x57: bad parameter' }
}

Test 'Visual Studio 2026 is found where it is, with no path written down' {
    $folder = New-Workspace
    $record = New-VisualStudio -Folder $folder -Release '18'
    Use-VsWhere -Matching $record -All $record
    $studio = Find-VisualStudio -VsWhere 'vswhere'
    Assert-Equal $studio.InstallationPath $record.installationPath
    Assert-Equal $studio.Version '18.8.12023.21'
    Assert-Equal $studio.ToolsVersion '14.50.35717'
    Assert-True (Test-Path -LiteralPath $studio.VcVarsAll) 'vcvarsall.bat should exist'
    $arguments = $script:VsWhere.Calls[0]
    foreach ($needed in '-latest', '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.ARM64', '-format', 'json') {
        Assert-True ($arguments -contains $needed) "vswhere was not given $needed"
    }
}

Test 'Visual Studio 2022 is found the same way' {
    $folder = New-Workspace
    $record = New-VisualStudio -Folder $folder -Release '2022'
    $record.installationVersion = '17.14.37516.0'
    Use-VsWhere -Matching $record -All $record
    Assert-True ((Find-VisualStudio -VsWhere 'vswhere').InstallationPath -like '*2022*')
}

Test 'an installation without the ARM64 tools is named in the error' {
    $folder = New-Workspace
    $record = New-VisualStudio -Folder $folder
    Use-VsWhere -Matching @() -All $record
    Assert-Throws 'No Visual Studio installation has the component[\s\S]*Visual Studio Enterprise 2026 18\.8' { Find-VisualStudio -VsWhere 'vswhere' }
    Use-VsWhere -Matching @() -All @()
    Assert-Throws 'No Visual Studio is installed' { Find-VisualStudio -VsWhere 'vswhere' }
}

Test 'a failing vswhere, or an installation without vcvarsall.bat, stops the step' {
    $folder = New-Workspace
    Use-VsWhere -Matching @() -All @() -ExitCode 87
    Assert-Throws 'vswhere failed \(exit code 87\)' { Find-VisualStudio -VsWhere 'vswhere' }
    $record = New-VisualStudio -Folder $folder -NoVcVars
    Use-VsWhere -Matching $record -All $record
    Assert-Throws 'has no VC.Auxiliary.Build.vcvarsall\.bat' { Find-VisualStudio -VsWhere 'vswhere' }
}

Test 'a missing default tools version is reported as empty, not as an error' {
    $folder = New-Workspace
    $record = New-VisualStudio -Folder $folder -NoDefault
    Use-VsWhere -Matching $record -All $record
    Assert-Equal (Find-VisualStudio -VsWhere 'vswhere').ToolsVersion ''
}

Test 'vswhere is taken from the first place it exists' {
    $folder = New-Workspace
    $present = Join-Path $folder 'b' 'vswhere.exe'
    New-File $present
    Assert-Equal (Resolve-VsWhere -Candidates (Join-Path $folder 'a' 'vswhere.exe'), $present) $present
}

Test 'the output of set becomes a dictionary' {
    $text = "banner without equals`r`n=C:=C:\work`r`n  INCLUDE=C:\a;C:\b`r`n  FLAGS=/DX=1`r`n  EMPTY=`r`n"
    $parsed = ConvertFrom-SetOutput -Text $text
    Assert-Equal @($parsed.Keys) @('INCLUDE', 'FLAGS', 'EMPTY')
    Assert-Equal $parsed['FLAGS'] '/DX=1'
    Assert-Equal $parsed['EMPTY'] ''
}

function Use-Cmd {
    # Stand in for cmd.exe running the command file.
    param([string]$Output, [int]$ExitCode = 0)
    $script:Cmd = @{ Output = $Output; ExitCode = $ExitCode; File = ''; Text = ''; Arguments = @() }
    function script:Invoke-Native {
        param($FilePath, $ArgumentList)
        $script:Cmd.Arguments = @($ArgumentList)
        $script:Cmd.File = $ArgumentList[-1]
        $script:Cmd.Text = Get-Content -LiteralPath $script:Cmd.File -Raw
        return [pscustomobject]@{ ExitCode = $script:Cmd.ExitCode; Output = $script:Cmd.Output }
    }
}

$VcVarsOutput = @(
    '**********************************************************************',
    '** Visual Studio 2026 Developer Command Prompt v18.8',
    'NOT_A_VARIABLE=printed before the marker',
    '===ENVIRONMENT===',
    'INCLUDE=C:\VS\include;C:\SDK\include',
    'LIB=C:\VS\lib\arm64',
    'VCToolsInstallDir=C:\VS\VC\Tools\MSVC\14.50.35717\',
    'Path=C:\VS\bin\Hostarm64\arm64;C:\SDK\bin;C:\Windows'
) -join "`r`n"

Test 'vcvarsall.bat is called from a command file and its environment read back' {
    $folder = New-Workspace
    Use-Cmd -Output $VcVarsOutput
    $vcvars = Join-Path $folder 'Visual Studio' 'vcvarsall.bat'
    $environment = Get-VcVarsEnvironment -VcVarsAll $vcvars -Architecture arm64 -WorkDirectory $folder
    Assert-True ($script:Cmd.Text -match [regex]::Escape("call `"$vcvars`" arm64")) 'the command file does not call vcvarsall.bat with the architecture'
    Assert-True ($script:Cmd.Text -match 'if errorlevel 1 exit /b 1') 'a failing vcvarsall.bat would go unnoticed'
    Assert-Equal @($script:Cmd.Arguments[0..1]) @('/d', '/c')
    Assert-Equal $environment['LIB'] 'C:\VS\lib\arm64'
    Assert-True (-not $environment.Contains('NOT_A_VARIABLE')) 'text before the marker was read as a variable'
    Assert-True (-not (Test-Path -LiteralPath $script:Cmd.File)) 'the command file was left behind'
}

Test 'a vcvarsall.bat that fails, or sets up nothing, stops the step' {
    $folder = New-Workspace
    Use-Cmd -Output '[ERROR:vcvarsall.bat] Invalid argument found : arm64' -ExitCode 1
    Assert-Throws 'vcvarsall.bat arm64 failed \(exit code 1\)[\s\S]*Invalid argument' { Get-VcVarsEnvironment -VcVarsAll 'v.bat' -WorkDirectory $folder }
    Use-Cmd -Output "===ENVIRONMENT===`r`nPath=C:\Windows"
    Assert-Throws 'did not set VCToolsInstallDir' { Get-VcVarsEnvironment -VcVarsAll 'v.bat' -WorkDirectory $folder }
    Use-Cmd -Output 'nothing useful'
    Assert-Throws 'vcvarsall.bat arm64 failed' { Get-VcVarsEnvironment -VcVarsAll 'v.bat' -WorkDirectory $folder }
    Assert-True (-not (Test-Path -LiteralPath $script:Cmd.File)) 'the command file was left behind after a failure'
}

Test 'only what changed is passed on, and PATH keeps its order' {
    New-Workspace | Out-Null
    $before = [ordered]@{ Path = 'C:\Windows;C:\Tools\'; SAME = '1'; CHANGED = 'old'; 'ProgramFiles(x86)' = 'C:\PF86' }
    $after = [ordered]@{
        Path = 'C:\VS\bin;C:\SDK\bin;c:\windows;C:\Tools;C:\VS\bin\'
        SAME = '1'; CHANGED = 'new'; ADDED = 'x'; 'ProgramFiles(x86)' = 'different'
    }
    $change = Export-EnvironmentChange -Before $before -After $after
    Assert-Equal (Get-Lines GITHUB_ENV) @('CHANGED=new', 'ADDED=x')
    Assert-Equal @($change.PathEntries) @('C:\VS\bin', 'C:\SDK\bin')
    # Last written comes first on PATH, so the file holds them reversed.
    Assert-Equal (Get-Lines GITHUB_PATH) @('C:\SDK\bin', 'C:\VS\bin')
}

Test 'detect reports Visual Studio and changes nothing' {
    $folder = New-Workspace
    $record = New-VisualStudio -Folder $folder
    Use-VsWhere -Matching $record -All $record
    Initialize-Msvc -Mode detect -VsWhere 'vswhere'
    Assert-Equal (Get-Lines GITHUB_ENV) @()
    Assert-Equal (Get-Lines GITHUB_PATH) @()
    $outputs = Get-Lines GITHUB_OUTPUT
    Assert-True ($outputs -contains "vs-installation-path=$($record.installationPath)")
    Assert-True ($outputs -contains 'vs-version=18.8.12023.21')
    Assert-True ($outputs -contains 'msvc-tools-version=14.50.35717')
}

Test 'import passes the MSVC environment to later steps' {
    $folder = New-Workspace
    $record = New-VisualStudio -Folder $folder
    $json = ConvertTo-Json -InputObject @($record) -Depth 4
    function script:Invoke-Native {
        param($FilePath, $ArgumentList)
        if ($ArgumentList -contains '-format') { return [pscustomobject]@{ ExitCode = 0; Output = $json } }
        return [pscustomobject]@{ ExitCode = 0; Output = $VcVarsOutput }
    }
    Initialize-Msvc -Mode import -VsWhere 'vswhere'
    $written = Get-Lines GITHUB_ENV
    Assert-True ($written -contains 'INCLUDE=C:\VS\include;C:\SDK\include')
    Assert-True ($written -contains 'LIB=C:\VS\lib\arm64')
    Assert-True (-not ($written -match '^Path=')) 'PATH must go to GITHUB_PATH, not GITHUB_ENV'
    Assert-Equal (Get-Lines GITHUB_PATH)[-1] 'C:\VS\bin\Hostarm64\arm64'
}

Test 'a mode that does not exist is refused' {
    Assert-Throws 'Cannot validate argument on parameter .Mode' { Initialize-Msvc -Mode 'activate' -VsWhere 'vswhere' }
}


Write-Host 'pkg-config'

function Use-Vcpkg {
    # vcpkg succeeds on the attempt given; pkgconf.exe appears if $Creates.
    param([string]$Root, [int]$SucceedsOn = 1, [bool]$Creates = $true, [int]$ProbeExit = 0)
    $script:Vcpkg = @{ Root = $Root; SucceedsOn = $SucceedsOn; Creates = $Creates; Calls = 0; ProbeExit = $ProbeExit; Where = '' }
    New-File (Join-Path $Root 'vcpkg.exe')
    function script:Invoke-Native {
        param($FilePath, $ArgumentList)
        if ($ArgumentList[0] -eq '--version') {
            return [pscustomobject]@{ ExitCode = $script:Vcpkg.ProbeExit; Output = "2.5.1`n" }
        }
        $script:Vcpkg.Calls++
        $script:Vcpkg.Arguments = @($ArgumentList)
        $script:Vcpkg.Where = (Get-Location).Path
        if ($script:Vcpkg.Calls -lt $script:Vcpkg.SucceedsOn) {
            return [pscustomobject]@{ ExitCode = 1; Output = 'error: failed to download' }
        }
        if ($script:Vcpkg.Creates) {
            New-File (Join-Path $script:Vcpkg.Root 'installed' 'arm64-windows' 'tools' 'pkgconf' 'pkgconf.exe') 'binary'
        }
        return [pscustomobject]@{ ExitCode = 0; Output = 'Total install time: 12 s' }
    }
}

Test 'pkgconf is installed, copied to the name tools look for, and selected' {
    $folder = New-Workspace
    $root = Join-Path $folder 'vcpkg'
    Use-Vcpkg -Root $root
    $start = (Get-Location).Path
    Install-PkgConf -VcpkgRoot $root
    $tools = Join-Path $root 'installed' 'arm64-windows' 'tools' 'pkgconf'
    $pkgConfig = Join-Path $tools 'pkg-config.exe'
    Assert-Equal @($script:Vcpkg.Arguments) @('install', 'pkgconf:arm64-windows')
    Assert-Equal (Get-Content -LiteralPath $pkgConfig -Raw) 'binary'
    Assert-Equal (Get-Lines GITHUB_ENV) @("PKG_CONFIG=$pkgConfig")
    Assert-Equal (Get-Lines GITHUB_PATH) @($tools)
    Assert-Equal (Split-Path -Leaf $script:Vcpkg.Where) 'vcpkg' 'vcpkg must run from its own directory'
    Assert-Equal (Get-Location).Path $start 'the working directory was not restored'
}

Test 'a failed download is tried again, a persistent failure stops the step' {
    $folder = New-Workspace
    $root = Join-Path $folder 'vcpkg'
    Use-Vcpkg -Root $root -SucceedsOn 3
    Install-PkgConf -VcpkgRoot $root
    Assert-Equal $script:Vcpkg.Calls 3
    $folder = New-Workspace
    $root = Join-Path $folder 'vcpkg'
    Use-Vcpkg -Root $root -SucceedsOn 99
    $start = (Get-Location).Path
    Assert-Throws 'failed 3 times' { Install-PkgConf -VcpkgRoot $root }
    Assert-Equal $script:Vcpkg.Calls 3
    Assert-Equal (Get-Location).Path $start 'the working directory was not restored after a failure'
    Assert-Equal (Get-Lines GITHUB_ENV) @()
}

Test 'success without the program, or a program that does not run, stops the step' {
    $folder = New-Workspace
    $root = Join-Path $folder 'vcpkg'
    Use-Vcpkg -Root $root -Creates $false
    Assert-Throws 'reported success but .*pkgconf\.exe does not exist' { Install-PkgConf -VcpkgRoot $root }
    $folder = New-Workspace
    $root = Join-Path $folder 'vcpkg'
    Use-Vcpkg -Root $root -ProbeExit 1
    Assert-Throws 'pkg-config\.exe does not run' { Install-PkgConf -VcpkgRoot $root }
    Assert-Equal (Get-Lines GITHUB_ENV) @()
}

Test 'vcpkg is taken from the first candidate that has it' {
    $folder = New-Workspace
    $root = Join-Path $folder 'second'
    New-File (Join-Path $root 'vcpkg.exe')
    Assert-Equal (Resolve-VcpkgRoot -Candidates (Join-Path $folder 'first'), $root) $root
    Assert-Throws 'vcpkg\.exe was not found[\s\S]*VCPKG_INSTALLATION_ROOT' { Resolve-VcpkgRoot -Candidates (Join-Path $folder 'none') }
}


Remove-Item -LiteralPath $script:Root -Recurse -Force -ErrorAction SilentlyContinue
Write-Host ''
if ($script:Failures.Count) {
    Write-Host "$($script:Passed) passed, $($script:Failures.Count) FAILED:"
    $script:Failures | ForEach-Object { Write-Host "  $_" }
    exit 1
}
Write-Host "$($script:Passed) passed"
exit 0
