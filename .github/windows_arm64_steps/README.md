# Windows ARM64 build dependencies

A composite action for jobs on `windows-11-arm`. It installs LLVM, locates the
Visual Studio C++ tools and installs pkg-config, so that Meson can build the
wheels with `clang-cl` and `flang`.

| File | Role |
|---|---|
| `action.yml` | inputs, outputs, one step per function |
| `windows_arm64.ps1` | the logic, with the developer notes beside each function |
| `tests/test_windows_arm64.ps1` | 29 tests; they run on Linux, macOS and Windows |

## Use

```yaml
- name: Windows ARM64 setup
  if: ${{ runner.os == 'Windows' && matrix.buildplat[2] == 'ARM64' }}
  uses: $/.github/windows_arm64_steps
```

After the step, for the rest of the job:

| Set | Value |
|---|---|
| `CC`, `CXX` | `clang-cl` |
| `FC` | `flang-new`, or `flang` when the LLVM release has only that name |
| `TARGET_ARCH` | `ARM64` |
| `PKG_CONFIG` | full path of the ARM64 `pkg-config.exe` |
| `PATH` | LLVM `bin` and the pkgconf directory in front |

### Inputs

| Input | Default | Meaning |
|---|---|---|
| `llvm-version` | `20.1.8` | LLVM release to install |
| `llvm-sha256` | the hash of that release | SHA-256 of `LLVM-<version>-woa64.exe`; change both together |
| `msvc-environment` | `detect` | `detect` or `import`, see below |
| `self-test` | `false` | `true` runs the tests on the runner first (a few seconds) |

### Outputs

`llvm-bin`, `fortran-compiler`, `vs-installation-path`, `vs-version`,
`msvc-tools-version`, `vcvarsall`, `pkg-config`.

## What the three steps do, and why

### 1. LLVM

Downloads `LLVM-<version>-woa64.exe` from the LLVM release page, checks its
SHA-256, runs it silently, and checks that `clang-cl` and a Fortran compiler
run before selecting them.

- `clang-cl` is clang with MSVC's command line and ABI, the ABI CPython on
  Windows is built with. `cl` itself is not used because there is no MSVC
  Fortran compiler, and C, C++ and Fortran must agree.
- The runner image has an LLVM of its own (20.1.6 at the time of writing).
  A fixed, checked release is installed over it so that the compiler does
  not change when the image does.

### 2. Visual Studio

`clang-cl` has no headers or libraries of its own: it compiles against the
MSVC toolset and the Windows SDK that come with Visual Studio. This step
finds the newest Visual Studio that has the component
`Microsoft.VisualStudio.Component.VC.Tools.ARM64`, using `vswhere`, and
reports its version, location and `vcvarsall.bat`.

| `msvc-environment` | Effect on later steps |
|---|---|
| `detect` (default) | none. `clang-cl` and vcpkg find Visual Studio by themselves. |
| `import` | the environment `vcvarsall.bat arm64` sets up (`INCLUDE`, `LIB`, `LIBPATH`, `VCToolsInstallDir`, the MSVC and SDK tools on `PATH`, ...) is passed on. |

`detect` is the default because it is what every build that passed ran
with, although the previous version of this action looked as if it
activated MSVC. It called

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" arm64
```

and a batch file cannot change the environment of the PowerShell that calls
it: it runs in a `cmd.exe` of its own, which ends when the batch file does.
The line had no effect while the file existed, and stopped the job when it
did not. `import` does what that line meant to do: it runs the batch file
inside one `cmd.exe`, prints the environment there, and writes the
differences to `GITHUB_ENV` and `GITHUB_PATH`.

Switch to `import` if a build fails to find `windows.h`, the C runtime
libraries, `rc.exe` or `mt.exe`.

### 3. pkg-config

Installs `pkgconf:arm64-windows` with the vcpkg of the runner image, copies
`pkgconf.exe` to `pkg-config.exe`, and sets `PKG_CONFIG` to it.

The `pkg-config` that is first on `PATH` of a Windows runner is Strawberry
Perl's: an x64 program with Perl's own search path, which does not find the
`.pc` files of this build. Chocolatey's `pkgconfiglite`, used for x64, has
no ARM64 build. `PKG_CONFIG` tells Meson exactly which program to run.

## What broke in October 2026

Between 21 and 30 September 2026 GitHub moved the `windows-11-arm` label
from the image with Visual Studio 2022 to the one with Visual Studio 2026.
Visual Studio went from `C:\Program Files\Microsoft Visual Studio\2022\Enterprise`
to `C:\Program Files\Microsoft Visual Studio\18\Enterprise`, and the path
written in this action no longer existed:

```text
The term 'C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat'
is not recognized as a name of a cmdlet, function, script file, or executable program.
```

Every ARM64 wheel job failed at that line, for every Python version, because
the step runs before any Python-specific work.

Replacing `2022` with `18` would have worked until the next image. The
installer puts `vswhere.exe` at one fixed place for exactly this purpose,
so the action asks it instead.

## When a step fails

| Message | Meaning | What to do |
|---|---|---|
| `This action sets up Windows on ARM64; the runner is ...` | the step ran on another runner | add `if: runner.os == 'Windows' && runner.arch == 'ARM64'` |
| `Download failed after 4 attempts` | GitHub's release download was unreachable | re-run the job |
| `Checksum mismatch` | the file is not the release the hash belongs to | if you changed `llvm-version`, set `llvm-sha256` from the release page; otherwise re-run |
| `Neither flang-new.exe nor flang.exe` | that LLVM release has no Fortran compiler for Windows ARM64 | use a release whose `woa64` installer has one; 20.1.8 does |
| `vswhere.exe was not found` | no Visual Studio on the runner | the image changed; see the image's README, linked below |
| `No Visual Studio installation has the component ...ARM64` | Visual Studio is there without the ARM64 C++ tools; the message lists what is installed | use a runner image that has them |
| `vcvarsall.bat arm64 failed` (only with `import`) | the batch file reported an error; its output follows | read the output; `detect` skips this call |
| `vcpkg.exe was not found` | the image no longer has vcpkg where it says | set `VCPKG_INSTALLATION_ROOT` in the job |
| `vcpkg install pkgconf:arm64-windows failed 3 times` | vcpkg could not build or download; its output is above | usually a network error: re-run |
| `error STL1000: Unexpected compiler version, expected Clang N or newer` (later, while compiling) | the Visual Studio on the image needs a newer clang than the LLVM installed here | raise `llvm-version` and `llvm-sha256` |
| `fatal error: 'windows.h' file not found`, or a missing `.lib` (later) | `clang-cl` did not find the SDK or MSVC by itself | set `msvc-environment: import` |

## Changing it

- **A newer LLVM.** Take the version and the SHA-256 of
  `LLVM-<version>-woa64.exe` from the release page and set both inputs. If
  the release only has `flang.exe`, nothing else changes.
- **Trying the next runner image early.** GitHub announces a label change
  some weeks ahead and offers the new image under its own label
  (`windows-11-vs2026-arm` was the one for this change). Run the wheel
  workflow once with that label.
- **Editing the script.** Run the tests first and after:

  ```bash
  pwsh -NoProfile -File .github/windows_arm64_steps/tests/test_windows_arm64.ps1
  ```

  They replace the three functions that touch the machine (`Invoke-Native`,
  `Invoke-Download`, `Start-Installer`), so they need neither Windows nor a
  network. What they cannot cover is those three functions and the real
  programs behind them; a run with `self-test: 'true'` on the real runner,
  followed by a wheel build, covers that.

## Sources

Runner image
- [`windows-11-arm` moves to Visual Studio 2026 (announcement, dates)](https://github.com/actions/runner-images/issues/14602)
- [Software on the Windows 11 Arm64 image with Visual Studio 2026](https://github.com/actions/runner-images/blob/main/images/windows/Windows11-VS2026-Arm64-Readme.md)
- [Software on the earlier Windows 11 Arm64 image](https://github.com/actions/runner-images/blob/main/images/windows/Windows11-Arm64-Readme.md)

Visual Studio
- [Finding the C++ tools with vswhere](https://github.com/microsoft/vswhere/wiki/Find-VC)
- [Where vswhere is installed](https://github.com/microsoft/vswhere/wiki/Installing)
- [`vcvarsall.bat` and the developer command prompt](https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line)
- [Component IDs of Visual Studio Enterprise](https://learn.microsoft.com/en-us/visualstudio/install/workload-component-id-vs-enterprise)

LLVM
- [LLVM releases](https://github.com/llvm/llvm-project/releases)
- [clang's MSVC compatibility](https://clang.llvm.org/docs/MSVCCompatibility.html)
- [The flang driver](https://flang.llvm.org/docs/FlangDriver.html)

pkg-config and Meson
- [vcpkg `install`](https://learn.microsoft.com/en-us/vcpkg/commands/install)
- [Meson: cross and native file reference (the `pkg-config` binary)](https://mesonbuild.com/Machine-files.html)

GitHub Actions
- [Composite actions](https://docs.github.com/en/actions/tutorials/create-actions/create-a-composite-action)
- [Passing values to later steps: `GITHUB_ENV`, `GITHUB_PATH`, `GITHUB_OUTPUT`](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-commands)
