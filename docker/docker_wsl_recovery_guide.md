# 🛠 Docker Desktop WSL Recovery Guide (Windows) + Cross-Platform Notes (Linux & macOS)

> **Canonical, safe, production-grade recovery procedure for restoring Docker environments when backend storage is intact but runtime layer is broken.**

---

# 🧠 1. Situation Overview (Windows WSL2 Case)

This guide applies when:

- Docker Desktop is stuck on **“Starting…”**
- `wsl -l -v` shows **no distributions**
- BUT Docker disk files still exist:
  - `ext4.vhdx`
  - `docker_data.vhdx`

👉 Interpretation:

- ❌ WSL metadata broken
- ❌ Docker engine not registered
- ❌ UI cannot attach backend
- ✅ Container data still exists on disk

---

# 🧱 2. Architecture Model (Critical Understanding)

Docker Desktop WSL2 stores state in virtual disks:

| Component | File | Purpose |
|----------|------|--------|
| System | `ext4.vhdx` | Linux VM + Docker engine |
| Data | `docker_data.vhdx` | images, containers, volumes |

👉 These are full filesystem images (not archives)

---

# ⚠️ 3. Critical Safety Rules

### ❌ DO NOT

- Delete `E:\DockerDesktopWSL`
- Run `wsl --unregister docker-desktop-data`
- Reinstall Docker Desktop before recovery
- Manually convert or open `.vhdx`

### ✅ DO

- Keep `.vhdx` files untouched
- Use only WSL import mechanisms
- Verify before restarting Docker

---

# 🔍 4. Pre-check (System Validation)

```powershell
wsl --version
wsl --status
```

Expected:
- WSL installed
- Virtualization enabled

---

# 🚀 5. Recovery (Preferred Modern Method)

## 🟢 Step 1 — Stop services

```powershell
wsl --shutdown
Stop-Service com.docker.service
```

---

## 🟢 Step 2 — Restore system disk

```powershell
wsl --import-in-place docker-desktop "E:\DockerDesktopWSL\DockerDesktopWSL\main\ext4.vhdx"
```

---

## 🟢 Step 3 — Restore data disk

```powershell
wsl --import-in-place docker-desktop-data "E:\DockerDesktopWSL\DockerDesktopWSL\disk\docker_data.vhdx"
```

---

# 🟡 6. Fallback Method (Compatibility Mode)

If `--import-in-place` is not available:

## System

```powershell
wsl --import docker-desktop E:\DockerDesktopWSL\recovery\docker-desktop "E:\DockerDesktopWSL\DockerDesktopWSL\main\ext4.vhdx" --vhd
```

## Data

```powershell
wsl --import docker-desktop-data E:\DockerDesktopWSL\recovery\docker-desktop-data "E:\DockerDesktopWSL\DockerDesktopWSL\disk\docker_data.vhdx" --vhd
```

---

# 🔍 7. Verification

```powershell
wsl -l -v
```

Expected:

```
docker-desktop
docker-desktop-data
```

---

# 🐳 8. Restart Docker

```powershell
Start-Service com.docker.service
```

Then open Docker Desktop UI.

---

# 🧠 9. Root Causes

Typical reasons for breakage:

- Windows update breaks WSL registry
- Docker Desktop upgrade mismatch
- Corrupted WSL metadata

👉 `.vhdx` files remain untouched unless manually deleted

---

# 🧯 10. Failure Handling

## ❌ Import fails (file in use)

```powershell
wsl --shutdown
```
Retry import.

---

## ❌ WSL not available

```powershell
wsl --install
```
Reboot system.

---

# 🟢 11. Recovery Outcome

If `.vhdx` intact:

✔ Containers restored
✔ Images restored
✔ Volumes preserved
✔ No data loss

---

# 📌 12. Key Principle

> Docker Desktop = WSL control layer + VHDX storage layer

If storage exists:

👉 Data is safe
👉 Only mapping must be rebuilt

---

# 🧾 13. Summary Flow

1. Stop WSL + Docker
2. Import `ext4.vhdx`
3. Import `docker_data.vhdx`
4. Verify WSL distros
5. Restart Docker Desktop

---

# 🐧 14. Linux Recovery Notes (Native Docker Engine)

If Docker breaks on Linux:

## 📍 Default storage location

```bash
/var/lib/docker
```

## 🧯 Recovery scenarios

### Service restart
```bash
sudo systemctl restart docker
```

### Inspect storage
```bash
sudo du -sh /var/lib/docker
```

### Corruption recovery
- Stop Docker
- Backup `/var/lib/docker`
- Reinstall `docker-ce`
- Restore volumes if needed

👉 No WSL layer exists on Linux
👉 Storage is direct filesystem-based

---

# 🍎 15. macOS Recovery Notes (Docker Desktop + HyperKit / QEMU VM)

Docker Desktop on macOS stores everything inside a **virtual disk image**.

## 📍 Storage location

```bash
~/Library/Containers/com.docker.docker/Data/vms/0/
```

Key file:

- `Docker.raw` (main disk image)

---

## 🧯 Recovery scenarios

### Restart Docker VM
- Quit Docker Desktop
- Restart application

### Reset WSL-like VM layer
Docker Desktop → Troubleshoot → Restart / Reset

### Full recovery (advanced)
- Backup `Docker.raw`
- Reinstall Docker Desktop
- Restore disk image

---

# 🧠 Cross-Platform Insight

| OS | Storage Model |
|----|--------------|
| Windows | WSL2 + VHDX |
| Linux | Native filesystem (/var/lib/docker) |
| macOS | VM disk image (Docker.raw) |

👉 All platforms use **persistent storage abstraction layer**

---

# 🛡 Final Warning

Do not perform cleanup or reinstall until recovery is verified.

This procedure is safe **only if disk files remain intact and untouched**.
