.. _docker-recovery-index:

======================================================================
Docker Desktop WSL Recovery Guidelines
======================================================================

Windows WSL + Cross-Platform Notes
-----------------------------------------

Canonical, safe, production-grade recovery procedure for restoring Docker
environments when backend storage is intact but runtime layer is broken.


1. Situation Overview (Windows WSL2 Case)
-----------------------------------------

This guide applies when:

- Docker Desktop is stuck on "Starting..."
- ``wsl -l -v`` shows no distributions
- BUT Docker disk files still exist:

  - ext4.vhdx
  - docker_data.vhdx

Interpretation:

- WSL metadata is broken
- Docker engine is not registered
- UI cannot attach backend
- Container data still exists on disk


2. Architecture Model (Critical Understanding)
----------------------------------------------

Docker Desktop WSL2 stores state in virtual disks:

+------------+----------------------+--------------------------+
| Component  | File                 | Purpose                  |
+============+======================+==========================+
| System     | ext4.vhdx            | Linux VM + Docker engine |
+------------+----------------------+--------------------------+
| Data       | docker_data.vhdx     | images, containers, vols |
+------------+----------------------+--------------------------+

These are full filesystem images (not archives).


3. Critical Safety Rules
------------------------

DO NOT:

- Delete E:\DockerDesktopWSL
- Run wsl --unregister docker-desktop-data
- Reinstall Docker Desktop before recovery
- Manually convert or modify .vhdx files

DO:

- Keep .vhdx files untouched
- Use WSL import mechanisms only
- Verify before restarting Docker


4. Pre-check (System Validation)
--------------------------------

.. code-block:: powershell

   wsl --version
   wsl --status

Expected:

- WSL installed
- Virtualization enabled


5. Recovery Strategy (Preferred Method)
---------------------------------------

Step 1 — Stop services

.. code-block:: powershell

   wsl --shutdown
   Stop-Service com.docker.service

Step 2 — Restore system disk

.. code-block:: powershell

   wsl --import-in-place docker-desktop "E:\DockerDesktopWSL\DockerDesktopWSL\main\ext4.vhdx"

Step 3 — Restore data disk

.. code-block:: powershell

   wsl --import-in-place docker-desktop-data "E:\DockerDesktopWSL\DockerDesktopWSL\disk\docker_data.vhdx"


6. Fallback Method (Compatibility Mode)
---------------------------------------

If import-in-place is not available:

System:

.. code-block:: powershell

   wsl --import docker-desktop E:\DockerDesktopWSL\recovery\docker-desktop \
   "E:\DockerDesktopWSL\DockerDesktopWSL\main\ext4.vhdx" --vhd

Data:

.. code-block:: powershell

   wsl --import docker-desktop-data E:\DockerDesktopWSL\recovery\docker-desktop-data \
   "E:\DockerDesktopWSL\DockerDesktopWSL\disk\docker_data.vhdx" --vhd


7. Verification
---------------

.. code-block:: powershell

   wsl -l -v

Expected:

- docker-desktop
- docker-desktop-data


8. Restart Docker
-----------------

.. code-block:: powershell

   Start-Service com.docker.service

Then open Docker Desktop UI.


9. Root Causes
--------------

Typical failures:

- Windows update breaks WSL registry
- Docker Desktop upgrade mismatch
- Corrupted WSL metadata

.vhdx files remain intact unless deleted manually.


10. Failure Handling
--------------------

File in use:

.. code-block:: powershell

   wsl --shutdown

WSL missing:

.. code-block:: powershell

   wsl --install


11. Recovery Outcome
--------------------

If .vhdx intact:

- Containers restored
- Images restored
- Volumes preserved
- No data loss


12. Key Principle
-----------------

Docker Desktop = WSL control layer + VHDX storage layer

If storage exists:

- Data is safe
- Only mapping must be rebuilt


13. Summary Flow
----------------

1. Stop WSL + Docker
2. Import ext4.vhdx
3. Import docker_data.vhdx
4. Verify WSL distros
5. Restart Docker Desktop


14. Linux Recovery Notes
------------------------

Storage:

/var/lib/docker

Restart:

.. code-block:: bash

   sudo systemctl restart docker

Inspect:

.. code-block:: bash

   sudo du -sh /var/lib/docker


15. macOS Recovery Notes
------------------------

Storage:
::

  ~/Library/Containers/com.docker.docker/Data/vms/0/Docker.raw

Actions:

- Restart Docker Desktop
- Use Troubleshoot reset if needed


Cross-platform model:

+---------+--------------------------+
| OS      | Storage model            |
+=========+==========================+
| Windows | WSL2 + VHDX             |
| Linux   | Native filesystem        |
| macOS   | VM disk image           |
+---------+--------------------------+


Final Warning
-------------

Do not reinstall or clean Docker before recovery is confirmed.
