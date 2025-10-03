# 🧪 Pipenv

A reproducible Python project using [**Pipenv**](https://pipenv.pypa.io).

See Also:
- https://pipenv.pypa.io/en/latest/faq.html#how-do-i-use-pipenv-with-vs-code
- https://pipenv.pypa.io/en/latest/faq.html#where-does-pipenv-store-virtual-environments
- https://pipenv.pypa.io/en/latest/faq.html#dependency-management
- https://pipenv.pypa.io/en/latest/faq.html#how-do-i-configure-pipenv


## 🐍 Python Environment Setup (with Pipenv)

### Install dependencies:

```bash
pip install pipenv         # Only once, if you don't have it
```

```bash
pipenv install             # Install default packages `Pipfile` with/without `Pipfile.lock`
```

```bash
pipenv install --dev       # Also install dev packages (pytest, black, etc.)
```

### Activate virtual environment:

```bash
pipenv shell
```

### Docker

```bash
# Option 1: Docker use pipenv directly (`Pipfile.lock`)
RUN pip install pipenv && pipenv install --deploy --ignore-pipfile
```

### 📦 [Pipenv Quick Start Guide][quick_start]

[quick_start]: https://pipenv.pypa.io/en/latest/quick_start.html

| **Action**                                                                                                                 | **Command**                                                                       |
|----------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| 🧩 **Project Initialization**                                                                                              |                                                                                   |
| Create a new project environment with Pipenv                                                                               | `pipenv --python 3.11`                                                            |
|                                                                                                                            |                                                                                   |
| 🧼 **Environment Cleanup**                                                                                                 |                                                                                   |
| Uninstall all dependencies                                                                                                 | `pipenv uninstall --all`                                                          |
| Delete/Remove the virtual environment                                                                                      | `pipenv --rm`                                                                     |
|                                                                                                                            |                                                                                   |
| 🔎 **Inspect & Debug**                                                                                                     |                                                                                   |
| Check Python version in venv                                                                                               | `pipenv run python --version`                                                     |
| Show environment path                                                                                                      | `pipenv --venv`                                                                   |
| Check installed packages                                                                                                   | `pipenv graph`                                                                    |
|                                                                                                                            |                                                                                   |
| 🧩 **Importing Existing Dependency Files**                                                                                 |                                                                                   |
| Install package(s)                                                                                                         | `pipenv install requests`                                                         |
| Import from `requirements.txt` to `Pipfile`                                                                                | `pipenv install -r ../../requirements/all.txt`                                    |
| Import from `requirements.txt` to `Pipfile` (dev)                                                                          | `pipenv install --dev -r requirements_dev.txt`                                    |
| Export back to `requirements.txt` from `Pipfile.lock`                                                                      | `pipenv requirements > requirements.txt`                                          |
| Generate dev-only `requirements.txt` from `Pipfile.lock`                                                                   | `pipenv requirements --dev > requirements.txt`                                    |
| (legacy) Export back to `requirements.txt` from `Pipfile.lock`                                                             | `pipenv lock --requirements > requirements.txt`                                   |
| (legacy) Generate dev-only `requirements.txt` from `Pipfile.lock`                                                          | `pipenv lock --requirements --dev > requirements.txt`                             |
|                                                                                                                            |                                                                                   |
| ⚙️ **Dependency Management**                                                                                               |                                                                                   |
| Installs packages exactly as specified in the `Pipfile.lock` without updating it.                                          | `pipenv sync`                                                                     |
| Installs packages specified in the `Pipfile`, updates the `Pipfile.lock` if necessary, and installs the packages.          | `pipenv install`                                                                  |
| Install packages exactly as locked in `Pipfile.lock`, Ignores the `Pipfile` entirely so No dependency resolution happens.  | `pipenv install --ignore-pipfile`                                                 |
| Install packages just enforce present and up-to-date / strict mode (Fail fast if lockfile is missing/stale or out-of-date).| `pipenv install --deploy`                                                         |
| Install packages control directly into the currently active Python interpreter or a new venv, No virtualenv is created.    | `pipenv install --system` or `PIPENV_IGNORE_VIRTUALENVS=1 pipenv install`         |
| Install packages exactly the locked packages into the intended Python environment.                                         | `pipenv install --ignore-pipfile --deploy --system`                               |
| Install package dev-only                                                                                                   | `pipenv install --dev pytest`                                                     |
| Activate virtual environment current shell                                                                                 | `pipenv shell`                                                                    |
| In virtual environment run Python command                                                                                  | `pipenv run python script.py`                                                     |
| Verify `Pipfile.lock`                                                                                                      | `pipenv verify`                                                                   |
| Lock dependencies to update the lock file. (like freeze)                                                                   | `pipenv lock`                                                                     |
| Regenerate lockfile (clear cache)                                                                                          | `pipenv lock --clear`                                                             |
| Exit/Deactivate venv                                                                                                       | `exit`                                                                            |

---

General Questions

What is Pipenv?

Pipenv is a Python dependency management tool that combines pip, virtualenv, and Pipfile into a single unified interface. It creates and manages virtual environments for your projects automatically, while also maintaining a Pipfile for package requirements and a Pipfile.lock for deterministic builds.

Why should I use Pipenv instead of pip?

While pip is excellent for installing Python packages, Pipenv offers several advantages:
* Automatic virtualenv management: Creates and manages virtual environments for you
* Dependency resolution: Resolves dependencies and sub-dependencies
* Lock file: Generates a Pipfile.lock with exact versions and hashes for deterministic builds
* Development vs. production dependencies: Separates dev dependencies from production
* Security features: Checks for vulnerabilities and verifies hashes
* Environment variable management: Automatically loads .env files

How does Pipenv compare to Poetry?

Both Pipenv and Poetry are modern Python dependency management tools, but they have different focuses:

Pipenv:
* Focuses on application development
* Simpler, more straightforward approach
* Officially recommended by Python Packaging Authority (PyPA)
* Better integration with pip and virtualenv

Poetry:
* Focuses on both application and library development
* Includes package building and publishing features
* Has its own dependency resolver
* More opinionated about project structure

Choose Pipenv if you want a straightforward tool for application development that integrates well with the existing Python ecosystem. Choose Poetry if you’re developing libraries or need its additional packaging features.

Pipfile and Pipfile.lock

What is the difference between Pipfile and Pipfile.lock?
- Pipfile: A human-readable file that specifies your project’s dependencies with version constraints. It’s meant to be edited by humans.
- Pipfile.lock: A machine-generated file that contains exact versions and hashes of all dependencies (including sub-dependencies). It ensures deterministic builds and should not be edited manually.

Should I commit both Pipfile and Pipfile.lock to version control?
- Pipfile: Contains your direct dependencies and version constraints
- Pipfile.lock: Ensures everyone using your project gets the exact same dependencies

---
