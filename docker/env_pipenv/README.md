# 🧪 Pipenv

A reproducible Python project using [**Pipenv**](https://pipenv.pypa.io).

---

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

| **Action**                                              | **Command**                                             |
|---------------------------------------------------------|---------------------------------------------------------|
| 🧩 **Project Initialization**                           |                                                         |
| Create a new project with Pipenv                        | `pipenv install --python 3.11`                          |
|                                                         |                                                         |
| 🧼 **Environment Cleanup**                              |                                                         |
| Uninstall all dependencies                              | `pipenv uninstall --all`                                |
| Delete virtual environment                              | `pipenv --rm`                                           |
|                                                         |                                                         |
| 🔎 **Inspect & Debug**                                  |                                                         |
| Check installed packages                                | `pipenv graph`                                          |
| Show environment path                                   | `pipenv --venv`                                         |
| Check Python version in venv                            | `pipenv run python --version`                           |
|                                                         |                                                         |
| 🧩 **Importing Existing Dependency Files**              |                                                         |
| Import from `requirements.txt` to `Pipfile`             | `pipenv install -r ../../requirements/all.txt`          |
| Import from `requirements.txt` to `Pipfile` (dev)       | `pipenv install --dev -r requirements-dev.txt`          |
| Export back to `requirements.txt` from `Pipfile.lock`   | `pipenv lock --requirements > requirements.txt`         |
| Generate dev-only `requirements.txt` from `Pipfile.lock`| `pipenv lock --requirements --dev > requirements.txt`   |
|                                                         |                                                         |
| ⚙️ **Dependency Management**                            |                                                         |
| Install packages `Pipfile` with/without `Pipfile.lock`  | `pipenv install`                                        |
| Install packages exactly as locked in `Pipfile.lock`    | `pipenv install --ignore-pipfile`                       |
| Install package                                         | `pipenv install requests`                               |
| Install dev-only package                                | `pipenv install --dev pytest`                           |
| Lock dependencies (like freeze)                         | `pipenv lock`                                           |
| Regenerate lockfile (clear cache)                       | `pipenv lock --clear`                                   |
| Run Python command                                      | `pipenv run python script.py`                           |
| Activate virtual environment shell                      | `pipenv shell`                                          |
| Exit/Deactivate venv                                    | `exit`                                                  |
