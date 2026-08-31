"""
Maintenance tracking helper files like pytest `tests` logic structure.

scikitplot/.../_sphinx_ai_assistant/       ← runtime + tests only
maintenances/_externals/_sphinx_ext/
└── _sphinx_ai_assistant/                  ← maintenance control plane
    ├── _backup/
    ├── _maintenance/
    │   ├── checkpoints/
    │   ├── history/
    │   └── APP_STREAMING_RUNBOOK.md
    ├── MAINTAINING.md
    ├── todo/                              ← existing task content preserved here
    └── __init__.py

repository/
├── scikitplot/
│   └── _externals/
│       └── _sphinx_ext/
│           └── _sphinx_ai_assistant/
│               ├── runtime code
│               ├── deployment helpers
│               ├── tests/
│               └── NO maintenance-only files
│
└── maintenances/
    └── _externals/
        └── _sphinx_ext/
            └── _sphinx_ai_assistant/
                ├── _backup/
                ├── _maintenance/
                │   ├── checkpoints/
                │   ├── history/
                │   ├── schemas/
                │   ├── APP_STREAMING_RUNBOOK.md
                │   └── ...
                ├── MAINTAINING.md
                ├── todo/
                │   ├── lessons.md
                │   └── todo.md
                └── __init__.py
"""
