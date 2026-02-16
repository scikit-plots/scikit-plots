# Workflow templates

This folder contains **script templates** intended to be copied out and edited.

Recommended usage:

```bash
python -c "import scikitplot.cython as c; print(c.list_workflows())"
python -c "import scikitplot.cython as c; c.copy_workflow('churn_basic', dest_dir='.')"
cd churn_basic
python cli.py train --help
python cli.py train --data data.csv --out model_artifact.txt
python cli.py hpo
python cli.py predict --model model_artifact.txt
```

All scripts follow the canonical pattern:
- `parse_args(argv=None)`
- `main(argv=None) -> int`
- `if __name__ == '__main__': raise SystemExit(main())`
