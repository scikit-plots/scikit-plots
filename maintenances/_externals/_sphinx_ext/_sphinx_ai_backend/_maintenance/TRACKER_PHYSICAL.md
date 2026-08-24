# Physical Tracker — `_sphinx_ai_backend`

> **PROPOSED submodule.** Its contents currently live inside
> `_sphinx_ai_assistant/`. This pair describes the target, and the
> rationale for getting there.

Re-derived from the tree. Do not hand-edit:

```console
$ python scikitplot/_externals/_sphinx_ext/_sphinx_ai_backend/_maintenance/check_trackers.py
```

## Totals

```text
source files    15   source LOC     9896
test files       0   test LOC          0
backup files     0   backup LOC        0
```

test : source LOC = **0.00**

## By extension

```text
{
  ".py": 6,
  ".txt": 3,
  ".md": 3,
  "": 1,
  ".js": 1,
  ".toml": 1
}
```

## Paths in scope

```text
  _hf_spaces_proxy
  _hf_spaces_model
  _cf_worker
  dev_proxy.py
```

## Tripwires

| Metric | Now | Tripwire |
|---|---|---|
| test files | **0** | must become > 0 before any change |
| services | 3 | a fourth without a deployment contract |
| secrets in source | 0 (unverified) | **any** |

## Known physical debt

### 1. **Zero tests on 9 896 LOC of internet-facing code**

The proxy handles user input, forwards to a model, and collects a dataset. It is the highest-risk code in the family and the least verified.

### 2. Mixed into an installed Python package

Deployed services and `pip install` artifacts have different lifecycles.

### 3. `robots.txt` and `DATASET_COLLECTION_GUIDANCE.md` sit beside the app

Policy documents inside a service directory; they belong in `_maintenance/`.
