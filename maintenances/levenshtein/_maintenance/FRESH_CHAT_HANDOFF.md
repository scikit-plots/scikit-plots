# Fresh-chat handoff

Start with:

1. `maintenances/levenshtein/MAINTAINING.md`
2. `maintenances/levenshtein/_maintenance/RESUME.md`
3. `maintenances/levenshtein/_maintenance/DESIGN.md`
4. `maintenances/levenshtein/REVIEW.json`

Then run the focused checker and tests from `VERIFICATION.md`.

The next design problem is not "add more fuzzy matching functions." It is to
resolve the backend execution plan once per operation so runtime fallback and
large ranking share one deterministic, warning-bounded path.
