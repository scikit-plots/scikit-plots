"""
Backend discovery and safe fallback
===================================

Inspect backend capability without making optional accelerators mandatory.
"""

from scikitplot.levenshtein import available_backends, backend_info, distance

for info in available_backends():
    print(
        f"{info.name:11s} available={str(info.available):5s} "
        f"license={info.license}"
    )

auto = backend_info("auto")
assert auto.available

# This path always exists, even in a minimal installation.
assert distance("kitten", "sitting", backend="python") == 3

# The GPL backend is visible to explicit capability discovery but is not part
# of the automatic preference chain.
names = [item.name for item in available_backends()]
assert names == ["internal", "rapidfuzz", "levenshtein", "python"]
print("auto selected:", auto.name)
