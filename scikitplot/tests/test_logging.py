# tests/test_module_getattr_contract.py
import pkgutil
import importlib
import scikitplot

SENTINEL = object()

# def test_scikitplot_module_getattr_contract():
#     for modinfo in pkgutil.walk_packages(scikitplot.__path__, scikitplot.__name__ + "."):
#         m = importlib.import_module(modinfo.name)
#         ga = getattr(m, "__getattr__", None)
#         if callable(ga):
#             assert getattr(m, "__mro__", SENTINEL) is SENTINEL
#             assert getattr(m, "THIS_ATTRIBUTE_SHOULD_NOT_EXIST_12345", SENTINEL) is SENTINEL
