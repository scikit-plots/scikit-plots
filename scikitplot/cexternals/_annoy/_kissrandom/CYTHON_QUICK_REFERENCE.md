# Cython Quick Reference

Quick lookup guide for common Cython patterns in the kissrandom project.

## File Types Quick Reference

| Extension | Purpose | Edit When |
|-----------|---------|-----------|
| `.h` | C++ header | Never (upstream) |
| `.pyi` | Python type hints | Public API changes |
| `.pxd` | Cython declarations | Wrapping C++ code |
| `.pxi` | Shared code (deprecated) | Never (avoid) |
| `.pyx` | Cython implementation | Main development |

## Common Cython Patterns

### Wrapping a C++ Class

```cython
# In .pxd (declarations)
cdef extern from "header.h" namespace "MyNamespace" nogil:
    cdef cppclass CppClass:
        void method()
        int value

# In .pyx (implementation)
cdef class PyClass:
    cdef CppClass* _obj

    def __cinit__(self):
        self._obj = new CppClass()

    def __dealloc__(self):
        if self._obj is not NULL:
            del self._obj

    cpdef void method(self):
        self._obj.method()
```

### Function Types

```cython
# Python-only (slow)
def python_func(x):
    return x * 2

# C-only (fast, not callable from Python)
cdef int c_func(int x):
    return x * 2

# Both (best of both worlds)
cpdef int hybrid_func(int x):
    return x * 2
```

### Type Declarations

```cython
# Cython types
cdef int x = 5
cdef double y = 3.14
cdef uint32_t z = 42
cdef size_t idx = 0

# Python types
cdef object obj = "hello"
cdef list lst = [1, 2, 3]
cdef dict dct = {"key": "value"}

# Pointers
cdef int* ptr
cdef CppClass* obj_ptr

# Arrays
cdef int[10] array
cdef int[:] memview  # Memory view (modern)
```

### Memory Management

```cython
# Allocation
cdef CppClass* obj = new CppClass()

# Deallocation
del obj

# Check before delete
if obj is not NULL:
    del obj

# Memory view (no manual management)
cdef int[:] view = array
```

### nogil Usage

```cython
# Function that releases GIL
cpdef int fast_func(int x) nogil:
    return x * 2

# With block (release GIL temporarily)
with nogil:
    result = c_level_function()

# Can't use Python objects in nogil
cpdef int bad_nogil(self) nogil:
    py_obj = [1, 2, 3]  # ERROR!
```

### Compile-Time Constants

```cython
# Use DEF (modern)
DEF MAX_SIZE = 1000
DEF IS_DEBUG = False

cdef int array[MAX_SIZE]

# Not IF/ELSE (old style, avoid)
IF PLATFORM == "Linux":  # Avoid
    pass
```

### Static Methods

```cython
cdef class MyClass:
    @staticmethod
    def static_method():
        return 42

    @staticmethod
    cdef int c_static(int x):
        return x * 2
```

### Properties (Getters/Setters)

```cython
cdef class MyClass:
    cdef int _value  # Private storage

    @property
    def value(self) -> int:
        """Getter."""
        return self._value

    @value.setter
    def value(self, int new_value):
        """Setter with validation."""
        if new_value < 0:
            raise ValueError("Must be non-negative")
        self._value = new_value

# Usage:
obj = MyClass()
print(obj.value)  # Getter
obj.value = 42    # Setter
```

### Inline Functions

```cython
cdef inline int fast_add(int a, int b) nogil:
    return a + b
```

## Compiler Directives

```cython
# At top of .pyx file
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# cython: binding=True
```

| Directive | Effect | Use When |
|-----------|--------|----------|
| `language_level=3` | Python 3 syntax | Always |
| `boundscheck=False` | Skip array bounds checks | Trusted code |
| `wraparound=False` | Disable negative indexing | Performance |
| `cdivision=True` | C division (no zero check) | Performance |
| `embedsignature=True` | Docstring signatures | Public API |
| `binding=True` | Better introspection | Public API |

## Build Commands

```bash
# Development build
python setup.py build_ext --inplace

# Clean build
python setup.py clean --all
python setup.py build_ext --inplace

# Install
pip install .

# Install in development mode
pip install -e .

# Create source distribution
python setup.py sdist

# Create wheel
python setup.py bdist_wheel

# Generate annotation HTML
cython -a kissrandom.pyx
```

## Import Patterns

```cython
# C standard library
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.stdint cimport uint32_t, uint64_t
from libc.stddef cimport size_t

# C++ standard library
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map

# Import from .pxd (C-level)
cimport mymodule
from mymodule cimport MyClass, my_function

# Import from .pyx (Python-level)
import mymodule
from mymodule import MyClass
```

## Type Conversions

```cython
# Python int to C int
cdef int c_val = <int>py_val

# C type to Python
py_val = <object>c_val

# Explicit cast
cdef uint32_t u = <uint32_t>-1

# Memory view to pointer
cdef int[:] view = array
cdef int* ptr = &view[0]
```

## Exception Handling

```cython
# Declare function can raise exceptions
cdef int may_fail() except -1:
    if error:
        raise ValueError("Error!")
    return result

# Propagate all exceptions
cdef int may_fail() except *:
    # Can raise any exception
    pass

# Check returned value
cdef int* get_ptr() except NULL:
    if allocation_failed:
        raise MemoryError()
    return ptr
```

## NumPy Integration

```cython
import numpy as np
cimport numpy as cnp

# Initialize NumPy C API
cnp.import_array()

# Type numpy arrays
def process(cnp.ndarray[cnp.int32_t, ndim=1] arr):
    cdef int i
    for i in range(arr.shape[0]):
        arr[i] = arr[i] * 2
```

## Common Pitfalls and Solutions

### Pitfall 1: Memory Leaks

```cython
# ❌ BAD: Memory leak
cdef class Bad:
    cdef int* data
    def __cinit__(self):
        self.data = <int*>malloc(1000 * sizeof(int))
    # Missing __dealloc__!

# ✅ GOOD: Proper cleanup
cdef class Good:
    cdef int* data
    def __cinit__(self):
        self.data = <int*>malloc(1000 * sizeof(int))
    def __dealloc__(self):
        if self.data is not NULL:
            free(self.data)
```

### Pitfall 2: Python Objects in nogil

```cython
# ❌ BAD: Python object in nogil
cpdef int bad() nogil:
    lst = [1, 2, 3]  # ERROR!
    return len(lst)

# ✅ GOOD: Use C types
cpdef int good() nogil:
    cdef int arr[3]
    arr[:] = [1, 2, 3]
    return 3
```

### Pitfall 3: Forgetting to Rebuild After .pxd Changes

```bash
# ❌ BAD: Incremental build misses .pxd changes
python setup.py build_ext --inplace

# ✅ GOOD: Force complete rebuild
python setup.py clean --all
python setup.py build_ext --inplace
```

### Pitfall 4: Type Mismatches

```cython
# ❌ BAD: Implicit conversion may lose data
cdef uint32_t large = 4294967295
cdef int signed = large  # Overflow!

# ✅ GOOD: Explicit cast and check
cdef uint32_t large = 4294967295
if large > 2147483647:
    raise ValueError("Value too large for int")
cdef int signed = <int>large
```

## Performance Tips

1. **Type everything**: Untyped variables are Python objects (slow)
2. **Use nogil**: Release GIL for C-only code
3. **Inline small functions**: `cdef inline` for hot paths
4. **Use memory views**: Faster than ndarray indexing
5. **Disable checks in production**: `boundscheck=False`, etc.
6. **Profile first**: Use `cython -a` to find bottlenecks
7. **Avoid Python API calls**: In nogil sections
8. **Use C types**: `int`, not `object`

## Testing Tips

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=kissrandom

# Run benchmarks
pytest -m benchmark

# Skip slow tests
pytest -m "not slow"

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

## Debugging Tips

```bash
# Generate annotated HTML
cython -a kissrandom.pyx

# Enable debug symbols
DEBUG=1 python setup.py build_ext --inplace

# Use gdb
gdb python
> run script.py

# Check for memory leaks (Linux)
valgrind --leak-check=full python script.py
```

## Resources

- **Cython Docs**: https://cython.readthedocs.io/
- **C++ Interop**: https://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html
- **Modern Patterns**: https://github.com/cython/cython/issues/4310
- **Performance Tips**: https://cython.readthedocs.io/en/latest/src/userguide/optimization.html

## Version Compatibility

```python
# Check Cython version
import Cython
print(Cython.__version__)

# Require minimum version in setup.py
setup_requires=["cython>=0.29.0"]

# Check Python version
import sys
if sys.version_info < (3, 7):
    raise RuntimeError("Python 3.7+ required")
```

---

**Quick Command Summary**

```bash
# Build
python setup.py build_ext --inplace

# Test
pytest

# Clean
python setup.py clean --all

# Annotate
cython -a *.pyx

# Package
python setup.py sdist bdist_wheel
```
