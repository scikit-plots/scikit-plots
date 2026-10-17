#!/usr/bin/env python3
"""
Comprehensive test suite for memmap module (Windows memory mapping).

This test suite validates:
- Basic memory mapping operations
- Anonymous and file-backed mappings
- Memory protection changes
- Read/write operations
- Context manager support
- Error handling and edge cases

Test Organization
-----------------
- test_module_*: Module-level tests
- test_anonymous_*: Anonymous mapping tests
- test_file_*: File-backed mapping tests
- test_protection_*: Memory protection tests
- test_error_*: Error handling tests
- test_context_*: Context manager tests

Running Tests
-------------
Run all tests:
    pytest test_mman.py -v

Run specific test:
    pytest test_mman.py::test_anonymous_basic -v

Run with coverage:
    pytest test_mman.py --cov=mman --cov-report=html

Skip slow tests:
    pytest test_mman.py -m "not slow"

Platform Notes
--------------
- These tests are Windows-specific
- Some tests require file I/O permissions
- Memory locking tests may require elevated privileges

Design Principles
-----------------
- Test each public method thoroughly
- Validate edge cases (zero size, invalid fds, etc.)
- Test resource cleanup (context managers)
- Validate error handling
- Use fixtures for common setup
- Clear test names and docstrings
"""

import sys
import gc
import os
import subprocess
import tempfile
import textwrap
import pytest
from typing import Generator

# Platform check
# if sys.platform != "win32":
#     pytest.skip("mman tests are Windows-only", allow_module_level=True)

# Import the module to test
try:
    from scikitplot import memmap
    from scikitplot.memmap import (
        PY_PROT_NONE, PY_PROT_READ, PY_PROT_WRITE, PY_PROT_EXEC,
        PY_MAP_SHARED, PY_MAP_PRIVATE, PY_MAP_ANONYMOUS, PY_MAP_ANON,
        PY_MS_SYNC, PY_MS_ASYNC,
        MemoryMap,
        mmap_region,
        MMapError, MMapAllocationError,
    )
except ImportError as e:
    pytest.skip(f"memmap module not built: {e}", allow_module_level=True)


# ===========================================================================
# Test Fixtures
# ===========================================================================

@pytest.fixture
def temp_file() -> Generator[tuple[int, str], None, None]:
    """
    Fixture providing a temporary file with file descriptor.

    Yields
    ------
    tuple[int, str]
        (file_descriptor, file_path)
    """
    with tempfile.NamedTemporaryFile(delete=False, mode='w+b') as f:
        # Write some test data
        test_data = b"Hello, World! " * 100
        f.write(test_data)
        f.flush()

        fd = f.fileno()
        path = f.name

        yield fd, path

    # Cleanup
    try:
        os.unlink(path)
    except:
        pass


@pytest.fixture
def small_mapping() -> Generator[MemoryMap, None, None]:
    """Fixture providing a small anonymous memory mapping."""
    mapping = MemoryMap.create_anonymous(4096, PY_PROT_READ | PY_PROT_WRITE)
    yield mapping
    if mapping.is_valid:
        mapping.close()


# ===========================================================================
# Module-Level Tests
# ===========================================================================

# def test_module_version():
#     """Test that module has version attribute."""
#     assert hasattr(memmap, '__version__')
#     assert isinstance(memmap.__version__, str)
#     assert len(memmap.__version__) > 0


def test_module_exports():
    """Test that module exports expected constants and classes."""
    # Protection flags
    assert hasattr(memmap, 'PY_PROT_NONE')
    assert hasattr(memmap, 'PY_PROT_READ')
    assert hasattr(memmap, 'PY_PROT_WRITE')
    assert hasattr(memmap, 'PY_PROT_EXEC')

    # Mapping flags
    assert hasattr(memmap, 'PY_MAP_SHARED')
    assert hasattr(memmap, 'PY_MAP_PRIVATE')
    assert hasattr(memmap, 'PY_MAP_ANONYMOUS')

    # Classes
    assert hasattr(memmap, 'MemoryMap')
    assert hasattr(memmap, 'MMapError')


def test_protection_flag_values():
    """Test that protection flags have correct values."""
    assert PY_PROT_NONE == 0
    assert PY_PROT_READ == 1
    assert PY_PROT_WRITE == 2
    assert PY_PROT_EXEC == 4


def test_mapping_flag_values():
    """Test that mapping flags have correct values."""
    assert PY_MAP_SHARED == 1
    assert PY_MAP_PRIVATE == 2
    assert PY_MAP_ANONYMOUS == 0x20
    assert PY_MAP_ANON == PY_MAP_ANONYMOUS


# ===========================================================================
# Anonymous Mapping Tests
# ===========================================================================

class TestAnonymousMapping:
    """Tests for anonymous memory mappings."""

    def test_create_basic(self):
        """Test basic anonymous mapping creation."""
        mapping = MemoryMap.create_anonymous(4096, PY_PROT_READ | PY_PROT_WRITE)
        assert mapping is not None
        assert mapping.is_valid
        assert mapping.size == 4096
        mapping.close()
        assert not mapping.is_valid

    def test_create_with_defaults(self):
        """Test creation with default parameters."""
        mapping = MemoryMap.create_anonymous(4096)
        assert mapping.is_valid
        mapping.close()

    def test_create_read_only(self):
        """Test creation with read-only protection."""
        mapping = MemoryMap.create_anonymous(4096, PY_PROT_READ)
        assert mapping.is_valid
        mapping.close()

    def test_create_invalid_size_zero(self):
        """Test that size=0 raises ValueError."""
        with pytest.raises(ValueError, match="Size must be positive"):
            MemoryMap.create_anonymous(0)

    def test_create_invalid_size_negative(self):
        """Test that negative size raises ValueError."""
        with pytest.raises(ValueError, match="Size must be positive"):
            MemoryMap.create_anonymous(-1)

    def test_addr_property(self, small_mapping):
        """Test addr property returns valid address."""
        addr = small_mapping.addr
        assert isinstance(addr, int)
        assert addr > 0  # Valid memory address

    def test_size_property(self, small_mapping):
        """Test size property returns correct size."""
        assert small_mapping.size == 4096

    def test_is_valid_property(self):
        """Test is_valid property reflects mapping state."""
        mapping = MemoryMap.create_anonymous(4096)
        assert mapping.is_valid
        mapping.close()
        assert not mapping.is_valid


# ===========================================================================
# Read/Write Tests
# ===========================================================================

class TestReadWrite:
    """Tests for read/write operations."""

    def test_write_basic(self, small_mapping):
        """Test basic write operation."""
        data = b"Hello, World!"
        n_written = small_mapping.write(data)
        assert n_written == len(data)

    def test_read_basic(self, small_mapping):
        """Test basic read operation."""
        # Write some data
        test_data = b"Test data"
        small_mapping.write(test_data)

        # Read it back
        read_data = small_mapping.read(len(test_data))
        assert read_data == test_data

    def test_write_with_offset(self, small_mapping):
        """Test write with offset."""
        data1 = b"First"
        data2 = b"Second"

        small_mapping.write(data1, offset=0)
        small_mapping.write(data2, offset=100)

        # Verify both writes
        assert small_mapping.read(len(data1), offset=0) == data1
        assert small_mapping.read(len(data2), offset=100) == data2

    def test_read_with_offset(self, small_mapping):
        """Test read with offset."""
        data = b"0123456789"
        small_mapping.write(data)

        # Read middle portion
        partial = small_mapping.read(5, offset=2)
        assert partial == b"23456"

    def test_write_read_only_fails(self):
        """Test that writing to read-only mapping fails."""
        mapping = MemoryMap.create_anonymous(4096, PY_PROT_READ)
        with pytest.raises(ValueError, match="not writable"):
            mapping.write(b"test")
        mapping.close()

    def test_read_beyond_bounds_fails(self, small_mapping):
        """Test that reading beyond bounds fails."""
        with pytest.raises(ValueError, match="beyond mapping bounds"):
            small_mapping.read(5000)  # Larger than 4096

    def test_write_beyond_bounds_fails(self, small_mapping):
        """Test that writing beyond bounds fails."""
        large_data = b"x" * 5000
        with pytest.raises(ValueError, match="beyond mapping bounds"):
            small_mapping.write(large_data)

    def test_read_closed_mapping_fails(self):
        """Test that reading closed mapping fails."""
        mapping = MemoryMap.create_anonymous(4096)
        mapping.close()
        with pytest.raises(ValueError, match="closed"):
            mapping.read(10)

    def test_write_closed_mapping_fails(self):
        """Test that writing to closed mapping fails."""
        mapping = MemoryMap.create_anonymous(4096)
        mapping.close()
        with pytest.raises(ValueError, match="closed"):
            mapping.write(b"test")


# ===========================================================================
# File-Backed Mapping Tests
# ===========================================================================

class TestFileBacked:
    """Tests for file-backed memory mappings."""

    def test_create_file_mapping(self, temp_file):
        """Test creating file-backed mapping."""
        fd, path = temp_file
        mapping = MemoryMap.create_file_mapping(fd, 0, 4096, PY_PROT_READ)
        assert mapping.is_valid
        assert mapping.size == 4096
        mapping.close()

    def test_read_from_file_mapping(self, temp_file):
        """Test reading from file-backed mapping."""
        fd, path = temp_file
        mapping = MemoryMap.create_file_mapping(fd, 0, 100, PY_PROT_READ)

        data = mapping.read(13)  # "Hello, World!"
        assert data == b"Hello, World!"

        mapping.close()

    def test_create_with_offset(self, temp_file):
        """Test creating mapping with file offset."""
        fd, path = temp_file
        # Note: offset should be page-aligned in real usage
        mapping = MemoryMap.create_file_mapping(fd, 0, 100, PY_PROT_READ)
        assert mapping.is_valid
        mapping.close()

    def test_create_invalid_fd_fails(self):
        """Test that invalid file descriptor fails."""
        with pytest.raises(ValueError, match="Invalid file descriptor"):
            MemoryMap.create_file_mapping(-999, 0, 4096)

    def test_create_negative_offset_fails(self, temp_file):
        """Test that negative offset fails."""
        fd, path = temp_file
        with pytest.raises(ValueError, match="non-negative"):
            MemoryMap.create_file_mapping(fd, -1, 4096)


# ===========================================================================
# Memory Protection Tests
# ===========================================================================

class TestMemoryProtection:
    """Tests for memory protection changes."""

    def test_mprotect_add_write(self):
        """Test adding write permission."""
        mapping = MemoryMap.create_anonymous(4096, PY_PROT_READ)

        # Initially can't write
        with pytest.raises(ValueError, match="not writable"):
            mapping.write(b"test")

        # Add write permission
        mapping.mprotect(PY_PROT_READ | PY_PROT_WRITE)

        # Now can write
        n = mapping.write(b"test")
        assert n == 4

        mapping.close()

    def test_mprotect_remove_write(self):
        """Test removing write permission."""
        mapping = MemoryMap.create_anonymous(4096, PY_PROT_READ | PY_PROT_WRITE)

        # Initially can write
        mapping.write(b"test")

        # Remove write permission
        mapping.mprotect(PY_PROT_READ)

        # Now can't write
        with pytest.raises(ValueError, match="not writable"):
            mapping.write(b"more")

        mapping.close()

    def test_mprotect_closed_fails(self):
        """Test that mprotect on closed mapping fails."""
        mapping = MemoryMap.create_anonymous(4096)
        mapping.close()

        with pytest.raises(ValueError, match="closed"):
            mapping.mprotect(PY_PROT_READ)


# ===========================================================================
# Sync Tests
# ===========================================================================

class TestSync:
    """Tests for msync operations."""

    def test_msync_basic(self, temp_file):
        """Test basic msync operation."""
        fd, path = temp_file
        mapping = MemoryMap.create_file_mapping(
            fd, 0, 4096, PY_PROT_READ | PY_PROT_WRITE, PY_MAP_SHARED
        )

        # Write some data
        mapping.write(b"Modified")

        # Sync to disk
        mapping.msync(PY_MS_SYNC)  # Should not raise

        mapping.close()

    def test_msync_async(self, temp_file):
        """Test async msync."""
        fd, path = temp_file
        mapping = MemoryMap.create_file_mapping(
            fd, 0, 4096, PY_PROT_READ | PY_PROT_WRITE, PY_MAP_SHARED
        )

        mapping.write(b"Data")
        mapping.msync(PY_MS_ASYNC)  # Should return immediately

        mapping.close()

    def test_msync_closed_fails(self):
        """Test that msync on closed mapping fails."""
        mapping = MemoryMap.create_anonymous(4096)
        mapping.close()

        with pytest.raises(ValueError, match="closed"):
            mapping.msync()


# ===========================================================================
# Context Manager Tests
# ===========================================================================

class TestContextManager:
    """Tests for context manager support."""

    def test_context_manager_basic(self):
        """Test basic context manager usage."""
        with MemoryMap.create_anonymous(4096) as mapping:
            assert mapping.is_valid
            mapping.write(b"test")

        # Should be closed after context
        assert not mapping.is_valid

    def test_context_manager_with_exception(self):
        """Test that mapping is closed even if exception occurs."""
        mapping = None
        try:
            with MemoryMap.create_anonymous(4096) as m:
                mapping = m
                assert mapping.is_valid
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should still be closed
        assert not mapping.is_valid

    def test_context_manager_read_write(self):
        """Test read/write within context manager."""
        test_data = b"Context manager test"

        with MemoryMap.create_anonymous(4096, PY_PROT_READ | PY_PROT_WRITE) as m:
            m.write(test_data)
            read_data = m.read(len(test_data))
            assert read_data == test_data


# ===========================================================================
# Module Function Tests
# ===========================================================================

class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_mmap_region_anonymous(self):
        """Test mmap_region for anonymous mapping."""
        mapping = mmap_region(4096)
        assert mapping.is_valid
        assert mapping.size == 4096
        mapping.close()

    def test_mmap_region_with_flags(self):
        """Test mmap_region with custom flags."""
        mapping = mmap_region(
            4096,
            prot=PY_PROT_READ | PY_PROT_WRITE,
            flags=PY_MAP_PRIVATE | PY_MAP_ANONYMOUS
        )
        assert mapping.is_valid
        mapping.close()


# ===========================================================================
# Error Handling Tests
# ===========================================================================

class TestErrorHandling:
    """Tests for error handling."""

    def test_double_close_idempotent(self):
        """Test that double close is safe."""
        mapping = MemoryMap.create_anonymous(4096)
        mapping.close()
        mapping.close()  # Should not raise

    def test_access_after_close_fails(self):
        """Test that accessing closed mapping fails."""
        mapping = MemoryMap.create_anonymous(4096, PY_PROT_READ | PY_PROT_WRITE)
        mapping.write(b"test")
        mapping.close()

        # All operations should fail
        with pytest.raises(ValueError):
            mapping.read(10)

        with pytest.raises(ValueError):
            mapping.write(b"more")

        with pytest.raises(ValueError):
            _ = mapping.addr

        with pytest.raises(ValueError):
            _ = mapping.size


# ===========================================================================
# String Representation Tests
# ===========================================================================

class TestStringRepresentation:
    """Tests for __repr__ and __str__."""

    def test_repr_valid_mapping(self):
        """Test repr of valid mapping."""
        mapping = MemoryMap.create_anonymous(4096)
        repr_str = repr(mapping)

        assert "MemoryMap" in repr_str
        assert "addr=" in repr_str
        assert "size=" in repr_str

        mapping.close()

    def test_repr_closed_mapping(self):
        """Test repr of closed mapping."""
        mapping = MemoryMap.create_anonymous(4096)
        mapping.close()

        repr_str = repr(mapping)
        assert "closed" in repr_str.lower()

    def test_str_valid_mapping(self):
        """Test str of valid mapping."""
        mapping = MemoryMap.create_anonymous(4096)
        str_repr = str(mapping)
        assert "MemoryMap" in str_repr
        mapping.close()


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestIntegration:
    """Integration tests for real-world usage."""

    def test_large_write_read(self):
        """Test writing and reading large data."""
        size = 1024 * 1024  # 1 MB
        test_data = b"x" * size

        with MemoryMap.create_anonymous(size, PY_PROT_READ | PY_PROT_WRITE) as m:
            m.write(test_data)
            read_data = m.read(size)
            assert read_data == test_data

    def test_multiple_mappings(self):
        """Test creating multiple independent mappings."""
        mappings = []
        try:
            for i in range(10):
                m = MemoryMap.create_anonymous(4096, PY_PROT_READ | PY_PROT_WRITE)
                m.write(f"Mapping {i}".encode())
                mappings.append(m)

            # Verify all mappings are independent
            for i, m in enumerate(mappings):
                data = m.read(len(f"Mapping {i}"))
                assert data == f"Mapping {i}".encode()

        finally:
            for m in mappings:
                if m.is_valid:
                    m.close()


class TestEnhancedFlagValidation:
    """Test enhanced flag validation with unknown bit detection."""

    def test_prot_unknown_bits_detected(self):
        """Unknown protection bits should raise ValueError."""

        # Test high bit
        with pytest.raises(ValueError, match="Unknown bits"):
            memmap.py_validate_prot_flags(0x100)

        # Test bit 3
        with pytest.raises(ValueError, match="Unknown bits"):
            memmap.py_validate_prot_flags(8)

        # Test bit 4
        with pytest.raises(ValueError, match="Unknown bits"):
            memmap.py_validate_prot_flags(16)

    def test_prot_negative_rejected(self):
        """Negative protection flags should raise ValueError."""

        with pytest.raises(ValueError, match="negative"):
            memmap.py_validate_prot_flags(-1)

        with pytest.raises(ValueError, match="negative"):
            memmap.py_validate_prot_flags(-100)

    def test_map_unknown_bits_detected(self):
        """Unknown mapping bits should raise ValueError."""

        # Test high bit
        with pytest.raises(ValueError, match="Unknown bits"):
            memmap.py_validate_map_flags(memmap.PY_MAP_PRIVATE | 0x1000)

        # Test multiple unknown bits
        with pytest.raises(ValueError, match="Unknown bits"):
            memmap.py_validate_map_flags(memmap.PY_MAP_SHARED | 0x8000 | 0x10000)

    def test_valid_flags_still_pass(self):
        """Valid flags should still be accepted."""

        # Valid protection flags
        memmap.py_validate_prot_flags(memmap.PY_PROT_NONE)
        memmap.py_validate_prot_flags(memmap.PY_PROT_READ)
        memmap.py_validate_prot_flags(memmap.PY_PROT_READ | memmap.PY_PROT_WRITE)

        # Valid mapping flags
        memmap.py_validate_map_flags(memmap.PY_MAP_SHARED)
        memmap.py_validate_map_flags(memmap.PY_MAP_PRIVATE | memmap.PY_MAP_ANONYMOUS)


class TestThreadSafety:
    """Test thread safety of utility functions."""

    def test_get_page_size_concurrent_access(self):
        """get_page_size() should be thread-safe under concurrent access."""
        import threading

        results = []

        def worker():
            for _ in range(100):
                results.append(memmap.py_get_page_size())

        # Launch 10 threads
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All results should be identical (no races)
        unique_results = set(results)
        assert len(unique_results) == 1, \
            f"get_page_size() returned different values: {unique_results}"

        # Result should be valid
        page_size = results[0]
        assert page_size > 0
        assert page_size in [4096, 8192, 16384, 65536]

    def test_is_page_aligned_concurrent(self):
        """is_page_aligned() should be thread-safe."""
        import threading

        results = []
        page_size = memmap.py_get_page_size()

        def worker():
            for _ in range(100):
                results.append(memmap.py_is_page_aligned(0))
                results.append(memmap.py_is_page_aligned(page_size))
                results.append(memmap.py_is_page_aligned(1))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Count should be deterministic
        true_count = sum(results)
        expected = 10 * 100 * 2  # 10 threads * 100 iterations * 2 aligned values
        assert true_count == expected


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])


# ===========================================================================
# View lifetime: buffer exports keep the region mapped (Run 1)
# ===========================================================================
#
# close() used to munmap unconditionally. An array from as_numpy_array() --
# for instance one returned from a ``with`` block, as the docstring example
# invites -- then pointed at unmapped memory, and reading it killed the
# process with SIGSEGV (exit 139). The mapping now exports its bytes through
# the buffer protocol and defers the unmap until the last export is released.


def _mapped_on_linux(address: int):
    """Whether ``address`` lies in a mapping of this process, or None off Linux."""
    try:
        with open("/proc/self/maps", encoding="ascii") as maps:
            lines = maps.read().splitlines()
    except OSError:
        return None
    for line in lines:
        start, end = (int(part, 16) for part in line.split()[0].split("-"))
        if start <= address < end:
            return True
    return False


def test_lifetime_a_view_returned_from_a_with_block_is_readable():
    """
    The crash reproduction, in a child process so a regression cannot take
    the test session down with it.
    """
    script = textwrap.dedent(
        """
        from scikitplot.memmap import MemoryMap

        def load():
            with MemoryMap.create_anonymous(4096) as m:
                arr = m.as_numpy_array()
                arr[:3] = [7, 8, 9]
                return arr

        arr = load()
        print(arr[:3].tolist())
        """
    )
    done = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, f"exit {done.returncode}; stderr:\n{done.stderr}"
    assert done.stdout.strip().splitlines()[-1] == "[7, 8, 9]"


def test_lifetime_close_ends_the_api_but_not_the_view():
    m = MemoryMap.create_anonymous(4096, PY_PROT_READ | PY_PROT_WRITE)
    arr = m.as_numpy_array()
    arr[:2] = [1, 2]
    m.close()
    assert m.is_valid is False
    with pytest.raises(ValueError, match="closed"):
        m.read(1)
    with pytest.raises(ValueError, match="closed"):
        m.as_numpy_array()
    with pytest.raises(BufferError):
        memoryview(m)
    arr[2] = 3  # still mapped, still writable
    assert arr[:3].tolist() == [1, 2, 3]
    m.close()  # idempotent while the view is alive


def test_lifetime_the_region_is_unmapped_after_the_last_view():
    size = 1 << 20  # large enough to be its own mapping
    m = MemoryMap.create_anonymous(size, PY_PROT_READ | PY_PROT_WRITE)
    address = m.addr
    if _mapped_on_linux(address) is None:
        pytest.skip("needs /proc/self/maps")
    first = m.as_numpy_array()
    second = memoryview(m)
    m.close()
    assert _mapped_on_linux(address) is True  # two exports alive
    del first
    gc.collect()
    assert _mapped_on_linux(address) is True  # one export alive
    second.release()
    assert _mapped_on_linux(address) is False  # unmapped at the last release


def test_lifetime_close_without_views_unmaps_at_once():
    m = MemoryMap.create_anonymous(1 << 20, PY_PROT_READ | PY_PROT_WRITE)
    address = m.addr
    if _mapped_on_linux(address) is None:
        pytest.skip("needs /proc/self/maps")
    m.close()
    assert _mapped_on_linux(address) is False


def test_buffer_protocol_exports_the_mapped_bytes():
    with MemoryMap.create_anonymous(4096, PY_PROT_READ | PY_PROT_WRITE) as m:
        m.write(b"abc")
        view = memoryview(m)
        assert (len(view), view.format, view.itemsize, view.readonly) == (4096, "B", 1, False)
        assert bytes(view[:3]) == b"abc"
        view[3:5] = b"de"
        assert m.read(5) == b"abcde"
        view.release()


def test_buffer_protocol_a_read_only_mapping_exports_read_only():
    with MemoryMap.create_anonymous(4096, PY_PROT_READ) as m:
        assert memoryview(m).readonly is True
        arr = m.as_numpy_array()
        assert arr.flags.writeable is False
        with pytest.raises(ValueError, match="read-only"):
            arr[0] = 1
        del arr


# ===========================================================================
# GIL-free sections hold the region as well (Run 1)
# ===========================================================================
#
# read(), write(), mprotect(), msync(), mlock() and munlock() release the GIL
# around a copy or a system call. close() from another thread used to munmap
# under a running read() (SIGSEGV), and mprotect() could take away a right
# that a live view had been exported with (SIGSEGV on its next access).


def test_lifetime_close_during_a_read_on_another_thread_does_not_crash():
    """The race reproduction, in a child process: it used to exit with -11."""
    script = textwrap.dedent(
        """
        import threading
        import time

        from scikitplot.memmap import MemoryMap

        SIZE = 64 * 1024 * 1024  # a copy long enough to be interrupted
        for _ in range(10):
            m = MemoryMap.create_anonymous(SIZE)
            started = threading.Event()

            def reader():
                started.set()
                try:
                    m.read(SIZE)  # copies with the GIL released
                except ValueError:
                    pass  # closed before the copy began: refused, no crash

            thread = threading.Thread(target=reader)
            thread.start()
            started.wait()
            time.sleep(0.001)
            m.close()
            thread.join()
        print("ok")
        """
    )
    done = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, f"exit {done.returncode}; stderr:\n{done.stderr}"
    assert done.stdout.strip().splitlines()[-1] == "ok"


def test_mprotect_refuses_to_remove_a_right_a_view_holds():
    with MemoryMap.create_anonymous(4096, PY_PROT_READ | PY_PROT_WRITE) as m:
        arr = m.as_numpy_array()
        with pytest.raises(BufferError, match="Release the views"):
            m.mprotect(PY_PROT_READ)
        arr[0] = 1  # still writable: the refusal kept the view valid
        del arr
        gc.collect()
        m.mprotect(PY_PROT_READ)  # no view left: allowed
        with pytest.raises(ValueError, match="not writable"):
            m.write(b"x")


def test_mprotect_may_add_a_right_under_a_view():
    with MemoryMap.create_anonymous(4096, PY_PROT_READ) as m:
        view = memoryview(m)
        m.mprotect(PY_PROT_READ | PY_PROT_WRITE)
        assert view.readonly is True  # exported read-only, stays read-only
        m.write(b"ok")
        assert bytes(view[:2]) == b"ok"
        view.release()


def test_mprotect_and_an_export_on_another_thread_do_not_race():
    """mprotect used to release the GIL before recording the new flags.

    A second thread, woken while the system call ran, exported writable views
    from the stale flags over pages already read-only (writing through one
    killed the process). The child below gates the second thread on the call;
    on the old code it found such a view on the first attempt, three runs out
    of three.
    """
    script = textwrap.dedent(
        """
        import threading
        import time

        from scikitplot.memmap import MemoryMap, PY_PROT_READ, PY_PROT_WRITE

        SIZE = 32 * 1024 * 1024
        m = MemoryMap.create_anonymous(SIZE, PY_PROT_READ | PY_PROT_WRITE)
        block = b"\\x01" * (1 << 20)
        for offset in range(0, SIZE, len(block)):
            m.write(block, offset)  # touch every page: mprotect takes a while

        in_call = threading.Event()
        stop = threading.Event()
        views = []

        def grab():
            while not stop.is_set():
                if in_call.is_set():
                    try:
                        views.append(memoryview(m))
                    except BufferError:
                        pass
                else:
                    time.sleep(0)

        thread = threading.Thread(target=grab)
        thread.start()
        try:
            for attempt in range(200):
                m.mprotect(PY_PROT_READ | PY_PROT_WRITE)
                in_call.set()
                try:
                    m.mprotect(PY_PROT_READ)
                    changed = True
                except BufferError:
                    changed = False  # views were alive: refused, as designed
                in_call.clear()
                if changed and any(not view.readonly for view in list(views)):
                    print(f"attempt {attempt}: writable view over read-only pages")
                    raise SystemExit(3)
                for view in list(views):
                    view.release()
                views.clear()
        finally:
            stop.set()
            thread.join()
        print("ok")
        """
    )
    done = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, (
        f"exit {done.returncode}; stdout: {done.stdout}; stderr:\n{done.stderr}"
    )
    assert done.stdout.strip().splitlines()[-1] == "ok"


def test_an_unreadable_mapping_exports_nothing_and_cannot_be_read():
    """In a child process: before the fix, ``read`` and ``bytes`` faulted."""
    script = textwrap.dedent(
        """
        from scikitplot.memmap import MemoryMap, PY_PROT_NONE, PY_PROT_READ, PY_PROT_WRITE

        def refused(call, error, text):
            try:
                call()
            except error as exc:
                assert text in str(exc), exc
                return
            raise AssertionError(f"{call} did not raise {error.__name__}")

        with MemoryMap.create_anonymous(4096, PY_PROT_NONE) as m:
            refused(lambda: memoryview(m), BufferError, "not readable")
            refused(m.as_numpy_array, BufferError, "not readable")
            refused(lambda: m.read(1), ValueError, "not readable")
        with MemoryMap.create_anonymous(4096, PY_PROT_READ | PY_PROT_WRITE) as m:
            m.mprotect(PY_PROT_NONE)  # no view alive: allowed
            refused(lambda: bytes(m), BufferError, "not readable")
        print("ok")
        """
    )
    done = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, f"exit {done.returncode}; stderr:\n{done.stderr}"
    assert done.stdout.strip().splitlines()[-1] == "ok"
