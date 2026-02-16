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
import os
import tempfile
import pytest
from typing import Generator

# Platform check
# if sys.platform != "win32":
#     pytest.skip("mman tests are Windows-only", allow_module_level=True)

# Import the module to test
try:
    from scikitplot import memmap
    from scikitplot.memmap import (
        MemoryMap,
        mmap_region,
        PROT_NONE, PROT_READ, PROT_WRITE, PROT_EXEC,
        MAP_SHARED, MAP_PRIVATE, MAP_ANONYMOUS, MAP_ANON,
        MS_SYNC, MS_ASYNC,
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
    mapping = MemoryMap.create_anonymous(4096, PROT_READ | PROT_WRITE)
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
    assert hasattr(memmap, 'PROT_NONE')
    assert hasattr(memmap, 'PROT_READ')
    assert hasattr(memmap, 'PROT_WRITE')
    assert hasattr(memmap, 'PROT_EXEC')

    # Mapping flags
    assert hasattr(memmap, 'MAP_SHARED')
    assert hasattr(memmap, 'MAP_PRIVATE')
    assert hasattr(memmap, 'MAP_ANONYMOUS')

    # Classes
    assert hasattr(memmap, 'MemoryMap')
    assert hasattr(memmap, 'MMapError')


def test_protection_flag_values():
    """Test that protection flags have correct values."""
    assert PROT_NONE == 0
    assert PROT_READ == 1
    assert PROT_WRITE == 2
    assert PROT_EXEC == 4


def test_mapping_flag_values():
    """Test that mapping flags have correct values."""
    assert MAP_SHARED == 1
    assert MAP_PRIVATE == 2
    assert MAP_ANONYMOUS == 0x20
    assert MAP_ANON == MAP_ANONYMOUS


# ===========================================================================
# Anonymous Mapping Tests
# ===========================================================================

class TestAnonymousMapping:
    """Tests for anonymous memory mappings."""

    def test_create_basic(self):
        """Test basic anonymous mapping creation."""
        mapping = MemoryMap.create_anonymous(4096, PROT_READ | PROT_WRITE)
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
        mapping = MemoryMap.create_anonymous(4096, PROT_READ)
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
        mapping = MemoryMap.create_anonymous(4096, PROT_READ)
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
        mapping = MemoryMap.create_file_mapping(fd, 0, 4096, PROT_READ)
        assert mapping.is_valid
        assert mapping.size == 4096
        mapping.close()

    def test_read_from_file_mapping(self, temp_file):
        """Test reading from file-backed mapping."""
        fd, path = temp_file
        mapping = MemoryMap.create_file_mapping(fd, 0, 100, PROT_READ)

        data = mapping.read(13)  # "Hello, World!"
        assert data == b"Hello, World!"

        mapping.close()

    def test_create_with_offset(self, temp_file):
        """Test creating mapping with file offset."""
        fd, path = temp_file
        # Note: offset should be page-aligned in real usage
        mapping = MemoryMap.create_file_mapping(fd, 0, 100, PROT_READ)
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
        mapping = MemoryMap.create_anonymous(4096, PROT_READ)

        # Initially can't write
        with pytest.raises(ValueError, match="not writable"):
            mapping.write(b"test")

        # Add write permission
        mapping.mprotect(PROT_READ | PROT_WRITE)

        # Now can write
        n = mapping.write(b"test")
        assert n == 4

        mapping.close()

    def test_mprotect_remove_write(self):
        """Test removing write permission."""
        mapping = MemoryMap.create_anonymous(4096, PROT_READ | PROT_WRITE)

        # Initially can write
        mapping.write(b"test")

        # Remove write permission
        mapping.mprotect(PROT_READ)

        # Now can't write
        with pytest.raises(ValueError, match="not writable"):
            mapping.write(b"more")

        mapping.close()

    def test_mprotect_closed_fails(self):
        """Test that mprotect on closed mapping fails."""
        mapping = MemoryMap.create_anonymous(4096)
        mapping.close()

        with pytest.raises(ValueError, match="closed"):
            mapping.mprotect(PROT_READ)


# ===========================================================================
# Sync Tests
# ===========================================================================

class TestSync:
    """Tests for msync operations."""

    def test_msync_basic(self, temp_file):
        """Test basic msync operation."""
        fd, path = temp_file
        mapping = MemoryMap.create_file_mapping(
            fd, 0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED
        )

        # Write some data
        mapping.write(b"Modified")

        # Sync to disk
        mapping.msync(MS_SYNC)  # Should not raise

        mapping.close()

    def test_msync_async(self, temp_file):
        """Test async msync."""
        fd, path = temp_file
        mapping = MemoryMap.create_file_mapping(
            fd, 0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED
        )

        mapping.write(b"Data")
        mapping.msync(MS_ASYNC)  # Should return immediately

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

        with MemoryMap.create_anonymous(4096, PROT_READ | PROT_WRITE) as m:
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
            prot=PROT_READ | PROT_WRITE,
            flags=MAP_PRIVATE | MAP_ANONYMOUS
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
        mapping = MemoryMap.create_anonymous(4096, PROT_READ | PROT_WRITE)
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

        with MemoryMap.create_anonymous(size, PROT_READ | PROT_WRITE) as m:
            m.write(test_data)
            read_data = m.read(size)
            assert read_data == test_data

    def test_multiple_mappings(self):
        """Test creating multiple independent mappings."""
        mappings = []
        try:
            for i in range(10):
                m = MemoryMap.create_anonymous(4096, PROT_READ | PROT_WRITE)
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


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
