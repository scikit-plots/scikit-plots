#!/usr/bin/env python3
"""
Comprehensive Test Suite for KISS Random Generator with serialization

Tests all features including:
✅ PyKiss32Random / PyKiss64Random with full serialization
✅ PyKissRandom auto-detection
✅ Context managers
✅ Thread safety
✅ KissSeedSequence
✅ KissBitGenerator
✅ KissGenerator
✅ KissRandomState (NEW)
✅ default_rng()
✅ kiss_context()
✅ Pickle support (__getstate__, __setstate__, __reduce__, __reduce_ex__)
✅ State management (get_state, set_state)
✅ Sklearn compatibility (get_params, set_params)
✅ JSON serialization (serialize, deserialize, to_dict, from_dict)
✅ Statistical validation
✅ Reproducibility
"""

import sys
import time
import pickle
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import numpy as np
from scipy import stats

# Import module
try:
    from scikitplot.cexternals._annoy._kissrandom.kissrandom import (
        PyKiss32Random,
        PyKiss64Random,
        PyKissRandom,
        KissSeedSequence,
        KissBitGenerator,
        KissGenerator,
        KissRandomState,
        default_rng,
        kiss_context,
    )
except ImportError as e:
    pytest.skip(f"kissrandom module not built: {e}", allow_module_level=True)

# ===========================================================================
# Test Fixtures
# ===========================================================================

@pytest.fixture
def kiss32():
    """PyKiss32Random with fixed seed."""
    return PyKiss32Random(42)

@pytest.fixture
def kiss64():
    """PyKiss64Random with fixed seed."""
    return PyKiss64Random(42)

@pytest.fixture
def seed_sequence():
    """KissSeedSequence with fixed seed."""
    return KissSeedSequence(42)

@pytest.fixture
def bit_generator():
    """KissBitGenerator with fixed seed."""
    return KissBitGenerator(42)

@pytest.fixture
def generator():
    """KissGenerator with fixed seed."""
    return KissGenerator(42)

@pytest.fixture
def random_state():
    """KissRandomState with fixed seed."""
    return KissRandomState(42)

# ===========================================================================
# Test Serialization - ALL CLASSES
# ===========================================================================

class TestPickleSupport:
    """Test pickle support for all classes."""

    def test_pickle_kiss32_round_trip(self, kiss32):
        """Test PyKiss32Random pickle round-trip."""
        # Generate sequence
        seq1 = [kiss32.kiss() for _ in range(5)]

        # Pickle and unpickle
        pickled = pickle.dumps(kiss32)
        restored = pickle.loads(pickled)

        # Generate sequence from restored
        seq2 = [restored.kiss() for _ in range(5)]

        # Sequences should match
        assert seq1 == seq2

    def test_pickle_kiss64_round_trip(self, kiss64):
        """Test PyKiss64Random pickle round-trip."""
        # Generate sequence
        seq1 = [kiss64.kiss() for _ in range(5)]

        # Pickle and unpickle
        pickled = pickle.dumps(kiss64)
        restored = pickle.loads(pickled)

        # Generate sequence from restored
        seq2 = [restored.kiss() for _ in range(5)]

        # Sequences should match
        assert seq1 == seq2

    def test_pickle_seed_sequence_round_trip(self, seed_sequence):
        """Test KissSeedSequence pickle round-trip."""
        # Get state before
        state1 = seed_sequence.generate_state(5)

        # Pickle and unpickle
        pickled = pickle.dumps(seed_sequence)
        restored = pickle.loads(pickled)

        # Generate state from restored
        state2 = restored.generate_state(5)

        # States should match
        assert np.array_equal(state1, state2)

    def test_pickle_bit_generator_round_trip(self, bit_generator):
        """Test KissBitGenerator pickle round-trip."""
        # Generate sequence
        seq1 = [bit_generator.random_raw() for _ in range(5)]

        # Pickle and unpickle
        pickled = pickle.dumps(bit_generator)
        restored = pickle.loads(pickled)

        # Generate sequence from restored
        seq2 = [restored.random_raw() for _ in range(5)]

        # Sequences should match
        assert seq1 == seq2

    def test_pickle_generator_round_trip(self, generator):
        """Test KissGenerator pickle round-trip."""
        # Generate sequence
        seq1 = generator.random(10)

        # Pickle and unpickle
        pickled = pickle.dumps(generator)
        restored = pickle.loads(pickled)

        # Generate sequence from restored
        seq2 = restored.random(10)

        # Sequences should match
        assert np.allclose(seq1, seq2)

    def test_pickle_random_state_round_trip(self, random_state):
        """Test KissRandomState pickle round-trip."""
        # Generate sequence
        seq1 = random_state.rand(10)

        # Pickle and unpickle
        pickled = pickle.dumps(random_state)
        restored = pickle.loads(pickled)

        # Generate sequence from restored
        seq2 = restored.rand(10)

        # Sequences should match
        assert np.allclose(seq1, seq2)

    def test_pickle_all_protocols(self):
        """Test all classes with all pickle protocols."""
        objects = [
            PyKiss32Random(42),
            PyKiss64Random(42),
            KissSeedSequence(42),
            KissBitGenerator(42),
            KissGenerator(42),
            KissRandomState(42),
        ]

        for obj in objects:
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                pickled = pickle.dumps(obj, protocol=protocol)
                restored = pickle.loads(pickled)
                assert type(restored) is type(obj)


class TestStateManagement:
    """Test get_state/set_state for all classes."""

    def test_kiss32_get_set_state(self):
        """Test PyKiss32Random state management."""
        rng1 = PyKiss32Random(42)
        seq1 = [rng1.kiss() for _ in range(5)]

        # Get state
        state = rng1.get_state()
        assert isinstance(state, dict)
        assert "seed" in state
        assert "__version__" in state

        # Create new RNG and set state
        rng2 = PyKiss32Random(0)
        rng2.set_state(state)

        # Generate same sequence
        seq2 = [rng2.kiss() for _ in range(5)]
        assert seq1 == seq2

    def test_kiss64_get_set_state(self):
        """Test PyKiss64Random state management."""
        rng1 = PyKiss64Random(42)
        seq1 = [rng1.kiss() for _ in range(5)]

        # Get state
        state = rng1.get_state()
        assert isinstance(state, dict)
        assert "seed" in state
        assert "__version__" in state

        # Create new RNG and set state
        rng2 = PyKiss64Random(0)
        rng2.set_state(state)

        # Generate same sequence
        seq2 = [rng2.kiss() for _ in range(5)]
        assert seq1 == seq2

    def test_seed_sequence_get_set_state(self):
        """Test KissSeedSequence state management."""
        seq1 = KissSeedSequence(42)
        state1 = seq1.generate_state(5)

        # Get state
        state = seq1.get_state()
        assert isinstance(state, dict)
        assert "entropy" in state

        # Create new and set state
        seq2 = KissSeedSequence(0)
        seq2.set_state(state)

        # Generate same state
        state2 = seq2.generate_state(5)
        assert np.array_equal(state1, state2)

    def test_bit_generator_get_set_state(self):
        """Test KissBitGenerator state management."""
        bg1 = KissBitGenerator(42)
        seq1 = [bg1.random_raw() for _ in range(5)]

        # Get state
        state = bg1.get_state()
        assert isinstance(state, dict)
        assert "seed_sequence" in state
        assert "seed_sequence_state" in state

        # Create new and set state
        bg2 = KissBitGenerator(0)
        bg2.set_state(state)

        # Generate same sequence
        seq2 = [bg2.random_raw() for _ in range(5)]
        assert seq1 == seq2

    def test_generator_get_set_state(self):
        """Test KissGenerator state management."""
        gen1 = KissGenerator(42)
        seq1 = gen1.random(10)

        # Get state
        state = gen1.get_state()
        assert isinstance(state, dict)
        assert "bit_generator" in state
        assert "bit_generator_state" in state

        # Create new and set state
        gen2 = KissGenerator(0)
        gen2.set_state(state)

        # Generate same sequence
        seq2 = gen2.random(10)
        assert np.allclose(seq1, seq2)

    def test_random_state_get_set_state(self):
        """Test KissRandomState state management."""
        rs1 = KissRandomState(42)
        seq1 = rs1.rand(10)

        # Get state
        state = rs1.get_state()
        assert isinstance(state, dict)

        # Create new and set state
        rs2 = KissRandomState(0)
        rs2.set_state(state)

        # Generate same sequence
        seq2 = rs2.rand(10)
        assert np.allclose(seq1, seq2)


class TestSklearnCompatibility:
    """Test get_params/set_params for sklearn compatibility."""

    def test_kiss32_get_set_params(self):
        """Test PyKiss32Random sklearn compatibility."""
        rng = PyKiss32Random(42)

        # Get params
        params = rng.get_params()
        assert isinstance(params, dict)
        assert "seed" in params
        assert params["seed"] == 42

        # Set params
        rng.set_params(seed=123)
        assert rng.seed == 123

    def test_kiss64_get_set_params(self):
        """Test PyKiss64Random sklearn compatibility."""
        rng = PyKiss64Random(42)

        # Get params
        params = rng.get_params()
        assert isinstance(params, dict)
        assert "seed" in params
        assert params["seed"] == 42

        # Set params
        rng.set_params(seed=123)
        assert rng.seed == 123

    def test_seed_sequence_get_set_params(self):
        """Test KissSeedSequence sklearn compatibility."""
        seq = KissSeedSequence(42)

        # Get params
        params = seq.get_params()
        assert isinstance(params, dict)
        assert "entropy" in params

        # Set params
        seq.set_params(entropy=123)
        assert seq.entropy == 123

    def test_bit_generator_get_set_params(self):
        """Test KissBitGenerator sklearn compatibility."""
        bg = KissBitGenerator(42)

        # Get params
        params = bg.get_params(deep=True)
        assert isinstance(params, dict)
        assert "seed" in params

    def test_generator_get_set_params(self):
        """Test KissGenerator sklearn compatibility."""
        gen = KissGenerator(42)

        # Get params
        params = gen.get_params(deep=True)
        assert isinstance(params, dict)

    def test_random_state_get_set_params(self):
        """Test KissRandomState sklearn compatibility."""
        rs = KissRandomState(42)

        # Get params (inherits from KissGenerator)
        params = rs.get_params(deep=True)
        assert isinstance(params, dict)


class TestJSONSerialization:
    """Test serialize/deserialize for JSON compatibility."""

    def test_kiss32_json_round_trip(self):
        """Test PyKiss32Random JSON serialization."""
        rng1 = PyKiss32Random(42)
        seq1 = [rng1.kiss() for _ in range(5)]

        # Serialize to JSON
        data = rng1.serialize()
        json_str = json.dumps(data)

        # Deserialize
        loaded_data = json.loads(json_str)
        rng2 = PyKiss32Random.deserialize(loaded_data)

        # Generate same sequence
        seq2 = [rng2.kiss() for _ in range(5)]
        assert seq1 == seq2

    def test_kiss64_json_round_trip(self):
        """Test PyKiss64Random JSON serialization."""
        rng1 = PyKiss64Random(42)
        seq1 = [rng1.kiss() for _ in range(5)]

        # Serialize to JSON
        data = rng1.serialize()
        json_str = json.dumps(data)

        # Deserialize
        loaded_data = json.loads(json_str)
        rng2 = PyKiss64Random.deserialize(loaded_data)

        # Generate same sequence
        seq2 = [rng2.kiss() for _ in range(5)]
        assert seq1 == seq2

    def test_seed_sequence_json_round_trip(self):
        """Test KissSeedSequence JSON serialization."""
        seq1 = KissSeedSequence(42)
        state1 = seq1.generate_state(5)

        # Serialize to JSON
        data = seq1.serialize()
        json_str = json.dumps(data)

        # Deserialize
        loaded_data = json.loads(json_str)
        seq2 = KissSeedSequence.deserialize(loaded_data)

        # Generate same state
        state2 = seq2.generate_state(5)
        assert np.array_equal(state1, state2)

    # @pytest.mark.xfail(reason="Test data unavailable.")
    def test_bit_generator_json_round_trip(self):
        """Test KissBitGenerator JSON serialization."""
        bg1 = KissBitGenerator(42)
        seq1 = [bg1.random_raw() for _ in range(5)]

        # Serialize to JSON
        data = bg1.serialize()
        json_str = json.dumps(data)

        # Deserialize
        # with pytest.raises(Exception) as e:
        loaded_data = json.loads(json_str)
        bg2 = KissBitGenerator.deserialize(loaded_data)

        # Generate same sequence
        seq2 = [bg2.random_raw() for _ in range(5)]
        assert seq1 == seq2

    def test_generator_json_round_trip(self):
        """Test KissGenerator JSON serialization."""
        gen1 = KissGenerator(42)
        seq1 = gen1.random(10)

        # Serialize to JSON
        data = gen1.serialize()
        json_str = json.dumps(data)

        # Deserialize
        loaded_data = json.loads(json_str)
        gen2 = KissGenerator.deserialize(loaded_data)

        # Generate same sequence
        seq2 = gen2.random(10)
        assert np.allclose(seq1, seq2)

    def test_random_state_json_round_trip(self):
        """Test KissRandomState JSON serialization (inherits)."""
        rs1 = KissRandomState(42)
        seq1 = rs1.rand(10)

        # Serialize to JSON
        data = rs1.serialize()
        json_str = json.dumps(data)

        # Deserialize
        loaded_data = json.loads(json_str)
        rs2 = KissRandomState.deserialize(loaded_data)

        # Generate same sequence
        seq2 = rs2.rand(10)
        assert np.allclose(seq1, seq2)

    def test_to_dict_from_dict_aliases(self):
        """Test to_dict/from_dict aliases."""
        gen1 = KissGenerator(42)

        # to_dict is alias for serialize
        data1 = gen1.to_dict()
        data2 = gen1.serialize()
        assert data1 == data2

        # from_dict is alias for deserialize
        gen2 = KissGenerator.from_dict(data1)
        assert isinstance(gen2, KissGenerator)


class TestMetadata:
    """Test metadata inclusion in serialization."""

    def test_metadata_in_serialization(self):
        """Test that metadata is included in serialize()."""
        gen = KissGenerator(42)
        data = gen.serialize()

        # Check metadata
        assert "__version__" in data
        assert "metadata" in data
        assert "python_version" in data["metadata"]
        assert "numpy_version" in data["metadata"]
        assert "platform" in data["metadata"]

    def test_class_and_module_in_serialization(self):
        """Test that __class__ and __module__ are included."""
        classes = [
            (PyKiss32Random(42), PyKiss32Random(42).__str__()),
            (PyKiss64Random(42), PyKiss64Random(42).__str__()),
            (KissSeedSequence(42), KissSeedSequence(42).__str__()),
            (KissBitGenerator(42), KissBitGenerator(42).__str__()),
            (KissGenerator(42), KissGenerator(42).__str__()),
            (KissRandomState(42), KissGenerator(42).__str__()),  # inherited
        ]

        for obj, class_name in classes:
            data = obj.serialize()
            assert "__class__" in data
            assert data["__class__"] == class_name
            assert "__module__" in data


# ===========================================================================
# Test PyKiss32Random / PyKiss64Random
# ===========================================================================

class TestLowLevelAPI:
    """Test low-level PyKiss32Random and PyKiss64Random."""

    def test_kiss32_basic(self, kiss32):
        """Test PyKiss32Random basic functionality."""
        # kiss() returns uint32
        value = kiss32.kiss()
        assert isinstance(value, int)
        assert 0 <= value <= 0xFFFFFFFF

        # flip() returns 0 or 1
        flip_val = kiss32.flip()
        assert flip_val in (0, 1)

        # index(n) returns value in [0, n-1]
        idx = kiss32.index(100)
        assert 0 <= idx < 100

    def test_kiss64_basic(self, kiss64):
        """Test PyKiss64Random basic functionality."""
        # kiss() returns uint64
        value = kiss64.kiss()
        assert isinstance(value, int)
        assert 0 <= value <= 0xFFFFFFFFFFFFFFFF

        # flip() returns 0 or 1
        flip_val = kiss64.flip()
        assert flip_val in (0, 1)

        # index(n) returns value in [0, n-1]
        idx = kiss64.index(100)
        assert 0 <= idx < 100

    def test_kiss32_seed_property(self):
        """Test PyKiss32Random seed property."""
        rng = PyKiss32Random(42)
        assert rng.seed == 42

        # Change seed
        rng.seed = 123
        assert rng.seed == 123

    def test_kiss64_seed_property(self):
        """Test PyKiss64Random seed property."""
        rng = PyKiss64Random(42)
        assert rng.seed == 42

        # Change seed
        rng.seed = 123
        assert rng.seed == 123

    def test_kiss32_reproducibility(self):
        """Test PyKiss32Random reproducibility."""
        rng1 = PyKiss32Random(42)
        rng2 = PyKiss32Random(42)

        # Same seed → same sequence
        seq1 = [rng1.kiss() for _ in range(10)]
        seq2 = [rng2.kiss() for _ in range(10)]

        assert seq1 == seq2

    def test_kiss64_reproducibility(self):
        """Test PyKiss64Random reproducibility."""
        rng1 = PyKiss64Random(42)
        rng2 = PyKiss64Random(42)

        # Same seed → same sequence
        seq1 = [rng1.kiss() for _ in range(10)]
        seq2 = [rng2.kiss() for _ in range(10)]

        assert seq1 == seq2

    def test_kiss32_context_manager(self):
        """Test PyKiss32Random context manager."""
        rng = PyKiss32Random(42)

        # Should not hang
        start = time.time()
        with rng:
            value = rng.kiss()
        elapsed = time.time() - start

        assert elapsed < 0.1  # Should be instant
        assert isinstance(value, int)

    def test_kiss64_context_manager(self):
        """Test PyKiss64Random context manager."""
        rng = PyKiss64Random(42)

        # Should not hang
        start = time.time()
        with rng:
            value = rng.kiss()
        elapsed = time.time() - start

        assert elapsed < 0.1  # Should be instant
        assert isinstance(value, int)

    def test_kiss64_nested_context_managers(self):
        """Test nested context managers (no infinite loop)."""
        start = time.time()

        with PyKiss64Random(100) as rng1:
            with PyKiss64Random(200) as rng2:
                val1 = rng1.kiss()
                val2 = rng2.kiss()

        elapsed = time.time() - start

        assert elapsed < 0.1  # Should be instant
        assert val1 != val2  # Different seeds

# ===========================================================================
# Test PyKissRandom Auto-Detection
# ===========================================================================

class TestAutoDetection:
    """Test PyKissRandom factory function."""

    def test_auto_detection(self):
        """Test auto bit width detection."""
        rng = PyKissRandom(42)

        # Should return one of the types
        assert isinstance(rng, (PyKiss32Random, PyKiss64Random))

    def test_explicit_32bit(self):
        """Test explicit 32-bit selection."""
        rng = PyKissRandom(42, bit_width=32)
        assert isinstance(rng, PyKiss32Random)

    def test_explicit_64bit(self):
        """Test explicit 64-bit selection."""
        rng = PyKissRandom(42, bit_width=64)
        assert isinstance(rng, PyKiss64Random)

    def test_none_bit_width(self):
        """Test bit_width=None (auto-detect)."""
        rng = PyKissRandom(42, bit_width=None)
        assert isinstance(rng, (PyKiss32Random, PyKiss64Random))

    def test_auto_string(self):
        """Test bit_width='auto'."""
        rng = PyKissRandom(42, bit_width='auto')
        assert isinstance(rng, (PyKiss32Random, PyKiss64Random))

# ===========================================================================
# Test KissSeedSequence
# ===========================================================================

class TestKissSeedSequence:
    """Test KissSeedSequence (NumPy SeedSequence compatible)."""

    def test_basic_creation(self):
        """Test basic KissSeedSequence creation."""
        seq = KissSeedSequence(42)
        assert seq.entropy == 42

    def test_none_entropy(self):
        """Test KissSeedSequence with None (OS entropy)."""
        seq = KissSeedSequence()
        assert seq.entropy is not None
        assert isinstance(seq.entropy, int)

    def test_large_int_entropy(self):
        """Test KissSeedSequence with large int (no overflow)."""
        # This was causing OverflowError before
        huge_int = 2**128
        seq = KissSeedSequence(huge_int)

        # Should mask to 64 bits
        assert seq.entropy is not None
        assert isinstance(seq.entropy, int)

    def test_sequence_entropy(self):
        """Test KissSeedSequence with sequence."""
        seq = KissSeedSequence([1, 2, 3, 4, 5])
        assert seq.entropy is not None
        assert isinstance(seq.entropy, int)

    def test_generate_state_uint32(self):
        """Test generate_state with uint32."""
        seq = KissSeedSequence(42)
        state = seq.generate_state(4, dtype=np.uint32)

        assert state.dtype == np.uint32
        assert state.shape == (4,)

    def test_generate_state_uint64(self):
        """Test generate_state with uint64."""
        seq = KissSeedSequence(42)
        state = seq.generate_state(2, dtype=np.uint64)

        assert state.dtype == np.uint64
        assert state.shape == (2,)

    def test_spawn(self):
        """Test KissSeedSequence spawn."""
        seq = KissSeedSequence(42)
        children = seq.spawn(3)

        assert len(children) == 3
        assert all(isinstance(child, KissSeedSequence) for child in children)
        assert seq.n_children_spawned == 3

    def test_state_property(self):
        """Test KissSeedSequence state property."""
        seq = KissSeedSequence(42)
        state = seq.state

        assert isinstance(state, dict)
        assert 'entropy' in state
        assert state['entropy'] == 42

# ===========================================================================
# Test KissBitGenerator
# ===========================================================================

class TestKissBitGenerator:
    """Test KissBitGenerator (NumPy BitGenerator protocol)."""

    def test_basic_creation(self, bit_generator):
        """Test basic BitGenerator creation."""
        assert isinstance(bit_generator, KissBitGenerator)
        assert hasattr(bit_generator, 'lock')

    def test_random_raw_scalar(self, bit_generator):
        """Test random_raw() returns scalar."""
        value = bit_generator.random_raw()

        assert isinstance(value, (int, np.integer))
        assert 0 <= value <= 0xFFFFFFFFFFFFFFFF

    def test_random_raw_array(self, bit_generator):
        """Test random_raw(size) returns array."""
        arr = bit_generator.random_raw(5)

        assert isinstance(arr, np.ndarray)
        assert arr.shape == (5,)
        assert arr.dtype == np.uint64

    def test_random_raw_2d(self, bit_generator):
        """Test random_raw with 2D shape."""
        arr = bit_generator.random_raw((3, 4))

        assert arr.shape == (3, 4)
        assert arr.dtype == np.uint64

    def test_state_property(self, bit_generator):
        """Test BitGenerator state property."""
        state = bit_generator.state

        assert isinstance(state, dict)
        assert state['seed_sequence'] == 'KissSeedSequence'
        assert 'seed_sequence_state' in state

    def test_state_setter(self):
        """Test BitGenerator state setter."""
        bg1 = KissBitGenerator(42)
        seq1 = [bg1.random_raw() for _ in range(5)]

        state = bg1.state

        # Continue generating
        bg1.random_raw()
        bg1.random_raw()

        # Restore state
        bg2 = KissBitGenerator(0)
        bg2.state = state

        seq2 = [bg2.random_raw() for _ in range(5)]

        assert seq1 == seq2

    def test_setstate_method(self):
        """Test __setstate__ method."""
        bg1 = KissBitGenerator(42)
        state = bg1.state

        bg2 = KissBitGenerator()
        bg2.__setstate__(state)

        # Should return self
        assert isinstance(bg2, KissBitGenerator)

    def test_spawn(self, bit_generator):
        """Test BitGenerator spawn."""
        children = bit_generator.spawn(3)

        assert len(children) == 3
        assert all(isinstance(child, KissBitGenerator) for child in children)

    def test_reproducibility(self):
        """Test BitGenerator reproducibility."""
        bg1 = KissBitGenerator(42)
        bg2 = KissBitGenerator(42)

        seq1 = [bg1.random_raw() for _ in range(10)]
        seq2 = [bg2.random_raw() for _ in range(10)]

        assert seq1 == seq2

# ===========================================================================
# Test KissGenerator
# ===========================================================================

class TestKissGenerator:
    """Test KissGenerator (high-level API)."""

    def test_basic_creation(self, generator):
        """Test basic Generator creation."""
        assert isinstance(generator, KissGenerator)
        assert hasattr(generator, 'bit_generator')

    def test_random_scalar(self, generator):
        """Test random() returns scalar."""
        value = generator.random()

        assert isinstance(value, (float, np.floating))
        assert 0.0 <= value < 1.0

    def test_random_array(self, generator):
        """Test random(size) returns array."""
        arr = generator.random(5)

        assert isinstance(arr, np.ndarray)
        assert arr.shape == (5,)
        assert np.all((arr >= 0.0) & (arr < 1.0))

    def test_integers(self, generator):
        """Test integers method."""
        ints = generator.integers(0, 100, size=10)

        assert isinstance(ints, np.ndarray)
        assert ints.shape == (10,)
        assert np.all((ints >= 0) & (ints < 100))

    def test_integers_endpoint(self, generator):
        """Test integers with endpoint=True."""
        ints = generator.integers(1, 7, size=100, endpoint=True)

        # Should include 7
        assert ints.min() >= 1
        assert ints.max() <= 7

    def test_normal(self, generator):
        """Test normal distribution."""
        samples = generator.normal(0, 1, size=1000)

        assert isinstance(samples, np.ndarray)
        assert samples.shape == (1000,)

        # Check mean and std (rough check)
        assert abs(samples.mean()) < 0.2
        assert abs(samples.std() - 1.0) < 0.2

    def test_uniform(self, generator):
        """Test uniform distribution."""
        samples = generator.uniform(10, 20, size=100)

        assert np.all((samples >= 10) & (samples < 20))

    def test_choice(self, generator):
        """Test choice method."""
        arr = ['A', 'B', 'C']
        choices = generator.choice(arr, size=10)

        assert len(choices) == 10
        assert all(c in arr for c in choices)

    def test_choice_weighted(self, generator):
        """Test choice with probabilities."""
        choices = generator.choice(['A', 'B', 'C'], size=1000, p=[0.5, 0.3, 0.2])

        # Count A's (should be ~50%)
        count_a = list(choices).count('A')
        assert 400 < count_a < 600  # Rough check

    def test_shuffle(self, generator):
        """Test shuffle method."""
        arr = np.arange(10)
        original = arr.copy()

        generator.shuffle(arr)

        # Should be permutation
        assert set(arr) == set(original)
        # Should be different order (with high probability)
        assert not np.array_equal(arr, original)

    def test_context_manager(self):
        """Test KissGenerator context manager."""
        gen = KissGenerator(42)

        start = time.time()
        with gen:
            data = gen.random(100)
        elapsed = time.time() - start

        assert elapsed < 0.1
        assert len(data) == 100

    def test_get_bit_generator(self, generator):
        """Test get_bit_generator method."""
        bg = generator.get_bit_generator()

        assert isinstance(bg, KissBitGenerator)

    def test_set_bit_generator(self):
        """Test set_bit_generator method."""
        gen = KissGenerator(42)

        # Create new BitGenerator
        new_bg = KissBitGenerator(123)

        # Set it
        gen.set_bit_generator(new_bg)

        # Verify
        assert gen.bit_generator is new_bg

    def test_spawn(self, generator):
        """Test Generator spawn."""
        children = generator.spawn(3)

        assert len(children) == 3
        assert all(isinstance(child, KissGenerator) for child in children)

# ===========================================================================
# Test KissRandomState (NEW)
# ===========================================================================

class TestKissRandomState:
    """Test KissRandomState (NumPy RandomState compatible)."""

    def test_basic_creation(self, random_state):
        """Test basic RandomState creation."""
        assert isinstance(random_state, KissRandomState)

    def test_rand_scalar(self, random_state):
        """Test rand() returns scalar."""
        value = random_state.rand()

        assert isinstance(value, (float, np.floating))
        assert 0.0 <= value < 1.0

    def test_rand_array(self, random_state):
        """Test rand(*shape) returns array."""
        arr = random_state.rand(3, 4)

        assert arr.shape == (3, 4)
        assert np.all((arr >= 0.0) & (arr < 1.0))

    def test_randn_scalar(self, random_state):
        """Test randn() returns scalar."""
        value = random_state.randn()

        assert isinstance(value, (float, np.floating))

    def test_randn_array(self, random_state):
        """Test randn(*shape) returns array."""
        arr = random_state.randn(3, 4)

        assert arr.shape == (3, 4)
        # Rough check for normality
        assert abs(arr.mean()) < 0.5

    def test_randint(self, random_state):
        """Test randint method."""
        ints = random_state.randint(0, 10, size=100)

        assert np.all((ints >= 0) & (ints < 10))

    def test_random_sample(self, random_state):
        """Test random_sample method."""
        samples = random_state.random_sample(size=10)

        assert len(samples) == 10
        assert np.all((samples >= 0.0) & (samples < 1.0))

    def test_seed_method(self):
        """Test seed() method."""
        rs = KissRandomState()

        rs.seed(42)
        seq1 = rs.rand(5)

        rs.seed(42)
        seq2 = rs.rand(5)

        assert np.allclose(seq1, seq2)

    def test_get_state(self, random_state):
        """Test get_state method."""
        state = random_state.get_state()

        assert isinstance(state, dict)
        assert 'bit_generator' in state

    def test_set_state(self):
        """Test set_state method."""
        rs1 = KissRandomState(42)
        seq1 = [rs1.rand() for _ in range(5)]

        state = rs1.get_state()

        rs2 = KissRandomState(0)
        rs2.set_state(state)

        seq2 = [rs2.rand() for _ in range(5)]

        assert np.allclose(seq1, seq2)

    def test_get_bit_generator(self, random_state):
        """Test get_bit_generator method."""
        bg = random_state.get_bit_generator()

        assert isinstance(bg, KissBitGenerator)

    def test_set_bit_generator(self):
        """Test set_bit_generator method."""
        rs = KissRandomState(42)

        new_bg = KissBitGenerator(123)
        rs.set_bit_generator(new_bg)

        assert rs.bit_generator is new_bg

# ===========================================================================
# Test default_rng()
# ===========================================================================

class TestDefaultRNG:
    """Test default_rng() convenience function."""

    def test_basic_usage(self):
        """Test basic default_rng() usage."""
        rng = default_rng(42)

        assert isinstance(rng, KissGenerator)

    def test_none_seed(self):
        """Test default_rng(None)."""
        rng = default_rng()

        assert isinstance(rng, KissGenerator)

    def test_with_seed_sequence(self):
        """Test default_rng with SeedSequence."""
        seq = KissSeedSequence(42)
        rng = default_rng(seq)

        assert isinstance(rng, KissGenerator)

    def test_with_bit_generator(self):
        """Test default_rng with BitGenerator."""
        bg = KissBitGenerator(42)
        rng = default_rng(bg)

        assert isinstance(rng, KissGenerator)

    def test_with_generator(self):
        """Test default_rng with Generator."""
        gen = KissGenerator(42)
        rng = default_rng(gen)

        assert rng is gen

# ===========================================================================
# Test kiss_context()
# ===========================================================================

class TestKissContext:
    """Test kiss_context() context manager helper."""

    def test_basic_usage(self):
        """Test basic kiss_context usage."""
        with kiss_context(42) as rng:
            data = rng.random(10)

        assert len(data) == 10
        assert isinstance(rng, KissGenerator)

    def test_cleanup(self):
        """Test context manager cleanup."""
        start = time.time()

        with kiss_context(42) as rng:
            _ = rng.random(100)

        elapsed = time.time() - start
        assert elapsed < 0.1  # Should be instant

# ===========================================================================
# Test Thread Safety
# ===========================================================================

class TestThreadSafety:
    """Test thread-safe operations."""

    def test_shared_rng_with_lock(self):
        """Test shared RNG with lock (like plot_kissrandom.py)."""
        shared_rng = PyKiss64Random(42)

        def worker(task_id):
            with shared_rng:
                return [shared_rng.kiss() for _ in range(3)]

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(worker, range(4)))

        # Check all values unique
        all_values = [v for r in results for v in r]
        unique_values = set(all_values)

        assert len(unique_values) == len(all_values)

    def test_generator_thread_safety(self):
        """Test KissGenerator thread safety."""
        gen = KissGenerator(42)

        def worker(task_id):
            with gen:
                return gen.random(10).tolist()

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(worker, range(2)))

        # Should have different values
        assert not np.allclose(results[0], results[1])

# ===========================================================================
# Test Statistical Properties
# ===========================================================================

class TestStatisticalProperties:
    """Test statistical properties of generated numbers."""

    def test_uniform_distribution(self):
        """Test uniform distribution (like plot_kissrandom.py)."""
        rng = default_rng(42)
        samples = rng.random(100000)

        # Check mean and std
        assert abs(samples.mean() - 0.5) < 0.01
        assert abs(samples.std() - 0.288675) < 0.01

    def test_chi_square_uniformity(self):
        """Test chi-square uniformity (like plot_kissrandom.py)."""
        rng = default_rng(42)
        samples = rng.random(100000)

        # Chi-square test
        observed, _ = np.histogram(samples, bins=20, range=(0, 1))
        expected = np.full(20, len(samples) / 20)
        chi2, p_value = stats.chisquare(observed, expected)

        # Should pass (p > 0.01)
        assert p_value > 0.01

    def test_normal_distribution(self):
        """Test normal distribution properties."""
        rng = default_rng(42)
        samples = rng.normal(0, 1, size=10000)

        # Check mean and std
        assert abs(samples.mean()) < 0.1
        assert abs(samples.std() - 1.0) < 0.1

# ===========================================================================
# Test Reproducibility
# ===========================================================================

class TestReproducibility:
    """Test reproducibility and determinism."""

    def test_same_seed_same_sequence(self):
        """Test same seed produces same sequence (like plot_kissrandom.py)."""
        seeds = [42, 123, 999]

        for seed in seeds:
            rng1 = default_rng(seed)
            seq1 = rng1.random(10)

            rng2 = default_rng(seed)
            seq2 = rng2.random(10)

            assert np.allclose(seq1, seq2)

    def test_state_serialization_reproducibility(self):
        """Test state serialization/deserialization."""
        rng1 = KissBitGenerator(42)
        sequence1 = [rng1.random_raw() for _ in range(5)]

        state = rng1.state

        # Continue generating
        rng1.random_raw()
        rng1.random_raw()

        # Restore state
        rng2 = KissBitGenerator()
        rng2.state = state

        sequence2 = [rng2.random_raw() for _ in range(5)]

        assert sequence1 == sequence2

# ===========================================================================
# Test Edge Cases
# ===========================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_index_zero(self):
        """Test index(0) returns 0."""
        rng = PyKiss64Random(42)
        assert rng.index(0) == 0

    def test_index_one(self):
        """Test index(1) returns 0."""
        rng = PyKiss64Random(42)
        assert rng.index(1) == 0

    def test_large_seed_no_overflow(self):
        """Test large seed doesn't cause overflow."""
        # This was causing OverflowError before
        huge_seed = 2**100
        seq = KissSeedSequence(huge_seed)

        # Should work without error
        assert seq.entropy is not None

    def test_invalid_bit_width(self):
        """Test invalid bit_width raises error."""
        with pytest.raises(ValueError):
            PyKissRandom(42, bit_width=16)

    def test_choice_without_replacement(self):
        """Test choice without replacement."""
        rng = default_rng(42)

        with pytest.raises(ValueError):
            rng.choice(10, size=20, replace=False)

# ===========================================================================
# Test Extensibility
# ===========================================================================

class TestExtensibility:
    """Test custom distributions (like plot_kissrandom.py)."""

    def test_custom_distribution(self):
        """Test extending KissGenerator with custom methods."""
        class ExtendedKissGenerator(KissGenerator):
            def exponential(self, scale=1.0, size=None):
                u = self.random(size)
                return -scale * np.log(1 - u)

        gen = ExtendedKissGenerator(42)
        samples = gen.exponential(2.0, size=100)

        # Check mean (should be ~2.0)
        assert 1.5 < samples.mean() < 2.5

# ===========================================================================
# Run Tests
# ===========================================================================

if __name__ == '__main__':
    # Run with: python test_kissrandom_comprehensive.py
    pytest.main([__file__, '-v'])
