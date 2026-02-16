
"""
Comprehensive Test Suite for Complete Annoy Implementation
===========================================================

Tests all implemented methods systematically.
"""

import random
import sys
import json
import pickle
import tempfile
import os
import math
import warnings

from ..annoylib import Index, BaseIndex, ReprHTMLMixin, _HTMLDocumentationLinkMixin

def test_initialization():
    """Test 1: Initialization with all metrics"""
    print("\n" + "="*70)
    print("TEST 1: Initialization")
    print("="*70)

    metrics = ['angular', 'euclidean', 'manhattan', 'dot', 'hamming']

    for metric in metrics:
        try:
            index = Index(f=10, metric=metric, seed=42)
            print(f"✓ {metric:12s}: {repr(index)}")
        except Exception as e:
            print(f"✗ {metric:12s}: {e}")


def test_add_and_build():
    """Test 2: Add items and build"""
    print("\n" + "="*70)
    print("TEST 2: Add Items and Build")
    print("="*70)

    random.seed(42)
    index = Index(f=5, metric='angular', seed=42)

    # Add items
    for i in range(10):
        vec = [random.random() for _ in range(5)]
        index.add_item(i, vec)

    print(f"✓ Added 10 items: n_items={index.get_n_items()}")

    # Build
    index.build(n_trees=5)
    print(f"✓ Built index: n_trees={index.get_n_trees()}")


def test_query_by_item():
    """Test 3: Query by item"""
    print("\n" + "="*70)
    print("TEST 3: Query by Item")
    print("="*70)

    random.seed(42)
    index = Index(f=5, metric='angular', seed=42)

    # Add items
    for i in range(20):
        vec = [random.random() for _ in range(5)]
        index.add_item(i, vec)

    index.build(n_trees=10)

    # Query without distances
    neighbors = index.get_nns_by_item(0, n=5)
    print(f"✓ Query item 0, n=5: {neighbors}")
    assert len(neighbors) == 5, "Expected 5 neighbors"
    assert neighbors[0] == 0, "Item should be nearest to itself"

    # Query with distances
    neighbors, distances = index.get_nns_by_item(0, n=5, include_distances=True)
    print(f"✓ With distances: neighbors={neighbors}")
    print(f"  Distances: {['%.4f' % d for d in distances]}")
    assert len(neighbors) == len(distances), "Neighbors and distances must match"


def test_query_by_vector():
    """Test 4: Query by vector"""
    print("\n" + "="*70)
    print("TEST 4: Query by Vector")
    print("="*70)

    random.seed(42)
    index = Index(f=5, metric='euclidean', seed=42)

    # Add items
    vectors = []
    for i in range(20):
        vec = [random.random() for _ in range(5)]
        vectors.append(vec)
        index.add_item(i, vec)

    index.build(n_trees=10)

    # Query with first vector
    query_vec = vectors[0]
    neighbors, distances = index.get_nns_by_vector(
        query_vec, n=3, include_distances=True
    )

    print(f"✓ Query vector, n=3: {neighbors}")
    print(f"  Distances: {['%.4f' % d for d in distances]}")
    assert 0 in neighbors, "First item should be in results"
    assert distances[neighbors.index(0)] < 0.01, "Distance to self should be ~0"


def test_get_item():
    """Test 5: Retrieve stored items"""
    print("\n" + "="*70)
    print("TEST 5: Get Item")
    print("="*70)

    index = Index(f=5, metric='angular')

    # Add known vectors
    vec1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    vec2 = [5.0, 4.0, 3.0, 2.0, 1.0]

    index.add_item(0, vec1)
    index.add_item(1, vec2)

    # Retrieve
    retrieved1 = index.get_item(0)
    retrieved2 = index.get_item(1)

    print(f"✓ Item 0: {['%.2f' % v for v in retrieved1]}")
    print(f"✓ Item 1: {['%.2f' % v for v in retrieved2]}")

    # Check approximate equality (may have small floating point errors)
    for i in range(5):
        assert abs(retrieved1[i] - vec1[i]) < 0.001, f"Vector 1 mismatch at {i}"


def test_distances():
    """Test 6: Distance calculations"""
    print("\n" + "="*70)
    print("TEST 6: Distances")
    print("="*70)

    index = Index(f=3, metric='euclidean')

    # Add known vectors
    vec1 = [0.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    vec3 = [0.0, 1.0, 0.0]

    index.add_item(0, vec1)
    index.add_item(1, vec2)
    index.add_item(2, vec3)

    # Test distances
    dist_01 = index.get_distance(0, 1)
    dist_02 = index.get_distance(0, 2)
    dist_12 = index.get_distance(1, 2)

    print(f"✓ Distance(0,1) = {dist_01:.4f} (expected ~1.0)")
    print(f"✓ Distance(0,2) = {dist_02:.4f} (expected ~1.0)")
    print(f"✓ Distance(1,2) = {dist_12:.4f} (expected ~√2≈1.414)")

    assert abs(dist_01 - 1.0) < 0.01, "Distance should be 1.0"
    assert abs(dist_02 - 1.0) < 0.01, "Distance should be 1.0"
    assert abs(dist_12 - math.sqrt(2)) < 0.01, "Distance should be √2"


def test_save_load():
    """Test 7: Save and load"""
    print("\n" + "="*70)
    print("TEST 7: Save and Load")
    print("="*70)

    random.seed(42)

    # Create and save index
    index1 = Index(f=5, metric='angular', seed=42)
    for i in range(10):
        vec = [random.random() for _ in range(5)]
        index1.add_item(i, vec)
    index1.build(n_trees=5)

    # Save
    fd, tmpfile = tempfile.mkstemp(suffix='.ann')
    os.close(fd)

    try:
        index1.save(tmpfile)
        print(f"✓ Saved to {tmpfile}")

        # Load into new index
        index2 = Index(f=5, metric='angular')
        index2.load(tmpfile)
        print(f"✓ Loaded: n_items={index2.get_n_items()}, n_trees={index2.get_n_trees()}")

        # Verify queries give same results
        neighbors1 = index1.get_nns_by_item(0, n=5)
        neighbors2 = index2.get_nns_by_item(0, n=5)

        assert neighbors1 == neighbors2, "Loaded index should give same results"
        print(f"✓ Query results match: {neighbors1}")

    finally:
        if os.path.exists(tmpfile):
            os.unlink(tmpfile)


def test_unbuild():
    """Test 8: Unbuild and rebuild"""
    print("\n" + "="*70)
    print("TEST 8: Unbuild and Rebuild")
    print("="*70)

    random.seed(42)
    index = Index(f=5, metric='angular', seed=42)

    # Add initial items
    for i in range(10):
        vec = [random.random() for _ in range(5)]
        index.add_item(i, vec)

    index.build(n_trees=5)
    print(f"✓ Initial build: n_items={index.get_n_items()}, n_trees={index.get_n_trees()}")

    # Unbuild
    index.unbuild()
    print(f"✓ After unbuild: n_trees={index.get_n_trees()}")

    # Add more items
    for i in range(10, 15):
        vec = [random.random() for _ in range(5)]
        index.add_item(i, vec)

    # Rebuild
    index.build(n_trees=5)
    print(f"✓ Rebuilt: n_items={index.get_n_items()}, n_trees={index.get_n_trees()}")

    assert index.get_n_items() == 15, "Should have 15 items after rebuild"


def test_hamming_metric():
    """Test 9: Hamming metric with binary vectors"""
    print("\n" + "="*70)
    print("TEST 9: Hamming Metric")
    print("="*70)

    index = Index(f=8, metric='hamming', seed=42)

    # Add binary vectors (as floats 0.0/1.0)
    vec1 = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]  # 0b10101010
    vec2 = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]  # 0b11110000
    vec3 = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]  # 0b00001111

    index.add_item(0, vec1)
    index.add_item(1, vec2)
    index.add_item(2, vec3)

    index.build(n_trees=5)

    # Check Hamming distances
    dist_01 = index.get_distance(0, 1)
    dist_02 = index.get_distance(0, 2)
    dist_12 = index.get_distance(1, 2)

    print(f"✓ Hamming(0,1) = {dist_01:.0f} bits differ")
    print(f"✓ Hamming(0,2) = {dist_02:.0f} bits differ")
    print(f"✓ Hamming(1,2) = {dist_12:.0f} bits differ")

    # Expected: 0b10101010 XOR 0b11110000 = 0b01011010 = 4 bits
    assert abs(dist_01 - 4.0) < 0.1, f"Expected ~4 bits, got {dist_01}"


def test_metric_aliases():
    """Test 10: Metric aliases"""
    print("\n" + "="*70)
    print("TEST 10: Metric Aliases")
    print("="*70)

    aliases = [
        ('cosine', 'angular'),
        ('l2', 'euclidean'),
        ('l1', 'manhattan'),
        ('.', 'dot'),
        ('dotproduct', 'dot'),
    ]

    for alias, canonical in aliases:
        index = Index(f=10, metric=alias)
        actual = index.metric
        if actual == canonical:
            print(f"✓ '{alias:12s}' → '{canonical}'")
        else:
            print(f"✗ '{alias:12s}' → expected '{canonical}', got '{actual}'")


def test_extended_parameters():
    """Test 9: Extended dtype parameters"""
    print("\n" + "="*70)
    print("TEST 9: Extended Type Parameters")
    print("="*70)

    # Create index with extended parameters
    index = Index(
        f=10,
        metric='angular',
        dtype='float32',
        index_dtype='int32',
        wrapper_dtype='uint64',
        random_dtype='uint64'
    )

    # Verify they're stored
    params = index.get_params()
    assert params['dtype'] == 'float32', "dtype should be stored"
    assert params['index_dtype'] == 'int32', "index_dtype should be stored"
    assert params['wrapper_dtype'] == 'uint64', "wrapper_dtype should be stored"
    assert params['random_dtype'] == 'uint64', "random_dtype should be stored"
    print("✓ Extended parameters stored correctly")

    # Verify they survive serialization
    data = index.serialize()
    index2 = Index.deserialize(data)
    params2 = index2.get_params()
    assert params2['dtype'] == 'float32', "dtype should survive serialization"
    print("✓ Extended parameters survive serialization")


def test_future_kwargs():
    """Test 10: Future extensibility with **kwargs"""
    print("\n" + "="*70)
    print("TEST 10: Future Extensibility (**kwargs)")
    print("="*70)

    # Test that unknown parameters trigger warning but don't fail
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        index = Index(
            f=10,
            metric='angular',
            future_param1='value1',
            future_param2=42
        )

        # Should have warned about unknown parameters
        assert len(w) > 0, "Should warn about unknown parameters"
        assert 'future_param' in str(w[-1].message), "Warning should mention unknown params"
        print(f"✓ Unknown parameters triggered FutureWarning: {w[-1].message}")

    # Index should still work
    index.add_item(0, [0.1] * 10)
    index.build()
    neighbors = index.get_nns_by_item(0, n=1)
    assert len(neighbors) == 1, "Index should work despite unknown params"
    print("✓ Index still functional with unknown parameters")


def test_future_kwargs2():
    """Test 8: Future parameter extensibility"""
    print("\n" + "="*70)
    print("TEST 8: Future Parameter Extensibility")
    print("="*70)

    # Test unknown parameters
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        index = Index(
            f=128,
            metric='angular',
            future_param='value',
            another_future_param=42
        )

        # Should warn but not fail
        assert len(w) > 0, "Should warn about unknown parameters"
        assert 'future_param' in str(w[-1].message)
        print(f"✓ Unknown parameters triggered warning: {w[-1].message}")

    # Index should still be functional
    params = index.get_params()
    assert params['f'] == 128
    print("✓ Index functional despite unknown parameters")

def test_mixin_hierarchy():
    """Test 1: Verify mixin inheritance"""
    print("\n" + "="*70)
    print("TEST 1: Mixin Hierarchy")
    print("="*70)

    # Check inheritance chain
    assert issubclass(Index, BaseIndex), "Index should inherit from BaseIndex"
    assert issubclass(BaseIndex, ReprHTMLMixin), "BaseIndex should inherit from ReprHTMLMixin"
    assert issubclass(BaseIndex, _HTMLDocumentationLinkMixin), "BaseIndex should inherit from _HTMLDocumentationLinkMixin"

    print("✓ Inheritance chain correct:")
    print("  Index → BaseIndex → ReprHTMLMixin, _HTMLDocumentationLinkMixin")

    # Check method availability
    index = Index(f=128, metric='angular')

    assert hasattr(index, '_get_doc_link'), "Should have _get_doc_link from mixin"
    assert hasattr(index, '_repr_html_'), "Should have _repr_html_ from mixin"
    assert hasattr(index, '_repr_mimebundle_'), "Should have _repr_mimebundle_ from mixin"
    assert hasattr(index, 'get_params'), "Should have get_params from BaseIndex"
    assert hasattr(index, 'set_params'), "Should have set_params from BaseIndex"

    print("✓ All mixin methods available")


# def test_dtype_support():
#     """Test 2: Multiple dtype support"""
#     print("\n" + "="*70)
#     print("TEST 2: Float Type Support")
#     print("="*70)

#     dtypes = [
#         ('float16', 'half precision'),
#         ('float32', 'single precision'),
#         ('float64', 'double precision'),
#         ('float80', 'extended precision'),
#         ('float128', 'quadruple precision'),
#     ]

#     for dtype, desc in dtypes:
#         try:
#             index = Index(f=128, metric='angular', dtype=dtype)
#             params = index.get_params()
#             assert params['dtype'] == dtype, f"dtype should be {dtype}"
#             print(f"✓ {dtype:10s} ({desc})")
#         except Exception as e:
#             print(f"✗ {dtype:10s}: {e}")

#     # Test aliases
#     aliases = [
#         ('half', 'float16'),
#         ('single', 'float32'),
#         ('double', 'float64'),
#         ('fp32', 'float32'),
#     ]

#     print("\n✓ Testing dtype aliases:")
#     for alias, canonical in aliases:
#         index = Index(f=10, metric='angular', dtype=alias)
#         params = index.get_params()
#         assert params['dtype'] == canonical, f"Alias {alias} should map to {canonical}"
#         print(f"  {alias:10s} → {canonical}")


def test_get_set_params():
    """Test 3: Parameter management (sklearn-style)"""
    print("\n" + "="*70)
    print("TEST 3: get_params() and set_params()")
    print("="*70)

    # Create index with known parameters
    index = Index(
        f=128,
        metric='euclidean',
        n_neighbors=10,
        seed=42,
        verbose=1,
        dtype='float32',
        index_dtype='int32'
    )

    # Get parameters
    params = index.get_params()
    print(f"✓ get_params() returned {len(params)} parameters")

    # Verify core parameters
    assert params['f'] == 128, "f should be 128"
    assert params['metric'] == 'euclidean', "metric should be euclidean"
    assert params['n_neighbors'] == 10, "n_neighbors should be 10"
    assert params['dtype'] == 'float32', "dtype should be float32"
    print(f"✓ Core parameters correct: f={params['f']}, metric={params['metric']}")

    # Test set_params (mutable parameters)
    index.set_params(n_neighbors=20, verbose=2)
    new_params = index.get_params()
    assert new_params['n_neighbors'] == 20, "n_neighbors should be updated"
    print("✓ set_params() updated mutable parameters")

    # Test immutability after construction
    # index._ensure_index()  # Construct the index
    try:
        index.set_params(f=256)
        print("✗ Should not allow changing f after construction")
    except ValueError as e:
        print(f"✓ Correctly prevented changing f: {e}")


def test_doc_link_generation():
    """Test 4: Documentation link generation"""
    print("\n" + "="*70)
    print("TEST 4: Documentation Link Generation")
    print("="*70)

    index = Index(f=128, metric='angular')

    # Get doc link
    doc_link = index._get_doc_link()
    print(f"✓ Generated doc link:\n  {doc_link}")

    # Verify structure
    assert 'scikit-plots' in doc_link or doc_link == '', \
        "Link should contain scikit-plots.github.io or be empty"

    if doc_link:
        assert 'Index' in doc_link, "Link should contain class name"
        print("✓ Link contains class name")


def test_repr_html():
    """Test 5: HTML representation"""
    print("\n" + "="*70)
    print("TEST 5: HTML Representation")
    print("="*70)

    index = Index(f=128, metric='euclidean', dtype='float64')

    # Get HTML
    html = index._repr_html_()
    print(f"✓ Generated HTML ({len(html)} chars)")

    # Verify structure
    assert 'sk-index' in html or 'annoy' in html.lower(), "HTML should have index class"
    assert '128' in html, "HTML should contain f value"
    assert 'euclidean' in html, "HTML should contain metric"
    assert 'float64' in html, "HTML should contain dtype"
    print("✓ HTML contains parameters")

    # Test MIME bundle
    mime_bundle = index._repr_mimebundle_()
    assert 'text/plain' in mime_bundle, "Should have text/plain"
    assert 'text/html' in mime_bundle, "Should have text/html"
    print("✓ MIME bundle generated correctly")


def test_enhanced_repr():
    """Test 2: Enhanced string representations"""
    print("\n" + "="*70)
    print("TEST 2: Enhanced __repr__ and __str__")
    print("="*70)

    index = Index(f=128, metric='angular', seed=42)

    # Test __repr__ (should include memory address)
    repr_str = repr(index)
    print(f"✓ repr: {repr_str}")
    assert "0x" in repr_str, "repr should include memory address"
    assert "Index" in repr_str, "repr should include class name"

    # Test __str__ (should be concise)
    str_str = str(index)
    print(f"✓ str: {str_str}")
    assert "Index" in str_str, "str should include class name"
    assert len(str_str) < len(repr_str), "str should be shorter than repr"


# def test_repr_consistency():
#     """Test 6: __repr__ and __str__ consistency"""
#     print("\n" + "="*70)
#     print("TEST 6: String Representation Consistency")
#     print("="*70)

#     index = Index(f=128, metric='angular', dtype='float32')

#     # Test __repr__
#     repr_str = repr(index)
#     print(f"✓ repr: {repr_str}")
#     assert 'Index' in repr_str
#     assert '128' in repr_str
#     assert 'angular' in repr_str
#     assert '0x' in repr_str, "repr should include memory address"

#     # Test __str__
#     str_str = str(index)
#     print(f"✓ str: {str_str}")
#     assert 'Index' in str_str
#     assert len(str_str) < len(repr_str), "str should be shorter than repr"


def test_modular_extensibility():
    """Test 7: Future extensibility"""
    print("\n" + "="*70)
    print("TEST 7: Modular Extensibility")
    print("="*70)

    # Test that Index can be subclassed
    class CustomIndex(Index):
        """Custom index subclass."""

        def custom_method(self):
            return "custom"

    custom = CustomIndex(f=10, metric='angular')
    assert custom.custom_method() == "custom"
    print("✓ Index can be subclassed")

    # Test that BaseIndex provides infrastructure
    assert hasattr(custom, 'get_params')
    assert hasattr(custom, '_get_doc_link')
    print("✓ Subclass inherits all base functionality")


def test_dtype_metadata():
    """Test 9: Float type metadata"""
    print("\n" + "="*70)
    print("TEST 9: Float Type Metadata")
    print("="*70)

    dtypes_and_sizes = [
        ('float16', 2),
        ('float32', 4),
        ('float64', 8),
        ('float80', 10),  # or 16 with padding
        ('float128', 16),
    ]

    for dtype, expected_size in dtypes_and_sizes:
        index = Index(f=128, metric='angular', dtype=dtype)

        # Get dtype from params
        params = index.get_params()
        actual_dtype = params['dtype']

        assert actual_dtype == dtype, f"dtype should be {dtype}"
        print(f"✓ {dtype:10s}: stored correctly (expected size: {expected_size} bytes)")


def test_schema_versioning():
    """Test 10: Schema versioning for compatibility"""
    print("\n" + "="*70)
    print("TEST 10: Schema Versioning")
    print("="*70)

    # Test default schema version
    index1 = Index(f=128, metric='angular')
    params1 = index1.get_params()
    assert 'schema_version' in params1
    print(f"✓ Default schema_version: {params1['schema_version']}")

    # Test explicit schema version
    index2 = Index(f=128, metric='angular', schema_version=2)
    params2 = index2.get_params()
    assert params2['schema_version'] == 2
    print(f"✓ Custom schema_version: {params2['schema_version']}")


def test_context_manager():
    """Test 1: Context manager protocol"""
    print("\n" + "="*70)
    print("TEST 1: Context Manager")
    print("="*70)

    random.seed(42)

    # Test with-statement
    with Index(f=10, metric='angular', seed=42) as index:
        for i in range(10):
            vec = [random.random() for _ in range(10)]
            index.add_item(i, vec)
        index.build(n_trees=5)

        neighbors = index.get_nns_by_item(0, n=5)
        print(f"✓ Context manager works: got {len(neighbors)} neighbors")

    # Index should still be usable after context
    neighbors2 = index.get_nns_by_item(0, n=5)
    assert neighbors == neighbors2, "Results should be same after context"
    print("✓ Index still usable after context exit")


def test_get_set_state():
    """Test 5: State dictionary management"""
    print("\n" + "="*70)
    print("TEST 5: get_state() and set_state()")
    print("="*70)

    random.seed(42)

    # Create and populate index
    index1 = Index(f=10, metric='manhattan', seed=42, n_neighbors=15)
    for i in range(20):
        vec = [random.random() for _ in range(10)]
        index1.add_item(i, vec)
    index1.build(n_trees=5)

    # Get state
    state = index1.get_state()
    print(f"✓ get_state() returned state with {len(state)} keys")

    # Verify state contents
    assert '__version__' in state, "State should have version"
    assert 'params' in state, "State should have params"
    assert 'constructed' in state, "State should have constructed flag"
    assert 'index_data' in state, "State should have index_data"
    print(f"✓ State contains: {list(state.keys())}")

    # Create new index and restore state
    index2 = Index()
    index2.set_state(state)
    print("✓ set_state() restored index")

    # Verify restoration
    assert index2.f == 10, "Dimension should match"
    assert index2.metric == 'manhattan', "Metric should match"
    assert index2.n_neighbors == 15, "n_neighbors should match"
    assert index2.get_n_items() == 20, "Items should match"
    print(f"✓ Parameters match: f={index2.f}, metric={index2.metric}, n_neighbors={index2.n_neighbors}")

    # Verify queries
    neighbors1 = index1.get_nns_by_item(0, n=5)
    neighbors2 = index2.get_nns_by_item(0, n=5)
    assert neighbors1 == neighbors2, "Queries should match"
    print("✓ Query results match")


def test_json_serialization():
    """Test 6: JSON-compatible serialization"""
    print("\n" + "="*70)
    print("TEST 6: JSON Serialization (serialize(), deserialize())")
    print("="*70)

    random.seed(42)

    # Create and populate index
    index1 = Index(f=10, metric='dot', seed=42)
    for i in range(20):
        vec = [random.random() for _ in range(10)]
        index1.add_item(i, vec)
    index1.build(n_trees=5)

    # Serialize to dict
    data = index1.serialize()
    print(f"✓ serialize() returned dict with {len(data)} keys")

    # Verify it's JSON-compatible (index_data will be base64)
    try:
        json_str = json.dumps(data)
        print(f"✓ Successfully converted to JSON ({len(json_str)} chars)")
    except Exception as e:
        print(f"✗ Failed to convert to JSON: {e}")

    # Deserialize from dict
    data_parsed = json.loads(json_str)
    index2 = Index.deserialize(data_parsed)
    print("✓ deserialize() restored index")

    # Verify
    assert index2.f == 10, "Dimension should match"
    assert index2.metric == 'dot', "Metric should match"
    neighbors1 = index1.get_nns_by_item(0, n=5)
    neighbors2 = index2.get_nns_by_item(0, n=5)
    assert neighbors1 == neighbors2, "Queries should match"
    print("✓ Deserialized index works correctly")


def test_to_dict_from_dict():
    """Test 7: Convenience aliases"""
    print("\n" + "="*70)
    print("TEST 7: to_dict() and from_dict() Aliases")
    print("="*70)

    random.seed(42)

    # Create index
    index1 = Index(f=10, metric='angular', seed=42)
    for i in range(10):
        vec = [random.random() for _ in range(10)]
        index1.add_item(i, vec)
    index1.build(n_trees=5)

    # Test to_dict (alias for serialize)
    data = index1.to_dict()
    print("✓ to_dict() returned dict")

    # Test from_dict (alias for deserialize)
    index2 = Index.from_dict(data)
    print("✓ from_dict() restored index")

    # Verify
    neighbors1 = index1.get_nns_by_item(0, n=5)
    neighbors2 = index2.get_nns_by_item(0, n=5)
    assert neighbors1 == neighbors2, "Queries should match"
    print("✓ Aliases work correctly")


def test_pickle_protocol():
    """Test 4: Pickle serialization"""
    print("\n" + "="*70)
    print("TEST 4: Pickle Protocol (__getstate__, __setstate__, __reduce__)")
    print("="*70)

    random.seed(42)

    # Create and populate index
    index1 = Index(f=10, metric='angular', seed=42)
    for i in range(20):
        vec = [random.random() for _ in range(10)]
        index1.add_item(i, vec)
    index1.build(n_trees=5)

    # Get query results from original
    neighbors1, dist1 = index1.get_nns_by_item(0, n=5, include_distances=True)
    print(f"✓ Original index: neighbors={neighbors1}")

    # Pickle
    pickled = pickle.dumps(index1)
    print(f"✓ Pickled to {len(pickled)} bytes")

    # Unpickle
    index2 = pickle.loads(pickled)
    print("✓ Unpickled successfully")

    # Verify parameters
    assert index2.f == 10, "Dimension should be preserved"
    assert index2.metric == 'angular', "Metric should be preserved"
    assert index2.get_n_items() == 20, "Items should be preserved"
    assert index2.get_n_trees() == 5, "Trees should be preserved"
    print(f"✓ Parameters preserved: f={index2.f}, metric={index2.metric}")

    # Verify queries give same results
    neighbors2, dist2 = index2.get_nns_by_item(0, n=5, include_distances=True)
    assert neighbors1 == neighbors2, "Neighbors should match"
    print("✓ Query results match after unpickling")


def test_clone():
    """Test 8: Clone with parameter overrides"""
    print("\n" + "="*70)
    print("TEST 8: clone() Method")
    print("="*70)

    # Create original index
    index1 = Index(f=128, metric='euclidean', seed=42, n_neighbors=10)

    # Clone with same parameters
    index2 = index1.clone()
    params2 = index2.get_params()
    assert params2['f'] == 128, "Clone should have same f"
    assert params2['metric'] == 'euclidean', "Clone should have same metric"
    assert params2['seed'] == 42, "Clone should have same seed"
    print("✓ clone() created copy with same parameters")

    # Clone with overrides
    index3 = index1.clone(seed=123, n_neighbors=20)
    params3 = index3.get_params()
    assert params3['f'] == 128, "Clone should keep original f"
    assert params3['metric'] == 'euclidean', "Clone should keep original metric"
    assert params3['seed'] == 123, "Clone should have new seed"
    assert params3['n_neighbors'] == 20, "Clone should have new n_neighbors"
    print("✓ clone(seed=123, n_neighbors=20) overrode parameters correctly")


def run_all_tests():
    """Run all modular architecture tests"""
    print("\n" + "="*70)
    print("MODULAR ARCHITECTURE TEST SUITE")
    print("="*70)

    tests = [
        ("Initialization", test_initialization),
        ("Add and Build", test_add_and_build),
        ("Query by Item", test_query_by_item),
        ("Query by Vector", test_query_by_vector),
        ("Get Item", test_get_item),
        ("Distances", test_distances),
        ("Save and Load", test_save_load),
        ("Unbuild", test_unbuild),
        ("Hamming Metric", test_hamming_metric),
        ("Metric Aliases", test_metric_aliases),
        ("Extended Parameters", test_extended_parameters),
        ("Future **kwargs", test_future_kwargs),
        ("Mixin Hierarchy", test_mixin_hierarchy),
        # ("Float Type Support", test_dtype_support),
        ("Parameter Management", test_get_set_params),
        ("Documentation Links", test_doc_link_generation),
        ("HTML Representation", test_repr_html),
        ("Enhanced __repr__/__str__", test_enhanced_repr),
        # ("String Representations", test_repr_consistency),
        ("Extensibility", test_modular_extensibility),
        # ("Future Parameters", test_future_kwargs),
        ("Float Type Metadata", test_dtype_metadata),
        ("Schema Versioning", test_schema_versioning),
        ("Context Manager", test_context_manager),
        ("get_state/set_state", test_get_set_state),
        ("JSON Serialization", test_json_serialization),
        ("to_dict/from_dict", test_to_dict_from_dict),
        ("Pickle Protocol", test_pickle_protocol),
        ("clone()", test_clone),
    ]

    results = []

    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ TEST FAILED: {name}")
            print(f"  Exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n✗ {total - passed} TEST(S) FAILED")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
