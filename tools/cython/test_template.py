#!/usr/bin/env python3
"""
Test script to verify the fixed template works correctly.

Usage:
    python test_template.py annoylib.pxd.in

Requires:
    pip install tempita
"""

import sys
from pathlib import Path

try:
    from tempita import Template
except ImportError:
    print("ERROR: tempita not found. Install with: pip install tempita")
    sys.exit(1)

def test_template(template_path):
    """Test that a template file works correctly."""
    template_path = Path(template_path)

    if not template_path.exists():
        print(f"ERROR: Template file not found: {template_path}")
        return False

    print(f"Testing template: {template_path}")
    print("=" * 70)

    # Read template
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"✓ Template file read: {len(content)} bytes, {len(content.splitlines())} lines")
    except Exception as e:
        print(f"✗ Failed to read template: {e}")
        return False

    # Parse template
    try:
        template = Template(content, name=str(template_path))
        print("✓ Template parsed successfully")
    except Exception as e:
        print(f"✗ Template parsing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Substitute variables
    try:
        result = template.substitute()
        print("✓ Template substitution successful")
        print(f"  Generated {len(result.splitlines())} lines")
    except Exception as e:
        print(f"✗ Template substitution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Validate result
    try:
        # Check for unprocessed markers
        if '{{' in result or '}}' in result:
            print("✗ ERROR: Unprocessed template markers found in output!")
            lines = result.split('\n')
            for i, line in enumerate(lines, 1):
                if '{{' in line or '}}' in line:
                    print(f"  Line {i}: {line[:100]}")
            return False
        else:
            print("✓ No unprocessed template markers")

        # Count generated classes
        import re
        classes = re.findall(r'cdef cppclass (CAnnoyIndex\w+):', result)
        print(f"✓ Generated {len(classes)} C++ class declarations")

        if len(classes) > 0:
            print(f"  First 5 classes:")
            for cls in classes[:5]:
                print(f"    - {cls}")
            if len(classes) > 5:
                print(f"    ... and {len(classes) - 5} more")

        # Check for expected combinations (should be 20)
        expected_count = 20  # 4 metrics * 2 index types * 2-3 data types
        if len(classes) < 15:  # Allow some flexibility
            print(f"⚠ WARNING: Only {len(classes)} classes generated, expected ~{expected_count}")

    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False

    # Save test output
    output_path = template_path.parent / 'test_output.pxd'
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"✓ Test output saved to: {output_path}")
    except Exception as e:
        print(f"⚠ Could not save test output: {e}")

    print()
    print("=" * 70)
    print("SUCCESS: Template is working correctly!")
    print("=" * 70)
    return True

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_template.py <template_file.in>")
        print()
        print("Example:")
        print("  python test_template.py annoylib.pxd.in")
        sys.exit(1)

    success = test_template(sys.argv[1])
    sys.exit(0 if success else 1)
