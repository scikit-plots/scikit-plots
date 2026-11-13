# corpus/_readers/tests/test__pdf_layout.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
PDFReader lays pages out the same way on every read (CX-16).

pdfminer.six breaks ties between equally distant text boxes by ``id()``, a
memory address. The tests build the same boxes with different addresses
(allocated in reverse) and compare the reading order.

Run with::

    pytest scikitplot/corpus/_readers/tests/test__pdf_layout.py -v
"""

from __future__ import annotations

import pytest

layout = pytest.importorskip("pdfminer.layout")
pytest.importorskip("pdfminer.converter")
pdfinterp = pytest.importorskip("pdfminer.pdfinterp")

from .._pdf import _stable_layout_classes  # noqa: E402


def _boxes(coords: list, reverse_allocation: bool) -> list:
    """Text boxes at *coords*, in list order, allocated in either order."""
    boxes: list = [None] * len(coords)
    spacers = []  # held so that addresses differ between the two orders
    order = range(len(coords))
    for i in reversed(order) if reverse_allocation else order:
        spacers.append([0] * (i + 1))
        box = layout.LTTextBoxHorizontal()
        x, y = coords[i]
        box.set_bbox((x, y, x + 10.0, y + 10.0))
        box.tag = i
        boxes[i] = box
    return boxes


def _reading_order(page_class: type, coords: list, reverse_allocation: bool) -> list:
    boxes = _boxes(coords, reverse_allocation)
    page = page_class(1, (0, 0, 200, 200))
    assigner = layout.IndexAssigner()
    for group in page.group_textboxes(layout.LAParams(), boxes):
        assigner.run(group)
    return [box.tag for box in sorted(boxes, key=lambda box: box.index)]


#: A 4 x 4 grid: many pairs of boxes are equally far apart.
GRID = [(x * 20.0, y * 20.0) for y in range(4) for x in range(4)]

#: Boxes whose pairwise distances all differ: no ties.
NO_TIES = [(0.0, 0.0), (13.0, 2.0), (31.0, 7.0), (5.0, 41.0), (60.0, 23.0)]


class TestStableGrouping:
    def test_ties_give_one_order_whatever_the_addresses(self):
        page_class, _ = _stable_layout_classes()
        assert _reading_order(page_class, GRID, False) == _reading_order(
            page_class, GRID, True
        )

    def test_pdfminer_itself_depends_on_the_addresses_here(self):
        # Shows the test above can fail: pdfminer's own grouping of the same
        # grid gives another order when the boxes have other addresses.
        assert _reading_order(layout.LTPage, GRID, False) != _reading_order(
            layout.LTPage, GRID, True
        )

    @pytest.mark.parametrize("reverse_allocation", [False, True])
    def test_without_ties_the_order_is_pdfminers(self, reverse_allocation):
        page_class, _ = _stable_layout_classes()
        assert _reading_order(page_class, NO_TIES, reverse_allocation) == (
            _reading_order(layout.LTPage, NO_TIES, reverse_allocation)
        )

    def test_one_box_is_one_group(self):
        page_class, _ = _stable_layout_classes()
        assert _reading_order(page_class, [(0.0, 0.0)], False) == [0]

    def test_the_device_lays_pages_out_with_the_stable_page(self):
        page_class, device_class = _stable_layout_classes()
        assert issubclass(page_class, layout.LTPage)
        converter = pytest.importorskip("pdfminer.converter")
        assert issubclass(device_class, converter.PDFPageAggregator)


class _Page:
    """The one attribute ``begin_page`` reads from a ``PDFPage``."""

    mediabox = (0, 0, 300, 400)


def _begin(device_class: type) -> object:
    device = device_class(pdfinterp.PDFResourceManager(), laparams=layout.LAParams())
    device.begin_page(_Page(), (1, 0, 0, 1, 0, 0))
    return device.cur_item


class TestStableDevice:
    def test_the_page_keeps_every_attribute_pdfminer_set(self):
        converter = pytest.importorskip("pdfminer.converter")
        page_class, device_class = _stable_layout_classes()
        ours = _begin(device_class)
        theirs = _begin(converter.PDFPageAggregator)
        assert type(ours) is page_class
        assert vars(ours).keys() == vars(theirs).keys()
        assert (ours.pageid, ours.bbox, ours.rotate) == (
            theirs.pageid,
            theirs.bbox,
            theirs.rotate,
        )


class TestStableGroupingEdges:
    def test_no_boxes_is_no_group(self):
        page_class, _ = _stable_layout_classes()
        page = page_class(1, (0, 0, 200, 200))
        assert page.group_textboxes(layout.LAParams(), []) == []

    @pytest.mark.parametrize("reverse_allocation", [False, True])
    def test_vertical_boxes_group_top_to_bottom_right_to_left(self, reverse_allocation):
        page_class, _ = _stable_layout_classes()
        boxes: list = [None] * len(GRID)
        order = range(len(GRID))
        for i in reversed(order) if reverse_allocation else order:
            box = layout.LTTextBoxVertical()
            x, y = GRID[i]
            box.set_bbox((x, y, x + 10.0, y + 10.0))
            boxes[i] = box
        page = page_class(1, (0, 0, 200, 200))
        (top,) = page.group_textboxes(layout.LAParams(), boxes)
        assert isinstance(top, layout.LTTextGroupTBRL)

    def test_vertical_ties_give_one_order_whatever_the_addresses(self):
        page_class, _ = _stable_layout_classes()

        def order(reverse_allocation: bool) -> list:
            boxes: list = [None] * len(GRID)
            spacers = []
            idx = range(len(GRID))
            for i in reversed(idx) if reverse_allocation else idx:
                spacers.append([0] * (i + 1))
                box = layout.LTTextBoxVertical()
                x, y = GRID[i]
                box.set_bbox((x, y, x + 10.0, y + 10.0))
                box.tag = i
                boxes[i] = box
            page = page_class(1, (0, 0, 200, 200))
            assigner = layout.IndexAssigner()
            for group in page.group_textboxes(layout.LAParams(), boxes):
                assigner.run(group)
            return [b.tag for b in sorted(boxes, key=lambda b: b.index)]

        assert order(False) == order(True)
