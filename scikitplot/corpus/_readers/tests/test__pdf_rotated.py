"""PDFReader reads rotated text as words in writing order (CX-48)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from .._pdf import _rotated_text


def _run(text, angle, origin=(100.0, 400.0), size=12.0, scale=1.0, gap_after=()):
    """
    Fake rotated LTChars of *text*, drawn from *origin* at *angle*.

    *size* is the font size in page units; *scale* is the text matrix's
    share of it (producers write ``12 Tf`` with a unit matrix, or ``1 Tf``
    with a matrix scaled by 12), so ``adv`` is in matrix units.
    """
    ux, uy = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    advance = size / 2  # page units
    chars, pos = [], 0.0
    for i, ch in enumerate(text):
        e, f = origin[0] + pos * ux, origin[1] + pos * uy
        # Box: advance along the direction, size across it.
        corners = [
            (e + dx * ux - dy * uy, f + dx * uy + dy * ux)
            for dx in (0.0, advance)
            for dy in (-0.2 * size, 0.8 * size)
        ]
        xs, ys = [p[0] for p in corners], [p[1] for p in corners]
        chars.append(
            SimpleNamespace(
                matrix=(scale * ux, scale * uy, -scale * uy, scale * ux, e, f),
                adv=advance / scale,
                bbox=(min(xs), min(ys), max(xs), max(ys)),
                upright=False,
                get_text=lambda ch=ch: ch,
            )
        )
        pos += advance + (2 * size if i in gap_after else 0.0)
    return chars


@pytest.mark.parametrize("angle", [90, 270, 180, 45, 315])
def test_one_run_reads_in_writing_order_whatever_the_drawing_order(angle):
    chars = _run("Year2015", angle)
    assert _rotated_text(list(reversed(chars))) == "Year2015\n"


@pytest.mark.parametrize("scale", [1.0, 7.0, 12.0])
def test_a_wide_gap_becomes_a_space_whatever_the_matrix_scale(scale):
    assert _rotated_text(_run("Year2015", 90, scale=scale)) == "Year2015\n"
    assert _rotated_text(_run("Year2015", 90, scale=scale, gap_after={3})) == "Year 2015\n"


def test_runs_on_different_baselines_or_angles_stay_apart_in_first_seen_order():
    left = _run("GDP", 90, origin=(50.0, 300.0))
    right = _run("Rate", 90, origin=(400.0, 300.0))
    down = _run("Down", 270, origin=(50.0, 300.0))
    assert _rotated_text(right + left + down) == "Rate\nGDP\nDown\n"


reportlab = pytest.importorskip("reportlab")
pytest.importorskip("pdfminer.high_level")


@pytest.fixture
def chart_pdf(tmp_path):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    path = tmp_path / "chart.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    c.drawString(72, 720, "Horizontal label text on the page")
    for i, angle in enumerate([90, 270, 180, 45]):
        c.saveState()
        c.translate(150 + i * 100, 400)
        c.rotate(angle)
        c.drawString(0, 0, f"Year {2015 + i}")
        c.restoreState()
    c.drawString(72, 100, "Footer line after the chart")
    c.save()
    return path


def test_reader_gives_rotated_labels_as_words(chart_pdf):
    from .._pdf import PDFReader

    text = "\n".join(c["text"] for c in PDFReader(input_path=chart_pdf).get_raw_chunks())
    lines = text.splitlines()
    for year in range(2015, 2019):
        assert f"Year {year}" in lines
    assert not [line for line in lines if len(line.strip()) == 1]
    # Upright text keeps its place: first and last, as before.
    assert lines[0] == "Horizontal label text on the page"
    assert lines[-1] == "Footer line after the chart"
