"""Tests for NAIP-CHM chips on a fixed map lattice (``chm_inference.tiling``)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from itertools import pairwise

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from csdv_core.chm_inference.tiling import (
    global_pixel_origin,
    lattice_chip_grid,
    lattice_starts,
)

CHIP, OVERLAP = 432, 0.2
STRIDE = CHIP - int(CHIP * OVERLAP)  # 346


def upstream_chip_grid(width, height, chip_size, overlap_fraction):
    """The upstream naip-chm ``create_chip_grid`` (commit 3a474f4), for reference."""
    stride = chip_size - int(chip_size * overlap_fraction)
    chips = []
    row = 0
    while row < height:
        col = 0
        while col < width:
            row_end = min(row + chip_size, height)
            col_end = min(col + chip_size, width)
            if row_end == height and row_end - row < chip_size:
                row_start = max(0, height - chip_size)
            else:
                row_start = row
            if col_end == width and col_end - col < chip_size:
                col_start = max(0, width - chip_size)
            else:
                col_start = col
            chips.append((row_start, col_start, row_end, col_end))
            if col_end == width:
                break
            col += stride
        if row_end == height:
            break
        row += stride
    return chips


def _coverage(chips, width, height):
    covered = np.zeros((height, width), dtype=int)
    for r0, c0, r1, c1 in chips:
        covered[r0:r1, c0:c1] += 1
    return covered


@pytest.mark.parametrize("origin", [0, 1, 101, STRIDE - 1, 483050])
@pytest.mark.parametrize("length", [100, CHIP, CHIP + 1, 1500, 17856])
def test_lattice_starts_cover_axis_with_enough_overlap(origin, length):
    starts = lattice_starts(length, CHIP, STRIDE, origin)
    assert starts[0] == 0
    assert min(starts[-1] + CHIP, length) == length
    for a, b in pairwise(starts):
        assert CHIP - (b - a) >= CHIP - STRIDE  # overlap at least the upstream one
    interior = [s for s in starts if 0 < s < length - CHIP]
    assert all((origin + s) % STRIDE == 0 for s in interior)


def test_aligned_origin_reproduces_upstream_grid():
    for width, height in [(1500, 1200), (17856, 24160), (300, 500)]:
        ours = lattice_chip_grid(width, height, CHIP, OVERLAP, 0, STRIDE * 7)
        assert sorted(ours) == sorted(upstream_chip_grid(width, height, CHIP, OVERLAP))


def test_shifted_mosaics_share_interior_chips():
    """A mosaic cropped by (dx, dy) px gets the same chips away from its edges."""
    col0, row0, dx, dy = 483050, -8171905, 101, 77
    width, height = 4000, 3500
    a = lattice_chip_grid(width, height, CHIP, OVERLAP, col0, row0)
    b = lattice_chip_grid(width - dx, height - dy, CHIP, OVERLAP, col0 + dx, row0 + dy)
    a_global = {(r + row0, c + col0) for r, c, *_ in a}
    b_global = {(r + row0 + dy, c + col0 + dx) for r, c, *_ in b}
    # Chips of B that are not anchored to B's edges are chips of A.
    b_interior = {
        (r + row0 + dy, c + col0 + dx)
        for r, c, r1, c1 in b
        if 0 < r and 0 < c and r1 < height - dy and c1 < width - dx
    }
    assert b_interior and b_interior <= a_global
    # Upstream, almost none of them coincide.
    up_a = {
        (r + row0, c + col0)
        for r, c, *_ in upstream_chip_grid(width, height, CHIP, OVERLAP)
    }
    up_b = {
        (r + row0 + dy, c + col0 + dx)
        for r, c, *_ in upstream_chip_grid(width - dx, height - dy, CHIP, OVERLAP)
    }
    assert len(up_a & up_b) < len(b_global & a_global)


def test_every_pixel_is_covered():
    chips = lattice_chip_grid(1000, 900, CHIP, OVERLAP, 12345, 678)
    assert _coverage(chips, 1000, 900).min() >= 1
    assert all(r1 - r0 == CHIP and c1 - c0 == CHIP for r0, c0, r1, c1 in chips)


def test_chip_size_must_be_multiple_of_16():
    with pytest.raises(ValueError):
        lattice_chip_grid(1000, 1000, 430, OVERLAP, 0, 0)


def test_global_pixel_origin(caplog):
    assert global_pixel_origin(from_origin(289830.0, 4903143.0, 1.0, 1.0)) == (
        483050,
        -8171905,
    )
    caplog.clear()
    global_pixel_origin(from_origin(289830.1, 4903143.0, 0.6, 0.6))
    assert "not on the 0.6 m lattice" in caplog.text


def test_runner_patches_upstream_chip_grid(tmp_path):
    """The runner swaps the chip grid that upstream code looks up at call time."""
    repo = tmp_path / "naip-chm"
    (repo / "src").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "src" / "__init__.py").write_text("")
    (repo / "src" / "inference_utils.py").write_text(textwrap.dedent("""
            def create_chip_grid(width, height, chip_size, overlap_fraction):
                return "upstream"

            def process(width, height):
                return create_chip_grid(width, height, 432, 0.2)
            """))
    (repo / "scripts" / "inference.py").write_text(textwrap.dedent("""
            import argparse, json, sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).resolve().parent.parent))
            from src.inference_utils import process
            parser = argparse.ArgumentParser()
            parser.add_argument("--naip-quad")
            parser.add_argument("--out")
            args = parser.parse_args()
            Path(args.out).write_text(json.dumps(process(1000, 900)))
            """))
    naip = tmp_path / "NAIP_q_20180923_0p6m.tif"
    with rasterio.open(
        naip,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="uint8",
        crs="EPSG:26919",
        transform=from_origin(289830.0, 4903143.0, 0.6, 0.6),
    ) as dst:
        dst.write(np.zeros((1, 4, 4), dtype="uint8"))
    out = tmp_path / "chips.json"
    env = os.environ | {"PYTHONPATH": os.pathsep.join(sys.path)}
    subprocess.run(
        [
            sys.executable,
            "-m",
            "csdv_core.chm_inference.tiling",
            str(repo / "scripts" / "inference.py"),
            "--naip-quad",
            str(naip),
            "--out",
            str(out),
        ],
        cwd=repo,
        env=env,
        check=True,
    )
    chips = [tuple(c) for c in json.loads(out.read_text())]
    col0, row0 = global_pixel_origin(from_origin(289830.0, 4903143.0, 0.6, 0.6))
    assert chips == lattice_chip_grid(1000, 900, 432, 0.2, col0, row0)
