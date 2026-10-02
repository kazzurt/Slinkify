"""Regression checks for shared profile, export and layer-analysis helpers."""
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import manifold3d as m3d
import numpy as np
import contextlib
import io
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import slinky


def _reference_layer_check(man, layer, height, gap, slack):
    """The previous all-slices implementation, to verify streaming preserves results."""
    zs = np.arange(layer / 2, height, layer)
    sections = [man.slice(float(z)) for z in zs]
    n_gap, n_slack = round(gap / layer), round(slack / layer)
    rows = []
    for k in range(1, len(sections)):
        area = sections[k].area()
        if area <= 0:
            continue
        unsupported = sections[k] - sections[k - 1]
        ua = unsupported.area()
        if ua <= 1e-6 or k <= n_gap:
            continue
        below = sections[k - n_gap - 1]
        for j in range(1, n_slack + 1):
            if k - n_gap - 1 - j >= 0:
                below = below + sections[k - n_gap - 1 - j]
        land = (unsupported ^ below).area() / ua
        deepest = None
        for j in range(2, min(k, 60) + 1):
            if (unsupported ^ sections[k - j]).area() >= 0.99 * ua:
                deepest = j - 1
                break
        rows.append((zs[k], area, ua, land, deepest))
    return rows


class SharedHelpersTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="slinky-helper-test-")
        self.addCleanup(self.temporary.cleanup)
        self.tmp_path = Path(self.temporary.name)

    def _check_invalid_profile(self, regions, message):
        with self.assertRaises(ValueError) as exc:
            slinky.validate_regions(regions)
        assert message in str(exc.exception)

    def test_profile_orientation_closure_and_large_drawing_coordinates(self):
        outer = np.array([[0, 0], [0, 4], [4, 4], [4, 0], [0, 0]], float) + 1e10
        hole = np.array([[1, 1], [2, 1], [2, 2], [1, 2]], float) + 1e10
        regions = slinky.validate_regions([{"outer": outer, "holes": [hole]}])
        assert len(regions[0]["outer"]) == 4
        assert slinky.signed_area(regions[0]["outer"]) == 16
        assert slinky.signed_area(regions[0]["holes"][0]) == -1
        assert len(outer) == 5

    def test_loop_loader_accepts_saved_project_and_utf8_bom(self):
        tmp_path = self.tmp_path
        path = tmp_path / "profile.slinky.json"
        path.write_text(json.dumps({"regions": [{"outer": [[0, 0], [2, 0], [0, 2]]}]}),
                        encoding="utf-8-sig")
        regions = slinky.load_loops(SimpleNamespace(step="", loops=str(path)))
        assert slinky.signed_area(regions[0]["outer"]) == 2

    def _check_step_plane_frame(self, normal):
        normal = np.array(normal, float)
        normal /= np.linalg.norm(normal)
        ex, ey = slinky._plane_basis(normal)
        assert np.allclose(np.array([ex, ey, normal]) @ np.array([ex, ey, normal]).T, np.eye(3))
        points = np.array([2 * ex + 3 * ey, -4 * ex + 5 * ey])
        projected = points @ np.array([ex, ey]).T
        assert np.allclose(projected, [[2, 3], [-4, 5]])
        assert np.allclose(np.cross(ex, ey), normal)

    def test_chunked_stl_is_valid_at_chunk_boundary_and_long_header(self):
        tmp_path = self.tmp_path
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
        faces = np.tile([0, 1, 2], (16385, 1))
        path = tmp_path / "mesh.stl"
        slinky.write_stl(path, vertices, faces, name=b"a" * 100)
        data = path.read_bytes()
        assert len(data) == 84 + len(faces) * 50
        assert data[:80] == b"a" * 80
        assert struct.unpack("<I", data[80:84])[0] == len(faces)
        records = np.frombuffer(data, dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")],
                                offset=84)
        assert np.allclose(records["v"][[0, -1]], vertices)
        assert np.allclose(records["n"][[0, -1]], [0, 0, 1])
        assert not records["a"].any()

    def test_invalid_stl_does_not_overwrite_existing_file(self):
        tmp_path = self.tmp_path
        path = tmp_path / "existing.stl"
        path.write_bytes(b"original")
        with self.assertRaisesRegex(ValueError, "invalid vertex index"):
            slinky.write_stl(path, np.zeros((3, 3)), np.array([[0, 1, 3]]))
        assert path.read_bytes() == b"original"

    def test_stl_omits_only_faces_collapsed_at_export_precision_across_chunks(self):
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [1e8, 0, 0], [1e8 + 1, 0, 0], [1e8, 1, 0],
            [0, 0, 0], [1e-12, 0, 0], [0, 1e-12, 0],
        ], dtype=np.float64)
        # This face has area in memory, but its first two vertices coincide in STL.
        assert np.cross(vertices[4] - vertices[3], vertices[5] - vertices[3])[2] == 1
        assert np.array_equal(vertices[3].astype(np.float32), vertices[4].astype(np.float32))
        faces = np.tile([0, 1, 2], (16387, 1))
        omitted = [0, 2, 16383, 16384, 16386]
        faces[omitted] = [3, 4, 5]
        faces[2] = [0, 0, 1]  # Also discard an already-degenerate input face.
        faces[1] = [6, 7, 8]  # Retain genuinely tiny nonzero geometry.
        path = self.tmp_path / "precision-filtered.stl"
        slinky.write_stl(path, vertices, faces)
        data = path.read_bytes()
        expected_faces = np.delete(faces, omitted, axis=0)
        assert struct.unpack("<I", data[80:84])[0] == len(expected_faces)
        assert len(data) == 84 + len(expected_faces) * 50
        records = np.frombuffer(data, dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")],
                                offset=84)
        assert np.array_equal(records["v"], vertices[expected_faces].astype(np.float32))
        assert np.allclose(records["n"], [0, 0, 1])

        # A chunk with no surviving faces still produces a complete empty STL.
        slinky.write_stl(path, vertices, np.array([[3, 4, 5]]))
        data = path.read_bytes()
        assert len(data) == 84
        assert struct.unpack("<I", data[80:84])[0] == 0

    def _check_streamed_layer_analysis(self, gap, slack):
        args = SimpleNamespace(step="", loops="", profile="", width=3, thickness=0.6,
            scale=1, flip_radial=False, mirror=False, tol=0.02, min_hole_area=0.5,
            od=0, radius=10, pitch=0, gap=gap, turns=3, no_bottom_trim=False,
            no_top_trim=False, seg=24, left=False)
        man, info = slinky.build(args)
        actual = slinky.layer_check(man, 0.05, info["H"], gap, slack)
        expected = _reference_layer_check(man, 0.05, info["H"], gap, slack)
        assert actual
        assert actual == expected

    def test_empty_layer_report_and_invalid_layer_height(self):
        man = m3d.Manifold.cube([1, 1, 1])
        info = dict(R=5, pitch=1.4, gap=0.4, width=1, height=1, N=1, H=1,
            umin=-0.5, umax=0.5, prof=[{"outer": np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
            "holes": []}], th_end=slinky.TAU, bottom_trim=True, top_trim=True, wedge_deg=0)
        args = SimpleNamespace(material="PLA", scale=1, left=False, layer=0.2, no_layer_check=False)
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            _, rows = slinky.report(man, info, args)
        assert rows == []
        assert "no bridged layers" in captured.getvalue()
        with self.assertRaisesRegex(ValueError, "positive"):
            slinky.layer_check(man, 0, 1, 0.4)

    def test_renderer_produces_nonempty_views_and_rejects_empty_mesh(self):
        tmp_path = self.tmp_path
        from PIL import Image
        man = m3d.Manifold.cube([8, 2, 1])
        top = tmp_path / "top.png"
        front = tmp_path / "front.png"
        slinky.render(man, top, size=64, elev=90, azim=0)
        slinky.render(man, front, size=64, elev=0, azim=0)
        top_pixels = np.asarray(Image.open(top))[:, :, :3]
        front_pixels = np.asarray(Image.open(front))[:, :, :3]
        assert (top_pixels < 250).any()
        assert (front_pixels < 250).any()
        assert np.count_nonzero(top_pixels[:, :, 0] < 250) > np.count_nonzero(front_pixels[:, :, 0] < 250)
        with self.assertRaisesRegex(ValueError, "empty mesh"):
            slinky.render(m3d.Manifold(), tmp_path / "empty.png", size=64)

    def test_empty_profile_error(self):
        self._check_invalid_profile([], "at least one region")

    def test_missing_outer_error(self):
        self._check_invalid_profile([{}], "missing outer")

    def test_too_few_points_error(self):
        self._check_invalid_profile([{"outer": [[0, 0], [1, 1]]}], "three distinct")

    def test_collinear_profile_error(self):
        self._check_invalid_profile([{"outer": [[0, 0], [1, 1], [2, 2]]}], "zero area")

    def test_nonfinite_profile_error(self):
        self._check_invalid_profile([{"outer": [[0, 0], [1, 0], [0, float("nan")]]}], "finite")

    def test_wrong_dimension_profile_error(self):
        self._check_invalid_profile([{"outer": [[0, 0, 0], [1, 0, 0], [0, 1, 0]]}], "[x, y]")

    def test_rotated_step_plane_preserves_lengths(self):
        self._check_step_plane_frame([1, 2, 3])

    def test_top_step_plane_preserves_lengths(self):
        self._check_step_plane_frame([0, 0, 1])

    def test_bottom_step_plane_preserves_lengths(self):
        self._check_step_plane_frame([0, 0, -1])

    def test_side_step_plane_preserves_lengths(self):
        self._check_step_plane_frame([1, 0, 0])

    def test_streamed_analysis_preserves_short_gap_history(self):
        self._check_streamed_layer_analysis(0.4, 0.1)

    def test_streamed_analysis_preserves_long_gap_history(self):
        self._check_streamed_layer_analysis(3.5, 0.2)

    def test_excessive_layer_samples_fail_before_allocation_or_slicing(self):
        man = mock.Mock()
        with mock.patch.object(slinky.np, "arange", side_effect=AssertionError("allocated samples")):
            with self.assertRaisesRegex(ValueError, "increase the layer height or disable"):
                slinky.layer_check(man, 0.001, slinky.MAX_LAYER_SAMPLES, 0.4)
        man.slice.assert_not_called()


if __name__ == "__main__":
    unittest.main()
