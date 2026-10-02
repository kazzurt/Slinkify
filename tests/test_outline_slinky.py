"""Regression coverage for outline geometry and bounded rise-field work."""
import contextlib
import io
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

import manifold3d as m3d
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import outline_slinky as osl


SQUARE = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=float)
HOLE = np.array([[5, 5], [5, 15], [15, 15], [15, 5]], dtype=float)


def arguments(**changes):
    values = dict(regions=[{"outer": SQUARE, "holes": []}], step="", dxf="", loops="",
                  circle=20.0, save_loops="", scale=1.0, tol=0.02, width=2.0, thickness=1.0,
                  band="inward", grid=1.0, gap=0.4, turns=3, edge="none", edge_size=0.0,
                  edge_step=0.0, gap_ladder=0.0, base=0.0, top="fill", left=False, holes=False,
                  min_hole_area=0.0, layer=0.2, material="PLA", no_layer_check=True)
    values.update(changes)
    return types.SimpleNamespace(**values)


class OutlineGeometryTests(unittest.TestCase):
    def assert_usable(self, mesh):
        self.assertEqual(mesh.status(), m3d.Error.NoError)
        self.assertFalse(mesh.is_empty())
        self.assertGreater(mesh.volume(), 0)

    def test_both_hands_work_for_every_band(self):
        for band in ("inward", "centered", "outward"):
            with self.subTest(band=band):
                right, _ = osl.coil_from_contour(SQUARE, 2, 1, 1.4, 3, band=band)
                left, _ = osl.coil_from_contour(SQUARE, 2, 1, 1.4, 3, band=band, left=True)
                self.assert_usable(right)
                self.assert_usable(left)
                self.assertAlmostEqual(right.volume(), left.volume(), places=6)
                self.assertAlmostEqual(left.bounding_box()[5], 5.2)

    def test_full_profile_handedness_preserves_hole_and_inward_slit(self):
        meshes, rises = [], []
        for left in (False, True):
            mesh, rise = osl.coil_full_region(SQUARE, [HOLE], 1, 1.4, 3, 1, left=left)
            self.assert_usable(mesh)
            self.assertTrue(rise["hit_hole"])
            self.assertGreater(rise["slit_len"], 4)
            self.assertAlmostEqual(mesh.bounding_box()[5], 5.2)
            meshes.append(mesh)
            rises.append(rise)
        np.testing.assert_allclose(rises[0]["N0"], rises[1]["N0"])
        self.assertAlmostEqual(meshes[0].volume(), meshes[1].volume(), places=6)

    def test_full_profile_requires_an_opening_and_a_seam_that_reaches_it(self):
        with self.assertRaisesRegex(ValueError, "interior opening"):
            osl.coil_full_region(SQUARE, [], 1.2, 1.8, 4, 0.5)
        offset_hole = np.array([[1, 5], [1, 15], [4, 15], [4, 5]], dtype=float)
        with self.assertRaisesRegex(ValueError, "seam does not reach"):
            osl.coil_full_region(SQUARE, [offset_hole], 1.2, 1.8, 4, 0.5)

    def test_band_cannot_remove_the_entire_interior_opening(self):
        for band, width in (("inward", 10), ("centered", 20)):
            with self.subTest(band=band), self.assertRaisesRegex(ValueError, "Reduce band width"):
                osl.coil_from_contour(SQUARE, width, 1.2, 1.8, 4, band=band)

    def test_tapered_annulus_has_one_solid_and_an_unblocked_seam(self):
        for left in (False, True):
            for edge in ("round", "chamfer"):
                with self.subTest(left=left, edge=edge):
                    mesh, rise = osl.coil_full_region(
                        SQUARE, [HOLE], 1.2, 1.8, 4, 0.3, left=left,
                        edge=edge, edge_size=0.3, edge_step=0.1)
                    self.assert_usable(mesh)
                    self.assertEqual(len(mesh.decompose()), 1)
                    self.assertEqual(mesh.genus(), 2)  # two deliberately filled annular ends
                    seam = rise["C0"] + 2.5 * rise["N0"]
                    for sign in (-1, 1):
                        point = seam + sign * 1e-5 * rise["T0"]
                        heights = np.array([hit.position[2] for hit in
                                            mesh.ray_cast([*point, -1], [*point, 10])])
                        self.assertEqual(len(heights), 8)
                        np.testing.assert_allclose(heights[2::2] - heights[1:-1:2], 0.6, atol=1e-7)
                        # Match the same physical gap on the two sides of the seam.
                        self.assertFalse(any(low < 3.3 < high
                                             for low, high in zip(heights[::2], heights[1::2])))
                    crossings = mesh.ray_cast([*(seam - 0.1 * rise["T0"]), 3.3],
                                              [*(seam + 0.1 * rise["T0"]), 3.3])
                    self.assertEqual(crossings, [])

    def test_ddr_taper_does_not_create_internal_cracks_or_void_shells(self):
        path = pathlib.Path(__file__).resolve().parents[1] / "DDR_profile.dxf"
        with contextlib.redirect_stdout(io.StringIO()):
            regions = osl.loops_from_dxf(str(path))
        points = [(-17.65998973, -10.75981838), (-9.04756349, 9.80680036),
                  (-30.15945714, -12.45380420)]
        for band, expected_genus in (("full", 26), ("inward", 2)):
            with self.subTest(band=band), contextlib.redirect_stdout(io.StringIO()):
                mesh, _ = osl.build(arguments(
                    regions=regions, band=band, width=3, thickness=1.2, gap=0.6, turns=6,
                    grid=0.15, min_hole_area=2, edge="round", edge_size=0.3, edge_step=0.1))
                self.assert_usable(mesh)
                self.assertEqual(len(mesh.decompose()), 1)
                self.assertEqual(mesh.genus(), expected_genus)
                for point in points:
                    heights = np.array([hit.position[2] for hit in
                                        mesh.ray_cast([*point, -1], [*point, 13])])
                    self.assertEqual(len(heights), 12)
                    # A sample on a tapered edge can have extra clearance, but
                    # cannot acquire a crack inside a lap or a reduced main gap.
                    self.assertTrue(np.all(heights[2::2] - heights[1:-1:2] >= 0.6 - 1e-7))
                    if point == points[0]:  # this sample is inside both full-width cores
                        np.testing.assert_allclose(heights[2::2] - heights[1:-1:2], 0.6, atol=1e-7)

    def test_single_open_lap_retains_sloped_top(self):
        flat, _ = osl.coil_from_contour(SQUARE, 2, 1, 1.4, 1, base=2, top="fill")
        opened, _ = osl.coil_from_contour(SQUARE, 2, 1, 1.4, 1, base=2, top="open")
        self.assert_usable(opened)
        self.assertLess(opened.volume(), flat.volume())
        self.assertAlmostEqual(opened.bounding_box()[2], 0)
        self.assertAlmostEqual(opened.bounding_box()[5], 4.4)

    def test_ladder_height_with_edge_tapers(self):
        args = arguments(turns=4, gap_ladder=0.2, edge="round", edge_size=0.3, base=0.6)
        mesh, info = osl.build(args)
        self.assert_usable(mesh)
        self.assertAlmostEqual(info["H"], 0.6 + sum([1.4, 1.6, 1.8, 2.0]) + 1.0)
        np.testing.assert_allclose(info["pitches"], [1.4, 1.6, 1.8, 2.0])

    def test_disabled_taper_has_no_extra_slice_clearance(self):
        _, info = osl.build(arguments(edge="none", edge_size=0.3))
        self.assertEqual(info["edge_size"], 0.0)

    def test_filled_single_lap_does_not_inset_narrow_band(self):
        mesh, _ = osl.coil_from_contour(SQUARE, 0.1, 1, 1.4, 1, edge="round", edge_size=0.3)
        self.assert_usable(mesh)

    def test_large_tolerance_reports_removed_loop(self):
        with self.assertRaisesRegex(ValueError, "reduce outline tolerance"):
            osl.build(arguments(tol=100))

    def test_invalid_settings_fail_before_build(self):
        for changes in ({"turns": 0}, {"turns": 1.5}, {"turns": 1001}, {"thickness": 0},
                        {"scale": -1}, {"gap": -0.1}, {"width": float("nan")},
                        {"band": "full", "grid": 0}, {"regions": []}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                osl.build(arguments(**changes))

    def test_project_json_can_supply_profile(self):
        with tempfile.TemporaryDirectory(prefix="slinky-geometry-test-") as directory:
            path = pathlib.Path(directory) / "example.slinky.json"
            path.write_text(json.dumps({"params": {}, "regions": [{"outer": SQUARE.tolist()}]}),
                            encoding="utf-8")
            mesh, _ = osl.build(arguments(regions=None, loops=str(path)))
            self.assert_usable(mesh)


class RiseFieldTests(unittest.TestCase):
    def test_nearest_s_matches_full_projection_across_chunks(self):
        rng = np.random.default_rng(2026)
        angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        polygon = np.column_stack((np.cos(angles), np.sin(angles))) * 20
        points = rng.uniform(-25, 25, (800, 2))
        seg, length, start, total = osl.polyline_lengths(polygon)
        delta = points[:, None, :] - polygon[None, :, :]
        t = np.clip(np.einsum("nmk,mk->nm", delta, seg) / length[None, :] ** 2, 0, 1)
        distance = ((delta - t[:, :, None] * seg) ** 2).sum(-1)
        nearest = np.argmin(distance, axis=1)
        expected = start[nearest] + t[np.arange(len(points)), nearest] * length[nearest]
        actual, actual_total = osl.nearest_s(points, polygon)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        self.assertEqual(actual_total, total)

    def test_tiny_grid_is_rejected_before_allocation(self):
        with mock.patch.object(osl.np, "meshgrid") as meshgrid:
            with self.assertRaisesRegex(ValueError, "increase grid spacing"):
                osl.harmonic_rise(SQUARE, [], 1e-6)
            meshgrid.assert_not_called()

    def test_coarse_grid_gives_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "reduce grid spacing"):
            osl.harmonic_rise(SQUARE, [], 100)

    def test_rise_sampling_is_reused_across_laps(self):
        counts = []
        for turns in (4, 12):
            with mock.patch.object(osl, "nearest_s", wraps=osl.nearest_s) as sample:
                mesh, _ = osl.coil_from_contour(SQUARE, 2, 1, 1.4, turns)
                self.assertEqual(mesh.status(), m3d.Error.NoError)
                counts.append(sample.call_count)
        self.assertGreater(counts[0], 0)
        self.assertEqual(counts[0], counts[1])

    def test_report_handles_no_slice_samples(self):
        args = arguments(no_layer_check=False)
        mesh, info = osl.build(args)
        output = io.StringIO()
        with mock.patch.object(osl, "layer_check", return_value=[]), contextlib.redirect_stdout(output):
            osl.report(mesh, info, args)
        self.assertIn("no bridged layers to measure", output.getvalue())

    def test_report_explains_skipped_ladder_slice_check(self):
        args = arguments(no_layer_check=False, gap_ladder=0.2)
        mesh, info = osl.build(args)
        output = io.StringIO()
        with mock.patch.object(osl, "layer_check") as check, contextlib.redirect_stdout(output):
            osl.report(mesh, info, args)
        check.assert_not_called()
        self.assertIn("skipped for gap ladder", output.getvalue())
        self.assertIn("variable-pitch layer checks are not supported", output.getvalue())


if __name__ == "__main__":
    unittest.main()
