"""Physical mesh checks for moving connections, plus app persistence and sizing."""
import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import manifold3d as m3d
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import outline_slinky as osl
import slinky_app as app
from slinky import from_manifold, write_stl
from test_outline_slinky import SQUARE, HOLE, arguments
from test_slinky_app import STATE, generation_options


def section_centroid(section):
    # Independent polygon-area centroid, including signed hole contributions.
    area = 0.0
    moment = np.zeros(2)
    for polygon in section.to_polygons():
        q = np.roll(polygon, -1, axis=0)
        cross = polygon[:, 0] * q[:, 1] - q[:, 0] * polygon[:, 1]
        area += cross.sum() / 2
        moment += ((polygon + q) * cross[:, None]).sum(0) / 6
    return moment / area


class StaggeredGeometryTests(unittest.TestCase):
    def test_gap_sections_only_contain_the_advancing_local_connection(self):
        mesh, info = osl.build(arguments(method="staggered", turns=5, thickness=0.8,
                                         bridge_advance=25, bridge_length=6, bridge_depth=2))
        self.assertEqual(mesh.status(), m3d.Error.NoError)
        self.assertEqual(len(mesh.decompose()), 1)
        expected = ([3, -9], [9, 3], [-3, 9], [-9, -3])
        for k, center in enumerate(expected):
            gap = mesh.slice(k * 1.2 + 1.0)
            self.assertEqual(len(gap.decompose()), 1)
            self.assertAlmostEqual(gap.area(), 8.0, places=5)
            np.testing.assert_allclose(section_centroid(gap), center, atol=1e-6)
        # Away from all four ramp patches, every physical layer has the full gap.
        heights = np.array([hit.position[2] for hit in mesh.ray_cast([-9, -9, -1], [-9, -9, 10])])
        self.assertEqual(len(heights), 10)
        np.testing.assert_allclose(heights[2::2] - heights[1:-1:2], 0.4, atol=1e-7)
        self.assertAlmostEqual(info["H"], 5.6)

    def test_winding_reverses_connections_and_preserves_volume(self):
        right, _ = osl.build(arguments(method="staggered", turns=5, thickness=0.8))
        left, _ = osl.build(arguments(method="staggered", turns=5, thickness=0.8, left=True))
        self.assertAlmostEqual(right.volume(), left.volume(), places=6)
        for k in range(4):
            rc = section_centroid(right.slice(k * 1.2 + 1.0))
            lc = section_centroid(left.slice(k * 1.2 + 1.0))
            np.testing.assert_allclose(lc, [-rc[0], rc[1]], atol=1e-6)

    def test_tapers_and_corner_crossing_ramps_stay_connected_in_every_strip(self):
        for band in ("full", "inward", "centered", "outward"):
            for edge in ("none", "round", "chamfer"):
                with self.subTest(band=band, edge=edge):
                    mesh, info = osl.build(arguments(method="staggered", band=band, turns=6,
                        regions=[dict(outer=SQUARE, holes=[HOLE])], thickness=0.8,
                        bridge_advance=20, bridge_length=12, bridge_depth=1.0,
                        edge=edge, edge_size=0.3))
                    self.assertEqual(mesh.status(), m3d.Error.NoError)
                    self.assertEqual(len(mesh.decompose()), 1)
                    self.assertAlmostEqual(info["H"], 6.8)

    def test_ddr_full_profile_preserves_clearance_except_at_local_ramps(self):
        with contextlib.redirect_stdout(io.StringIO()):
            regions = osl.loops_from_dxf(str(pathlib.Path(app.APP_DIR) / "DDR_profile.dxf"))
        mesh, info = osl.build(arguments(method="staggered", band="full", regions=regions,
            turns=6, thickness=0.8, edge="round", edge_size=0.3, base=0.5))
        self.assertEqual(mesh.status(), m3d.Error.NoError)
        self.assertEqual(len(mesh.decompose()), 1)
        self.assertAlmostEqual(info["H"], 7.3)
        full = mesh.slice(0.9).area()
        for k in range(5):
            ramp = mesh.slice(0.5 + k * 1.2 + 1.0)
            self.assertEqual(len(ramp.decompose()), 1)
            self.assertGreater(ramp.area(), 0)
            self.assertLess(ramp.area(), full * 0.05)

    def test_single_lap_and_ladder_have_expected_real_height(self):
        for turns in (1, 2, 6):
            with self.subTest(turns=turns):
                args = arguments(method="staggered", turns=turns, thickness=0.8, base=0.5,
                                 gap_ladder=0.2, edge="round", edge_size=0.3)
                mesh, info = osl.build(args)
                n = turns - 1
                self.assertAlmostEqual(info["H"], 0.5 + n * 1.2 + 0.2 * n * (n - 1) / 2 + 0.8)
                captured = io.StringIO()
                with contextlib.redirect_stdout(captured):
                    osl.report(mesh, info, args)
                self.assertIn("estimates do not apply", captured.getvalue())

    def test_actual_ddr_stl_remains_watertight_at_export_precision(self):
        with contextlib.redirect_stdout(io.StringIO()):
            regions = osl.loops_from_dxf(str(pathlib.Path(app.APP_DIR) / "DDR_profile.dxf"))
        with tempfile.TemporaryDirectory(prefix="staggered-stl-roundtrip-") as directory:
            path = pathlib.Path(directory) / "coupon.stl"
            for left in (False, True):
                for edge in ("none", "round", "chamfer"):
                    with self.subTest(left=left, edge=edge):
                        mesh, _ = osl.build(arguments(method="staggered", band="full", regions=regions,
                            turns=6, scale=1.5, left=left, thickness=0.8, edge=edge, edge_size=0.3, base=0.5))
                        vertices, faces = from_manifold(mesh)
                        write_stl(path, vertices, faces)
                        records = np.fromfile(path, dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")], offset=84)
                        vertices, index = np.unique(records["v"].reshape(-1, 3), axis=0, return_inverse=True)
                        exported = m3d.Manifold(m3d.Mesh64(vertices.astype(float), index.reshape(-1, 3).astype(np.uint64)))
                        self.assertEqual(exported.status(), m3d.Error.NoError)
                        self.assertEqual(len(exported.decompose()), 1)
                        self.assertAlmostEqual(exported.bounding_box()[5], 7.3, places=5)

    def test_invalid_bridge_settings_fail_before_mesh_work(self):
        for changes in (dict(method="unknown"), dict(bridge_advance=0), dict(bridge_advance=100),
                        dict(bridge_advance=float("nan")), dict(bridge_depth=-1), dict(bridge_length=0)):
            with self.subTest(changes=changes), mock.patch.object(osl, "staggered_laps") as builder:
                with self.assertRaises(ValueError):
                    osl.build(arguments(**(dict(method="staggered") | changes)))
                builder.assert_not_called()


class StaggeredAppTests(unittest.TestCase):
    def test_height_modes_choose_nearest_attainable_height_and_round_trip(self):
        for step in (0, 0.2, 6):
            for target in (0.1, 1.3, 4.5, 10, 75):
                with self.subTest(step=step, target=target):
                    height = lambda n: 0.5 + (n - 1) * 1.2 + step * (n - 1) * (n - 2) / 2 + 0.8
                    expected = min(range(1, 301), key=lambda n: abs(height(n) - target))
                    d = app.derive(STATE, "Scale factor", 1, "Height (mm)", target,
                                   0.8, 0.4, False, 0.2, 0.5, step > 0, step, "Staggered perimeter bridges")
                    self.assertEqual(d["laps"], expected)
                    self.assertAlmostEqual(d["H"], height(expected))
        update, _ = app.convert_height_mode(STATE, "Laps", "Height (mm)", 6,
            0.8, 0.4, False, 0.2, 0.5, False, 0.2, "Staggered perimeter bridges")
        self.assertAlmostEqual(update["value"], 7.3)

    def test_real_generation_saves_and_restores_bridge_controls(self):
        with tempfile.TemporaryDirectory(prefix="staggered-slinky-test-") as directory:
            with mock.patch.object(app, "gr_cache_dir", return_value=directory):
                _, report, _ = app.generate(**generation_options(directory, method="Staggered perimeter bridges",
                    height_val=5, bridge_advance=20, bridge_length=8, bridge_depth=1.5))
                project = pathlib.Path(directory) / "square.slinky.json"
                loaded = app.load_into_ui(str(project), "auto")
                raw = json.loads(project.read_text(encoding="utf-8"))
            restored = dict(zip(app.SETTING_KEYS, loaded[3:-2]))
            self.assertEqual(raw["params"]["method"], "staggered")
            self.assertEqual(restored["method"], "Staggered perimeter bridges")
            self.assertEqual(restored["bridge_advance"], 20)
            self.assertEqual(restored["bridge_length"], 8)
            self.assertEqual(restored["bridge_depth"], 1.5)
            self.assertAlmostEqual(raw["params"]["height_mm"], 6.6)
            self.assertIn("larger unsupported areas", report)
            self.assertIn("estimates do not apply", report)

    def test_invalid_app_controls_and_saved_method_are_rejected(self):
        for changes in (dict(method="invalid"), dict(bridge_advance=100), dict(bridge_advance=None),
                        dict(bridge_depth=0), dict(bridge_length=-1)):
            with self.subTest(changes=changes), mock.patch.object(app.osl, "build") as builder:
                with self.assertRaises(app.gr.Error):
                    app.generate(**generation_options("unused", **(dict(method="Staggered perimeter bridges") | changes)))
                builder.assert_not_called()
        with self.assertRaises(ValueError):
            app.saved_settings(dict(STATE, params=dict(method="invalid")))

    def test_real_ui_contains_method_controls_and_generation_wiring(self):
        with tempfile.TemporaryDirectory(prefix="staggered-ui-test-") as directory:
            with mock.patch.object(app, "gr_cache_dir", return_value=directory):
                demo = app.make_ui()
            config = demo.get_config_file()
        components = {item.get("props", {}).get("elem_id"): item for item in config["components"]}
        method = components["slinky-setting-method"]
        self.assertEqual(method["props"]["value"], "Continuous helix")
        for key in ("bridge_advance", "bridge_length", "bridge_depth"):
            self.assertIn("slinky-setting-" + key, components)
        generate = next(fn for fn in demo.fns.values() if fn.fn == app.generate_for_ui)
        self.assertEqual([component.elem_id for component in generate.inputs[-5:-1]],
                         ["slinky-setting-" + key for key in ("method", "bridge_advance", "bridge_length", "bridge_depth")])
        demo.close()


if __name__ == "__main__":
    unittest.main()
