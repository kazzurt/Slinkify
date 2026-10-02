"""Regression tests for the app's import, sizing and safe export workflows.

The generation smoke test uses a small square band and temporary output/cache
folders, so it exercises the real mesh pipeline without modifying user results.
"""
import contextlib
import io
import json
import pathlib
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import gradio as gr
import slinky_app as app


REGIONS = [{"outer": [[0, 0], [20, 0], [20, 20], [0, 20]], "holes": []}]
STATE = dict(name="square", params=None, regions=REGIONS, width_mm=20.0,
             height_mm=20.0, area_mm2=400.0, min_stroke_mm=20.0)


def generation_options(directory, **changes):
    options = dict(state=STATE, size_by="Scale factor", size_val=1.0,
                   height_by="Laps", height_val=2, strip="Band, inward",
                   band_width=2.0, holes_as_coils=False, thickness=1.0,
                   layer=0.2, gap=0.4, snap_gap=True, base=0.0, top="Flat fill",
                   hand="Left", material="PETG", grid=1.0, tol=0.02,
                   min_hole=0.0, slice_check=False, out_dir=str(directory),
                   out_name="square.stl", ladder=False, ladder_step=0.2,
                   edge="None", edge_size=0.0, progress=lambda *args, **kwargs: None)
    options.update(changes)
    return options


class ProfileImportTests(unittest.TestCase):
    def test_clear_profile_clears_state_and_preview(self):
        result = app.load_into_ui(None, "auto")
        self.assertIsNone(result[0])
        self.assertIsNone(result[2])
        self.assertIn("Pick a DXF", result[1])
        self.assertEqual(len(result), len(app.SETTING_KEYS) + 5)

    def test_invalid_json_and_empty_regions_clear_state(self):
        with tempfile.TemporaryDirectory(prefix="slinky-import-test-") as directory:
            path = pathlib.Path(directory) / "invalid.json"
            for contents in ("{broken", "[]", '{"regions": []}',
                             '[{"outer": [[0, 0], [1, 1]]}]',
                             '[{"outer": [[0, 0], [1, 0], [0, NaN]]}]'):
                with self.subTest(contents=contents):
                    path.write_text(contents, encoding="utf-8")
                    state, info, preview = app.load_profile(str(path), "auto")
                    self.assertIsNone(state)
                    self.assertIsNone(preview)
                    self.assertIn("Could not load", info)

    def test_importer_system_exit_is_a_recoverable_load_error(self):
        with mock.patch.object(app.osl, "loops_from_dxf", side_effect=SystemExit("no closed loops")):
            state, info, preview = app.load_profile("bad.dxf", "auto")
        self.assertIsNone(state)
        self.assertIsNone(preview)
        self.assertIn("no closed loops", info)

    def test_saved_project_restores_legacy_parameters_without_output_paths(self):
        parameters = dict(scale=1.5, turns=4, band="centered", width=2.5,
                          thickness=0.8, layer=0.2, gap=0.6, base=0.4,
                          top="open", left=True, holes=True, material="PLA",
                          edge="chamfer", edge_size=0.2, gap_ladder=0.4,
                          min_hole_area=1.5, grid=0.4, tol=0.03,
                          no_layer_check=True, out="untrusted-output.stl")
        with tempfile.TemporaryDirectory(prefix="slinky-project-test-") as directory:
            path = pathlib.Path(directory) / "square.slinky.json"
            path.write_text(json.dumps(dict(params=parameters, regions=REGIONS)), encoding="utf-8-sig")
            with mock.patch.object(app, "gr_cache_dir", return_value=directory):
                result = app.load_into_ui(str(path), "auto")
            self.assertTrue(pathlib.Path(result[2]).is_file())
        restored = dict(zip(app.SETTING_KEYS, result[3:-2]))
        self.assertEqual(result[0]["name"], "square")
        self.assertIn("restored", result[1])
        expected = dict(size_by="Scale factor", size_val=1.5, height_by="Laps", height_val=4,
                        strip="Band, centered", band_width=2.5, thickness=0.8, layer=0.2,
                        gap=0.6, snap=False, base=0.4, top="Open", hand="Left", holes=True,
                        material="PLA", edge="Chamfer", edge_size=0.2, ladder=True,
                        ladder_step=0.4, min_hole=1.5, grid=0.4, tol=0.03, slice_check=False,
                        method="Continuous helix", bridge_advance=25.0, bridge_length=6.0, bridge_depth=2.0)
        self.assertEqual(restored, expected)
        self.assertEqual(result[-2:], ("Scale factor", "Laps"))
        self.assertNotIn("out", restored)

    def test_invalid_saved_settings_cannot_leave_a_generatable_profile(self):
        for changes in ({"scale": float("nan")}, {"material": "unknown"},
                        {"width": True}, {"snap_gap": "false"}, {"band": []},
                        {"edge": {}}, {"material": []}, {"scale": 10 ** 400},
                        {"band": "outward-typo"}, {"left": "false"}):
            state = dict(STATE, params=changes)
            with self.subTest(changes=changes), mock.patch.object(app, "load_profile", return_value=(state, "loaded", "preview.png")):
                result = app.load_into_ui("project.json", "auto")
                self.assertIsNone(result[0])
                self.assertIsNone(result[2])
                self.assertIn("Could not load this project", result[1])


class SizingTests(unittest.TestCase):
    def test_ladder_selects_closest_height_including_large_step(self):
        for step in (0.01, 0.2, 6.0):
            for target in (0.1, 2.0, 4.5, 10.0, 75.0, 400.0):
                with self.subTest(step=step, target=target):
                    height = lambda n: 0.4 + n * 1.4 + step * n * (n - 1) / 2 + 1.0
                    expected = min(range(1, 301), key=lambda n: abs(height(n) - target))
                    actual = app.derive(STATE, "Scale factor", 1, "Height (mm)", target,
                                        1.0, 0.4, False, 0.2, 0.4, True, step)
                    self.assertEqual(actual["laps"], expected)
                    self.assertAlmostEqual(actual["H"], height(expected))

    def test_ladder_handles_very_large_target_directly(self):
        laps = 1_000_000
        target = laps * 1.4 + 0.2 * laps * (laps - 1) / 2 + 1.0
        result = app.derive(STATE, "Scale factor", 1, "Height (mm)", target,
                            1.0, 0.4, False, 0.2, 0, True, 0.2)
        self.assertEqual(result["laps"], laps)
        self.assertAlmostEqual(result["H"], target)

    def test_size_mode_changes_preserve_physical_width(self):
        physical_values = {"Scale factor": 2.0, "Width (in)": 40 / 25.4, "Width (mm)": 40.0}
        for old_mode, value in physical_values.items():
            for new_mode, expected in physical_values.items():
                with self.subTest(old=old_mode, new=new_mode):
                    update, mode = app.convert_size_mode(STATE, old_mode, new_mode, value)
                    self.assertEqual(mode, new_mode)
                    self.assertEqual(update["label"], new_mode)
                    self.assertAlmostEqual(update["value"], expected)

    def test_height_mode_round_trip_preserves_laps_with_gap_ladder(self):
        settings = (1.0, 0.4, True, 0.2, 0.4, True, 0.2)
        update, _ = app.convert_height_mode(STATE, "Laps", "Height (in)", 7, *settings)
        restored, mode = app.convert_height_mode(STATE, "Height (in)", "Laps", update["value"], *settings)
        self.assertEqual(mode, "Laps")
        self.assertEqual(restored["value"], 7)

    def test_height_modes_still_convert_after_profile_is_cleared(self):
        settings = (1.2, 0.6, True, 0.2, 0.0, False, 0.2)
        update, mode = app.convert_height_mode(None, "Laps", "Height (mm)", 6, *settings)
        self.assertEqual(mode, "Height (mm)")
        self.assertAlmostEqual(update["value"], 12.0)
        restored, mode = app.convert_height_mode(None, "Height (mm)", "Laps", 12, *settings)
        self.assertEqual(mode, "Laps")
        self.assertEqual(restored["value"], 6)

    def test_extreme_height_conversion_does_not_replace_input_with_infinity(self):
        settings = (1.2, 0.6, True, 0.2, 0.0, False, 0.2)
        for value in (1e308, 10 ** 400, float("inf"), float("nan")):
            for state in (None, STATE):
                with self.subTest(value=value, loaded=state is not None):
                    update, mode = app.convert_height_mode(state, "Height (in)", "Laps", value, *settings)
                    self.assertEqual(mode, "Laps")
                    self.assertNotIn("value", update)
                    self.assertEqual(update["label"], "Laps")

    def test_half_typed_number_does_not_break_readout(self):
        result = app.readout(STATE, "Scale factor", None, "Laps", None,
                             None, None, True, None, None)
        self.assertIn("positive size", result)
        self.assertNotIn("Error", result)


class ExportTests(unittest.TestCase):
    def test_output_filename_rejects_paths_and_windows_reserved_names(self):
        for name in ("../escape", "C:\\escape.stl", "nested/file", "CON", "com1.STL",
                     "LPT9.extra.stl", "bad?.stl", "bad\x00.stl", "name..stl", ".stl"):
            with self.subTest(name=name), self.assertRaises(gr.Error):
                app.output_filename(name, "fallback")
        self.assertEqual(app.output_filename("  ", "square_slinky_2t"), "square_slinky_2t.stl")
        self.assertEqual(app.output_filename("My Coil.STL", "fallback"), "My Coil.STL")

    def test_output_pair_skips_either_existing_file_and_keeps_contents(self):
        with tempfile.TemporaryDirectory(prefix="slinky-export-test-") as directory:
            folder = pathlib.Path(directory)
            existing_stl, existing_project = folder / "coil.stl", folder / "coil-2.slinky.json"
            existing_stl.write_bytes(b"keep STL")
            existing_project.write_text("keep project", encoding="utf-8")
            with app.output_pair(directory, "coil.stl") as (stl, project):
                self.assertEqual(pathlib.Path(stl).name, "coil-3.stl")
                self.assertEqual(pathlib.Path(project).name, "coil-3.slinky.json")
                pathlib.Path(stl).write_bytes(b"new STL")
                pathlib.Path(project).write_text("new project", encoding="utf-8")
            self.assertEqual(existing_stl.read_bytes(), b"keep STL")
            self.assertEqual(existing_project.read_text(encoding="utf-8"), "keep project")
            self.assertFalse((folder / "coil-2.stl").exists())

    def test_failed_pair_write_removes_both_partial_files(self):
        with tempfile.TemporaryDirectory(prefix="slinky-export-failure-test-") as directory:
            with self.assertRaisesRegex(OSError, "disk unavailable"):
                with app.output_pair(directory, "coil.stl") as (stl, project):
                    pathlib.Path(stl).write_bytes(b"partial mesh")
                    pathlib.Path(project).write_text("partial project", encoding="utf-8")
                    raise OSError("disk unavailable")
            self.assertEqual(list(pathlib.Path(directory).iterdir()), [])

    def test_invalid_numeric_settings_fail_before_geometry(self):
        for changes in ({"size_val": 0}, {"height_val": 1001}, {"gap": -1},
                        {"thickness": float("inf")}, {"tol": None},
                        {"out_name": "../escape.stl"}):
            with self.subTest(changes=changes), mock.patch.object(app.osl, "build") as build:
                with self.assertRaises(gr.Error):
                    app.generate(**generation_options("unused", **changes))
                build.assert_not_called()

    def test_excessive_layer_checks_are_rejected_before_geometry(self):
        for laps, layer in ((1000, 0.001), (2, 0.000001)):
            with self.subTest(laps=laps, layer=layer), mock.patch.object(app.osl, "build") as build:
                with self.assertRaisesRegex(gr.Error, "sampled layers"):
                    app.generate(**generation_options("unused", height_val=laps,
                                                      layer=layer, slice_check=True))
                build.assert_not_called()

    def test_real_generation_preserves_preview_named_download_and_captures_stderr(self):
        original_report = app.osl.report

        def report_with_warning(*args):
            print("stderr capture regression marker", file=sys.stderr)
            original_report(*args)

        with tempfile.TemporaryDirectory(prefix="slinky-generation-test-") as directory:
            folder = pathlib.Path(directory)
            output, cache = folder / "generated-models", folder / "preview-cache"
            cache.mkdir()
            outer_stderr = io.StringIO()
            with mock.patch.object(app, "gr_cache_dir", return_value=str(cache)), \
                    mock.patch.object(app.osl, "report", side_effect=report_with_warning), \
                    contextlib.redirect_stderr(outer_stderr):
                preview, report, downloads = app.generate(**generation_options(output, out_name="preview.stl"))
                loaded = app.load_into_ui(str(output / "preview.slinky.json"), "auto")
            self.assertEqual(outer_stderr.getvalue(), "")
            self.assertIn("stderr capture regression marker", report)
            self.assertIn("watertight", report)
            self.assertIn("Saved STL:", report)
            self.assertIn("Saved project:", report)
            self.assertIn("Total time", report)
            self.assertTrue(pathlib.Path(preview).is_file())
            self.assertEqual(len(downloads), 2)
            self.assertNotIn(preview, downloads)
            for copy in downloads:
                self.assertEqual(pathlib.Path(copy).read_bytes(), (output / pathlib.Path(copy).name).read_bytes())
            mesh_bytes = (output / "preview.stl").read_bytes()
            triangle_count, = struct.unpack_from("<I", mesh_bytes, 80)
            self.assertGreater(triangle_count, 0)
            self.assertEqual(len(mesh_bytes), 84 + 50 * triangle_count)
            project = json.loads((output / "preview.slinky.json").read_text(encoding="utf-8"))
            self.assertEqual(project["regions"], REGIONS)
            self.assertEqual(project["params"]["out"], str(output / "preview.stl"))
            self.assertAlmostEqual(project["params"]["height_mm"], 3.8)
            self.assertIsNotNone(loaded[0])
            restored = dict(zip(app.SETTING_KEYS, loaded[3:-2]))
            self.assertEqual(restored["hand"], "Left")
            self.assertEqual(restored["strip"], "Band, inward")
            self.assertEqual(restored["height_val"], 2)
            self.assertTrue(restored["snap"])


if __name__ == "__main__":
    unittest.main()
