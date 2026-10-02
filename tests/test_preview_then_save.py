"""Generate reviews geometry; Save exports exactly that reviewed snapshot."""
import errno
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import slinky_app as app
from slinky import write_stl
from test_slinky_app import generation_options


class PreviewThenSaveTests(unittest.TestCase):
    def preview(self, root, **changes):
        options = generation_options(root / "not-created-output", method="Staggered perimeter bridges",
                                     bridge_length=12, **changes)
        return app.generate_for_ui(**options)

    def test_generate_creates_only_glb_without_output_folder_or_downloads(self):
        with tempfile.TemporaryDirectory(prefix="slinky-preview-before-save-") as directory:
            root = Path(directory)
            cache = root / "preview-cache"
            cache.mkdir()
            with mock.patch.object(app, "gr_cache_dir", return_value=str(cache)), \
                    mock.patch.object(app, "write_stl", wraps=app.write_stl) as writer:
                payload, report, downloads, draft, button, token = self.preview(root)
                writer.assert_not_called()
                self.assertFalse((root / "not-created-output").exists())
                self.assertFalse(list(root.rglob("*.stl")))
                self.assertFalse(list(root.rglob("*.json")))
                self.assertIsNone(downloads)
                self.assertTrue(Path(draft["preview_path"]).is_file())
                self.assertIn(".glb", json.loads(payload)["url"])
                self.assertIn("have not been saved", report)
                self.assertTrue(button["interactive"])
                self.assertEqual(token, draft["token"])
                self.assertFalse(draft["vertices"].flags.writeable)
                self.assertFalse(draft["faces"].flags.writeable)
                app.cleanup_draft(draft)

    def test_save_uses_reviewed_mesh_without_rebuild_and_repeated_saves_preserve_files(self):
        with tempfile.TemporaryDirectory(prefix="slinky-save-reviewed-draft-") as directory:
            root = Path(directory)
            cache = root / "preview-cache"
            cache.mkdir()
            with mock.patch.object(app, "gr_cache_dir", return_value=str(cache)):
                _, generation_report, _, draft, _, token = self.preview(root)
                expected = root / "expected-mesh.stl"
                write_stl(expected, draft["vertices"], draft["faces"])
                expected_bytes = expected.read_bytes()
                output = root / "chosen-save-output"
                # Fields can be edited after generating. Save only accepts the
                # reviewed state and current output options, not geometry fields.
                current_controls = generation_options(output, height_val=9, bridge_length=20)
                with mock.patch.object(app.osl, "build", side_effect=AssertionError("Save must not build")):
                    report, downloads = app.save_preview(draft, current_controls["out_dir"], "reviewed", token)
                    app.save_preview(draft, str(output), "reviewed", token)
                for suffix in ("", "-2"):
                    stl = output / f"reviewed{suffix}.stl"
                    project = output / f"reviewed{suffix}.slinky.json"
                    self.assertEqual(stl.read_bytes(), expected_bytes)
                    params = json.loads(project.read_text())["params"]
                    self.assertEqual(params["turns"], 2)
                    self.assertEqual(params["bridge_length"], 12)
                    self.assertEqual(params["out"], str(stl))
                self.assertIn(draft["report"], report)
                self.assertNotIn("have not been saved", report)
                self.assertIn("Saved reviewed preview", report)
                self.assertEqual({Path(p).suffix for p in downloads}, {".stl", ".json"})
                self.assertEqual(draft["params"]["out"], "")
                app.cleanup_draft(draft)

    def test_output_filename_is_validated_on_save_and_not_on_preview(self):
        with tempfile.TemporaryDirectory(prefix="slinky-preview-invalid-save-name-") as directory:
            root = Path(directory)
            cache = root / "preview-cache"
            cache.mkdir()
            with mock.patch.object(app, "gr_cache_dir", return_value=str(cache)):
                _, _, _, draft, _, token = self.preview(root, out_name="../bad-name")
                with self.assertRaisesRegex(app.gr.Error, "Enter a file name"):
                    app.save_preview(draft, str(root / "save-output"), "../bad-name", token)
                self.assertFalse((root / "save-output").exists())
                self.assertTrue(Path(draft["preview_path"]).exists())
                app.save_preview(draft, str(root / "save-output"), "valid-name", token)
                self.assertTrue((root / "save-output" / "valid-name.stl").is_file())
                app.cleanup_draft(draft)

    def test_no_draft_and_stale_queued_save_have_clear_errors(self):
        with self.assertRaisesRegex(app.gr.Error, "Generate a preview first"):
            app.save_preview(None, "unused", "unused")
        with tempfile.TemporaryDirectory(prefix="slinky-stale-preview-save-") as directory:
            root = Path(directory)
            cache = root / "preview-cache"
            cache.mkdir()
            with mock.patch.object(app, "gr_cache_dir", return_value=str(cache)):
                _, _, _, draft, _, _ = self.preview(root)
                with self.assertRaisesRegex(app.gr.Error, "preview changed"):
                    app.save_preview(draft, str(root / "save-output"), "reviewed", "old-client-token")
                self.assertFalse((root / "save-output").exists())
                app.cleanup_draft(draft)

    def test_successful_replacement_cleans_old_draft_and_failed_replacement_retains_current(self):
        with tempfile.TemporaryDirectory(prefix="slinky-draft-cache-lifetime-") as directory:
            root = Path(directory)
            cache = root / "preview-cache"
            cache.mkdir()
            with mock.patch.object(app, "gr_cache_dir", return_value=str(cache)):
                _, _, _, old, _, _ = self.preview(root)
                _, _, _, current, _, _ = self.preview(root, previous_draft=old, height_val=3)
                self.assertFalse(Path(old["cache_dir"]).exists())
                self.assertTrue(Path(current["preview_path"]).exists())
                with mock.patch.object(app, "write_live_preview", side_effect=OSError(errno.ENOSPC, "disk full")):
                    with self.assertRaisesRegex(app.gr.Error, "disk filled"):
                        self.preview(root, previous_draft=current, height_val=4)
                self.assertTrue(Path(current["preview_path"]).exists())
                self.assertEqual([p.name for p in cache.iterdir()], [Path(current["cache_dir"]).name])
                self.assertTrue(app.cleanup_draft(current))
                self.assertFalse(list(cache.iterdir()))

    def test_cleanup_rejects_unregistered_and_outside_paths(self):
        with tempfile.TemporaryDirectory(prefix="slinky-draft-safe-cleanup-") as directory:
            root = Path(directory)
            cache, outside = root / "preview-cache", root / "draft-outside"
            cache.mkdir(); outside.mkdir()
            original = outside / "original.stl"
            original.write_bytes(b"retain original")
            app.DRAFT_CACHES.add(str(outside.resolve()))
            try:
                with mock.patch.object(app, "gr_cache_dir", return_value=str(cache)):
                    self.assertFalse(app.cleanup_draft(dict(cache_dir=str(outside))))
                    unregistered = cache / "draft-unregistered"
                    unregistered.mkdir()
                    self.assertFalse(app.cleanup_draft(dict(cache_dir=str(unregistered))))
                    self.assertTrue(unregistered.exists())
                self.assertEqual(original.read_bytes(), b"retain original")
            finally:
                app.DRAFT_CACHES.discard(str(outside.resolve()))

    def test_draft_copies_profile_and_saved_settings(self):
        with tempfile.TemporaryDirectory(prefix="slinky-draft-copy-isolation-") as directory:
            root = Path(directory)
            cache = root / "preview-cache"
            cache.mkdir()
            options = generation_options(root / "not-created-output", method="Staggered perimeter bridges")
            options["state"] = json.loads(json.dumps(options["state"]))
            with mock.patch.object(app, "gr_cache_dir", return_value=str(cache)):
                _, _, _, draft, _, token = app.generate_for_ui(**options)
                original_point = list(draft["regions"][0]["outer"][0])
                options["state"]["regions"][0]["outer"][0] = [123, 456]
                options["height_val"] = 10
                self.assertEqual(draft["regions"][0]["outer"][0], original_point)
                output = root / "save-output"
                app.save_preview(draft, str(output), "copied-profile", token)
                saved = json.loads((output / "copied-profile.slinky.json").read_text())
                self.assertEqual(saved["regions"][0]["outer"][0], original_point)
                self.assertEqual(saved["params"]["turns"], 2)
                app.cleanup_draft(draft)


if __name__ == "__main__":
    unittest.main()
