"""Real cache preservation and explicit storage failures for model generation."""
import errno
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest import mock

import slinky_app as app
from test_slinky_app import generation_options


class PreviewStorageTests(unittest.TestCase):
    def test_cache_pruning_only_removes_verified_copies_and_keeps_newest_two(self):
        with tempfile.TemporaryDirectory(prefix="slinky-cache-pruning-") as directory:
            root = Path(directory)
            cache, originals = root / "preview-cache", root / "saved-originals"
            cache.mkdir(); originals.mkdir()
            for k in range(4):
                original = originals / f"coil-{k}.stl"
                project = original.with_suffix(".slinky.json")
                original.write_bytes(b"original geometry " + bytes([k]))
                project.write_text(json.dumps(dict(params=dict(out=str(original)))))
                result = cache / f"result-{k}"
                result.mkdir()
                shutil.copyfile(original, result / original.name)
                shutil.copyfile(project, result / project.name)
                os.utime(result, (time.time() - (4 - k) * 60,) * 2)
            # One older cache differs from its original and must be preserved.
            (cache / "result-0" / "coil-0.stl").write_bytes(b"unsaved different geometry")
            with mock.patch.object(app, "gr_cache_dir", return_value=str(cache)):
                app.purge_cache(max_bytes=0)
            self.assertEqual({p.name for p in cache.iterdir()}, {"result-0", "result-2", "result-3"})
            self.assertEqual(len(list(originals.iterdir())), 8)
            self.assertEqual((originals / "coil-1.stl").read_bytes(), b"original geometry \x01")

    def test_disk_full_has_clear_error_before_any_result_is_saved(self):
        with tempfile.TemporaryDirectory(prefix="slinky-full-disk-") as directory:
            with mock.patch.object(app, "gr_cache_dir", return_value=directory), \
                    mock.patch.object(app.shutil, "disk_usage", return_value=shutil._ntuple_diskusage(100, 100, 0)):
                with self.assertRaisesRegex(app.gr.Error, "Not enough free disk space"):
                    app.generate(**generation_options(directory, method="Staggered perimeter bridges", bridge_length=12))
            self.assertFalse(list(Path(directory).glob("*.stl")))
            self.assertFalse(list(Path(directory).glob("*.slinky.json")))

    def test_real_live_generation_is_one_scene_with_original_print_downloads(self):
        with tempfile.TemporaryDirectory(prefix="slinky-live-export-") as directory:
            with mock.patch.object(app, "gr_cache_dir", return_value=directory):
                preview, report, downloads = app.generate(**generation_options(directory,
                    method="Staggered perimeter bridges", bridge_length=12, live_preview=True))
            self.assertTrue(Path(preview).is_file())
            self.assertIn("directly in the browser", report)
            self.assertEqual({Path(p).suffix for p in downloads}, {".stl", ".json"})
            project = json.loads((Path(directory) / "square.slinky.json").read_text())
            self.assertNotIn("live_preview", project["params"])
            self.assertNotIn("see_through", project["params"])

    def test_preview_disk_failure_preserves_saved_print_and_project(self):
        with tempfile.TemporaryDirectory(prefix="slinky-preview-full-disk-") as directory:
            with mock.patch.object(app, "gr_cache_dir", return_value=directory), \
                    mock.patch.object(app, "write_live_preview", side_effect=OSError(errno.ENOSPC, "No space left")):
                preview, report, downloads = app.generate(**generation_options(directory,
                    method="Staggered perimeter bridges", bridge_length=12, live_preview=True))
            self.assertIsNone(preview)
            self.assertIsNone(downloads)
            self.assertIn("disk filled", report)
            self.assertGreater((Path(directory) / "square.stl").stat().st_size, 84)
            project = json.loads((Path(directory) / "square.slinky.json").read_text())
            self.assertEqual(project["params"]["bridge_length"], 12)

    def test_ui_payload_keeps_print_work_separate_from_display_controls(self):
        options = generation_options("unused")
        keys = list(options)
        values = [options[key] for key in keys if key != "progress"]
        # The existing app interface accepts geometry positionally through edge_size.
        draft = dict(token="reviewed-client-token", params={"layer": 0.2})
        with mock.patch.object(app, "generate", return_value=(None, "report", draft)) as generate:
            payload, report, downloads, state, button, token = app.generate_for_ui(*values)
        self.assertIsNone(json.loads(payload)["url"])
        self.assertEqual((report, downloads), ("report", None))
        self.assertIs(state, draft)
        self.assertTrue(button["interactive"])
        self.assertEqual(token, draft["token"])
        self.assertTrue(generate.call_args.kwargs["live_preview"])
        self.assertTrue(generate.call_args.kwargs["preview_only"])

    def test_orphan_pruning_protects_live_registered_and_unrecognized_drafts(self):
        with tempfile.TemporaryDirectory(prefix="slinky-orphan-draft-cache-") as directory:
            root = Path(directory)
            cache = root / "preview-cache"
            cache.mkdir()
            folders = {}
            for name in ("draft-21-orphan", "draft-22-live", "draft-21-active", "draft-21-original"):
                folder = cache / name
                folder.mkdir()
                (folder / "preview.glb").write_bytes(b"viewer asset")
                os.utime(folder, (time.time() - 2 * 86400,) * 2)
                folders[name] = folder
            (folders["draft-21-original"] / "original.stl").write_bytes(b"preserve")
            # Adding the original changed directory mtime; restore the stale age.
            os.utime(folders["draft-21-original"], (time.time() - 2 * 86400,) * 2)
            active = str(folders["draft-21-active"].resolve())
            app.DRAFT_CACHES.add(active)
            try:
                with mock.patch.object(app, "gr_cache_dir", return_value=str(cache)), \
                        mock.patch.object(app, "process_is_running", side_effect=lambda pid: pid == 22):
                    app.purge_cache(max_bytes=0)
                    self.assertFalse(folders["draft-21-orphan"].exists())
                    self.assertTrue(folders["draft-22-live"].exists())
                    self.assertTrue(folders["draft-21-active"].exists())
                    self.assertEqual((folders["draft-21-original"] / "original.stl").read_bytes(), b"preserve")
                    self.assertTrue(app.cleanup_draft(dict(cache_dir=active)))
            finally:
                app.DRAFT_CACHES.discard(active)


if __name__ == "__main__":
    unittest.main()
