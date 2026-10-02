"""Checks that preview colors identify real ramps and leave print geometry alone."""
import json
import pathlib
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np

import outline_slinky as osl
import preview_colors as colors
import slinky_app as app
from slinky import from_manifold, write_stl
from test_outline_slinky import SQUARE, HOLE, arguments
from test_slinky_app import generation_options


def read_glb(path):
    data = pathlib.Path(path).read_bytes()
    magic, version, length = struct.unpack_from("<4sII", data)
    assert (magic, version, length) == (b"glTF", 2, len(data))
    json_length, kind = struct.unpack_from("<I4s", data, 12)
    assert kind == b"JSON" and json_length % 4 == 0
    document = json.loads(data[20:20 + json_length])
    binary_length, kind = struct.unpack_from("<I4s", data, 20 + json_length)
    binary = data[28 + json_length:]
    assert kind == b"BIN\x00" and binary_length == len(binary) == document["buffers"][0]["byteLength"]
    return document, binary


class ColoredPreviewTests(unittest.TestCase):
    def test_colors_only_connect_their_own_gap_across_corners_and_windings(self):
        for left in (False, True):
            for band in ("full", "inward", "centered", "outward"):
                with self.subTest(left=left, band=band):
                    mesh, info = osl.build(arguments(method="staggered", turns=8, thickness=0.8,
                        regions=[dict(outer=SQUARE, holes=[HOLE])], band=band,
                        bridge_advance=20, bridge_length=12, bridge_depth=1,
                        edge="round", edge_size=0.3, left=left, capture_preview=True))
                    preview = info["preview_manifold"].to_mesh()
                    labels = colors.face_materials(preview, info["preview_bridge_ids"])
                    z = preview.vert_properties[np.asarray(preview.tri_verts), 2]
                    for k in range(7):
                        midgap = k * 1.2 + 1.0
                        spanning = (z.min(1) < midgap) & (z.max(1) > midgap)
                        self.assertTrue(spanning.any())
                        # Independently inspect vertical spans: only this gap's ramp
                        # can cross its air space, even when palette colors repeat.
                        np.testing.assert_array_equal(np.unique(labels[spanning]), [1 + k % 6])
                    self.assertIn(0, labels)
                    self.assertEqual(len(mesh.decompose()), 1)

    def test_preview_capture_keeps_print_stl_byte_identical(self):
        options = dict(method="staggered", turns=6, thickness=0.8, bridge_length=12,
                       bridge_advance=20, edge="round", edge_size=0.3)
        plain, plain_info = osl.build(arguments(**options))
        colored, info = osl.build(arguments(**options, capture_preview=True))
        self.assertIsNone(plain_info["preview_manifold"])
        self.assertIsNotNone(info["preview_manifold"])
        self.assertAlmostEqual(plain.volume(), colored.volume(), places=9)
        with tempfile.TemporaryDirectory(prefix="slinky-preview-stl-check-") as directory:
            a, b = pathlib.Path(directory) / "plain.stl", pathlib.Path(directory) / "colored.stl"
            write_stl(str(a), *from_manifold(plain))
            write_stl(str(b), *from_manifold(colored))
            self.assertEqual(a.read_bytes(), b.read_bytes())

    def test_glb_embeds_valid_materials_bounds_normals_and_upright_geometry(self):
        mesh, info = osl.build(arguments(method="staggered", turns=5, thickness=0.8, capture_preview=True))
        with tempfile.TemporaryDirectory(prefix="slinky-preview-glb-check-") as directory:
            path = pathlib.Path(directory) / "preview.glb"
            colors.write_bridge_preview(path, info["preview_manifold"], info["preview_bridge_ids"])
            document, binary = read_glb(path)
        self.assertEqual(document["asset"]["version"], "2.0")
        self.assertEqual(len(document["materials"]), 5)
        self.assertEqual(document["materials"][0]["name"], "Laps")
        all_vertices = []
        for primitive in document["meshes"][0]["primitives"]:
            for attribute in ("POSITION", "NORMAL"):
                accessor = document["accessors"][primitive["attributes"][attribute]]
                view = document["bufferViews"][accessor["bufferView"]]
                vectors = np.frombuffer(binary, dtype="<f4", count=accessor["count"] * 3,
                                        offset=view["byteOffset"]).reshape(-1, 3)
                self.assertTrue(np.isfinite(vectors).all())
                self.assertEqual(accessor["count"] % 3, 0)
                self.assertLessEqual(view["byteOffset"] + view["byteLength"], len(binary))
                self.assertEqual(view["byteOffset"] % 4, 0)
                if attribute == "POSITION":
                    all_vertices.append(vectors)
                    np.testing.assert_allclose(vectors.min(0), accessor["min"])
                    np.testing.assert_allclose(vectors.max(0), accessor["max"])
                    triangles = vectors.reshape(-1, 3, 3)
                    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
                    normals /= np.linalg.norm(normals, axis=1)[:, None]
                else:
                    np.testing.assert_allclose(vectors[::3], normals, atol=1e-6)
                    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
        points = np.concatenate(all_vertices)
        np.testing.assert_allclose(points.min(0) + points.max(0), 0, atol=1e-6)
        np.testing.assert_allclose(np.ptp(points, axis=0), [20, info["H"], 20], atol=1e-6)

    def test_single_lap_has_only_gray_material(self):
        _, info = osl.build(arguments(method="staggered", turns=1, capture_preview=True))
        with tempfile.TemporaryDirectory(prefix="slinky-preview-single-lap-") as directory:
            path = pathlib.Path(directory) / "preview.glb"
            colors.write_bridge_preview(path, info["preview_manifold"], info["preview_bridge_ids"])
            document, _ = read_glb(path)
        self.assertEqual([material["name"] for material in document["materials"]], ["Laps"])

    def test_see_through_preview_has_transparent_laps_and_full_opaque_ramps(self):
        _, info = osl.build(arguments(method="staggered", turns=5, thickness=0.8, capture_preview=True))
        with tempfile.TemporaryDirectory(prefix="slinky-preview-see-through-") as directory:
            path = pathlib.Path(directory) / "preview.glb"
            colors.write_bridge_preview(path, info["preview_manifold"], info["preview_bridge_ids"],
                                        see_through=True, ramps=info["preview_ramps"])
            document, _ = read_glb(path)
        gray, *ramps = document["materials"]
        self.assertEqual(gray["alphaMode"], "BLEND")
        self.assertLess(gray["pbrMetallicRoughness"]["baseColorFactor"][3], 0.3)
        for material, primitive, (_, ramp) in zip(ramps, document["meshes"][0]["primitives"][1:], info["preview_ramps"]):
            self.assertEqual(material["pbrMetallicRoughness"]["baseColorFactor"][3], 1)
            position = document["accessors"][primitive["attributes"]["POSITION"]]
            self.assertEqual(position["count"], len(ramp.to_mesh().tri_verts) * 3)

    def test_real_app_returns_colored_glb_and_keeps_print_downloads(self):
        with tempfile.TemporaryDirectory(prefix="slinky-colored-generation-") as directory:
            with mock.patch.object(app, "gr_cache_dir", return_value=directory):
                preview, report, downloads = app.generate(**generation_options(directory,
                    method="Staggered perimeter bridges", height_val=5, thickness=0.8))
            self.assertEqual(pathlib.Path(preview).suffix, ".glb")
            document, _ = read_glb(preview)
            self.assertEqual(len(document["materials"]), 5)
            self.assertEqual([pathlib.Path(path).suffix for path in downloads], [".stl", ".json"])
            self.assertIn("Preview colors", report)
            self.assertIn("see-through laps", report)
            self.assertIn("11.3 degrees", report)
            project = json.loads((pathlib.Path(directory) / "square.slinky.json").read_text())
            self.assertNotIn("capture_preview", project["params"])

    def test_live_angle_uses_effective_gap_length_and_ladder(self):
        def readout(length=6, gap=0.4, snap=False, ladder=False):
            return app.bridge_readout("Staggered perimeter bridges", 0.8, gap, snap, 0.2, ladder, 0.2, length)
        self.assertIn("11.3°", readout())
        self.assertIn("5.7°", readout(length=12))
        self.assertIn("11.3°", readout(gap=0.31, snap=True))
        self.assertIn("higher ramps steeper", readout(ladder=True))
        self.assertEqual(app.bridge_readout("Continuous helix", 0.8, 0.4, False, 0.2, False, 0, 6), "")
        for length in (None, 0, float("nan"), 10 ** 400):
            self.assertIn("Enter positive", readout(length=length))


if __name__ == "__main__":
    unittest.main()
