"""DDR ramps work at varied lengths without changing the free sheet gaps."""
import contextlib
import io
import pathlib
import sys
import tempfile
import unittest

import manifold3d as m3d
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import outline_slinky as osl
from slinky import from_manifold, write_stl
from test_outline_slinky import arguments


class BridgeLengthRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            cls.regions = osl.loops_from_dxf(str(pathlib.Path(osl.__file__).parent / "DDR_profile.dxf"))

    def build_ddr(self, length, turns=6):
        return osl.build(arguments(regions=self.regions, method="staggered", scale=1.5,
            thickness=0.8, gap=0.4, base=0.5, turns=turns, band="full",
            edge="round", edge_size=0.3, bridge_advance=25, bridge_depth=2,
            bridge_length=length, capture_preview=True))

    def test_varied_lengths_make_one_solid_with_one_local_ramp_per_gap(self):
        for length in (4, 6, 8, 10, 12, 15, 20):
            with self.subTest(length=length):
                man, info = self.build_ddr(length)
                self.assertEqual(man.status(), m3d.Error.NoError)
                self.assertEqual(len(man.decompose()), 1)
                self.assertAlmostEqual(info["H"], 7.3)
                full_area = man.slice(0.9).area()
                for gap in range(5):
                    section = man.slice(0.5 + 1.2 * gap + 1.0)
                    self.assertEqual(len(section.decompose()), 1)
                    self.assertGreater(section.area(), 0)
                    self.assertLess(section.area(), 0.1 * full_area)

    def test_real_binary_stl_roundtrips_for_short_long_and_tall_ramps(self):
        # The reported 15 and 20 mm failures occurred at 30 laps. Exercise the
        # actual writer's float32 records, not just the double precision solid.
        with tempfile.TemporaryDirectory(prefix="slinky-bridge-length-roundtrip-") as directory:
            path = pathlib.Path(directory) / "ddr-bridge-length-check.stl"
            for turns, length in ((6, 4), (6, 12), (30, 4), (30, 15), (30, 20)):
                with self.subTest(turns=turns, length=length):
                    man, info = self.build_ddr(length, turns)
                    write_stl(path, *from_manifold(man))
                    records = np.fromfile(path, dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")], offset=84)
                    vertices, indices = np.unique(records["v"].reshape(-1, 3), axis=0, return_inverse=True)
                    mesh = m3d.Manifold(m3d.Mesh64(vertices.astype(float), indices.reshape(-1, 3).astype(np.uint64)))
                    self.assertEqual(mesh.status(), m3d.Error.NoError)
                    self.assertEqual(len(mesh.decompose()), 1)
                    self.assertAlmostEqual(mesh.bounding_box()[5], info["H"], places=5)
                    points = records["v"].astype(float)
                    twice_area = np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1)
                    self.assertTrue((twice_area > 0).all())
                    # Independently inspect columns with all sheets present.
                    # Such columns are away from both tapered edges and ramps.
                    checked = 0
                    bb = mesh.bounding_box()
                    polygons = mesh.slice(0.9).to_polygons()
                    starts = np.concatenate(polygons)
                    vectors = np.concatenate([np.roll(p, -1, axis=0) - p for p in polygons])
                    squares = np.einsum("ij,ij->i", vectors, vectors)
                    for x in np.linspace(bb[0] + 3, bb[3] - 3, 16):
                        for y in np.linspace(bb[1] + 3, bb[4] - 3, 8):
                            delta = np.array([x, y]) - starts
                            along = np.clip(np.einsum("ij,ij->i", delta, vectors) / squares, 0, 1)
                            distance = np.linalg.norm(delta - along[:, None] * vectors, axis=1).min()
                            if distance <= 2.05:
                                continue  # Outside the deep, untapered interior.
                            hits = mesh.ray_cast([x, y, -1], [x, y, info["H"] + 1])
                            z = np.array([hit.position[2] for hit in hits])
                            if len(z) != 2 * turns:
                                continue
                            gaps = z[2::2] - z[1:-1:2]
                            np.testing.assert_allclose(gaps, 0.4, atol=1e-5)
                            checked += 1
                    self.assertGreaterEqual(checked, 5)


if __name__ == "__main__":
    unittest.main()
