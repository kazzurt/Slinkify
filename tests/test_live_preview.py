"""Geometry and serving checks for the persistent browser-only preview scene."""
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
from urllib.parse import unquote

import manifold3d as m3d
import numpy as np

import live_preview as live
import outline_slinky as osl
from test_outline_slinky import arguments


def read_scene(path):
    data = Path(path).read_bytes()
    magic, version, length = struct.unpack_from("<4sII", data)
    assert (magic, version, length) == (b"glTF", 2, len(data))
    size, kind = struct.unpack_from("<I4s", data, 12)
    assert kind == b"JSON" and size % 4 == 0
    document = json.loads(data[20:20 + size])
    binary_size, kind = struct.unpack_from("<I4s", data, 20 + size)
    binary = data[28 + size:]
    assert kind == b"BIN\x00" and binary_size == len(binary) == document["buffers"][0]["byteLength"]
    return document, binary


def positions(document, binary, node):
    primitive = document["meshes"][node["mesh"]]["primitives"][0]
    accessor = document["accessors"][primitive["attributes"]["POSITION"]]
    view = document["bufferViews"][accessor["bufferView"]]
    values = np.frombuffer(binary, "<f4", accessor["count"] * 3,
                           offset=view["byteOffset"]).reshape(-1, 3)
    np.testing.assert_allclose(values.min(0), accessor["min"])
    np.testing.assert_allclose(values.max(0), accessor["max"])
    return values


class LivePreviewTests(unittest.TestCase):
    def scene(self, base=0.0, turns=4, method="staggered"):
        manifold, info = osl.build(arguments(method=method, turns=turns, thickness=0.8,
            base=base, capture_preview=True))
        original = manifold.to_mesh()
        with tempfile.TemporaryDirectory(prefix="slinky-live-preview-check-") as directory:
            path = Path(directory) / "feature-preview.glb"
            live.write_live_preview(path, info["preview_manifold"] or manifold,
                                    info["preview_bridge_ids"], info["preview_ramps"], base=base)
            document, binary = read_scene(path)
        # A preview writer must not modify the geometry being exported to print.
        after = manifold.to_mesh()
        np.testing.assert_array_equal(original.vert_properties, after.vert_properties)
        np.testing.assert_array_equal(original.tri_verts, after.tri_verts)
        return manifold, info, document, binary

    def test_groups_are_independently_named_and_both_bridge_representations_exist(self):
        _, _, document, binary = self.scene()
        nodes = document["nodes"]
        roles = [node["extras"]["slinky_role"] for node in nodes]
        self.assertEqual(roles.count("laps"), 1)
        self.assertEqual(roles.count("bridge-exposed"), 3)
        self.assertEqual(roles.count("bridge-full"), 3)
        self.assertNotIn("base", roles)
        self.assertEqual(len({node["mesh"] for node in nodes}), len(nodes))
        self.assertEqual(document["scenes"][0]["nodes"], list(range(len(nodes))))
        all_positions = np.concatenate([positions(document, binary, node) for node in nodes])
        self.assertTrue(np.isfinite(all_positions).all())
        np.testing.assert_allclose(np.ptp(all_positions, axis=0), [20, 4.4, 20], atol=1e-6)
        for node in nodes:
            self.assertTrue(node["name"].startswith("slinky:" + node["extras"]["slinky_role"]))
            primitive = document["meshes"][node["mesh"]]["primitives"][0]
            material = document["materials"][primitive["material"]]
            self.assertEqual(material["name"], node["name"])
            self.assertEqual(material["pbrMetallicRoughness"]["baseColorFactor"][3], 1)

    def test_base_is_separate_and_clipped_to_the_requested_height(self):
        base = 0.5
        manifold, _, document, binary = self.scene(base=base)
        center_z = (manifold.bounding_box()[2] + manifold.bounding_box()[5]) / 2
        lower = next(node for node in document["nodes"] if node["extras"]["slinky_role"] == "base")
        lower_z = positions(document, binary, lower)[:, 1] + center_z
        self.assertAlmostEqual(float(lower_z.min()), 0.0, places=6)
        self.assertAlmostEqual(float(lower_z.max()), base, places=6)
        for node in document["nodes"]:
            if node == lower:
                continue
            z = positions(document, binary, node)[:, 1] + center_z
            self.assertGreaterEqual(float(z.min()), base - 1e-6)
        exposed = [node for node in document["nodes"] if node["extras"]["slinky_role"] == "bridge-exposed"]
        self.assertEqual(len(exposed), 3)
        for lap, node in enumerate(exposed):
            z = positions(document, binary, node)[:, 1].reshape(-1, 3) + center_z
            mid_gap = base + lap * 1.2 + 1.0
            self.assertTrue(((z.min(1) < mid_gap) & (z.max(1) > mid_gap)).any())

    def test_full_ramps_retain_buried_attachments_and_have_valid_normals(self):
        manifold, _, document, binary = self.scene()
        center_z = (manifold.bounding_box()[2] + manifold.bounding_box()[5]) / 2
        full = next(node for node in document["nodes"] if node["name"] == "slinky:bridge-full:orange")
        exposed = next(node for node in document["nodes"] if node["name"] == "slinky:bridge-exposed:orange")
        full_positions = positions(document, binary, full)
        exposed_positions = positions(document, binary, exposed)
        z = full_positions[:, 1] + center_z
        self.assertAlmostEqual(float(z.min()), 0.0, places=6)
        self.assertAlmostEqual(float(z.max()), 2.0, places=6)
        self.assertNotEqual(full_positions.tobytes(), exposed_positions.tobytes())
        for node in document["nodes"]:
            primitive = document["meshes"][node["mesh"]]["primitives"][0]
            points = positions(document, binary, node).reshape(-1, 3, 3)
            a = document["accessors"][primitive["attributes"]["NORMAL"]]
            view = document["bufferViews"][a["bufferView"]]
            normals = np.frombuffer(binary, "<f4", a["count"] * 3, offset=view["byteOffset"]).reshape(-1, 3)
            cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
            lengths = np.linalg.norm(cross, axis=1)
            self.assertTrue((lengths > 0).all())
            np.testing.assert_allclose(normals[::3], cross / lengths[:, None], atol=1e-6)
            np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-6)

    def test_single_lap_and_helix_have_no_bridge_groups(self):
        for method, turns in (("staggered", 1), ("helix", 3)):
            with self.subTest(method=method):
                _, _, document, _ = self.scene(method=method, turns=turns)
                self.assertEqual([node["extras"]["slinky_role"] for node in document["nodes"]], ["laps"])

    def test_asymmetric_profile_and_print_height_project_upright_in_default_lh_camera(self):
        # An L-shaped solid has distinct right, upper, and lower landmarks.
        # Follow actual GLB vertices through Babylon's documented LH root
        # quaternion/scale, then independently build its LH view basis. This
        # catches a mirrored or upside-down profile while checking the base.
        outline = np.array([[0, 0], [8, 0], [8, 2], [3, 2], [3, 5], [0, 5]])
        manifold = m3d.CrossSection([outline]).extrude(3)
        original = manifold.to_mesh()
        raw = original.vert_properties[np.asarray(original.tri_verts), :3].reshape(-1, 3)
        with tempfile.TemporaryDirectory(prefix="slinky-preview-orientation-") as directory:
            path = Path(directory) / "asymmetric-profile.glb"
            live.write_live_preview(path, manifold, {})
            document, binary = read_scene(path)
        glb = positions(document, binary, document["nodes"][0])
        self.assertEqual(len(glb), len(raw))
        rotation_y_180 = np.diag([-1, 1, -1])  # glTF root quaternion [0,1,0,0]
        root_scale = np.diag([1, 1, -1])
        world = glb @ (rotation_y_180 @ root_scale).T
        beta = np.radians(50)
        eye = 30 * np.array([0, np.cos(beta), np.sin(beta)])
        forward = -eye / np.linalg.norm(eye)
        right = np.cross([0, 1, 0], forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)

        def project(landmark):
            index = np.flatnonzero(np.all(np.isclose(raw, landmark), axis=1))[0]
            point = world[index] - eye
            distance = np.dot(point, forward)
            self.assertGreater(distance, 0)
            return np.array([np.dot(point, right), np.dot(point, up)]) / distance

        lower = project([0, 0, 0])
        upper = project([0, 0, 3])
        rightward = project([8, 0, 3])
        upward = project([0, 5, 3])
        self.assertGreater(rightward[0], upper[0], "Original +X must appear to the right")
        self.assertGreater(upward[1], upper[1], "Original +Y must appear above")
        self.assertGreater(upper[1], lower[1], "Print +Z must rise above the base")
        # Transforming from source XYZ to glTF must preserve triangle orientation.
        raw_cross = np.cross(raw.reshape(-1, 3, 3)[:, 1] - raw.reshape(-1, 3, 3)[:, 0],
                             raw.reshape(-1, 3, 3)[:, 2] - raw.reshape(-1, 3, 3)[:, 0])
        glb_triangles = glb.reshape(-1, 3, 3)
        glb_cross = np.cross(glb_triangles[:, 1] - glb_triangles[:, 0],
                             glb_triangles[:, 2] - glb_triangles[:, 0])
        np.testing.assert_allclose(glb_cross, raw_cross[:, [0, 2, 1]] * [1, 1, -1], atol=1e-6)

    def test_payload_quotes_and_encodes_file_names(self):
        path = Path(tempfile.gettempdir()) / 'Slinky preview & "quoted" #1.glb'
        payload = json.loads(live.preview_payload(path, "Staggered perimeter bridges"))
        self.assertEqual(payload["method"], "Staggered perimeter bridges")
        self.assertEqual(unquote(payload["url"].split("file=", 1)[1]), path.resolve().as_posix())
        self.assertNotIn('"quoted"', payload["url"])
        self.assertNotIn("#", payload["url"])
        self.assertIsNone(json.loads(live.preview_payload(None, "helix"))["url"])

    def test_component_keeps_template_static_and_uses_local_client_controls(self):
        component = live.preview_component()
        self.assertNotIn("${value}", component.html_template)
        self.assertEqual(component.html_template.count('data-feature="'), 7)
        self.assertIn(live.viewer_asset_name(), component.js_on_load)
        self.assertIn("watch('value', loadValue)", component.js_on_load)
        apply = component.js_on_load.split("function applyDisplay() {", 1)[1].split("async function ensureViewer()", 1)[0]
        self.assertIn("mesh.setEnabled(visible)", apply)
        self.assertIn("details.markSceneMutated()", apply)
        for unwanted in ("loadModel", "resetCamera", "fetch(", "server.", "trigger("):
            self.assertNotIn(unwanted, apply)

    @unittest.skipUnless(shutil.which("node"), "Node.js is needed to exercise camera fit JavaScript")
    def test_camera_fit_projects_all_model_corners_inside_narrow_and_wide_viewports(self):
        # Execute the actual client fit function, then independently project a
        # DDR-sized bounding box into its perspective camera. This catches the
        # portrait-viewport clipping that a fixed radius/diagonal check misses.
        component = live.preview_component()
        fit = "function fitNewModel() {" + component.js_on_load.split(
            "function fitNewModel() {", 1)[1].split("async function loadValue()", 1)[0]
        harness = r"""
const cases = [];
let canvas, details;
class Vector {
    constructor(x=0,y=0,z=0) { this.x=x;this.y=y;this.z=z; }
    clone() { return new Vector(this.x,this.y,this.z); }
    copyFromFloats(x,y,z) { this.x=x;this.y=y;this.z=z;return this; }
}
for (const [width,height] of [[415,460],[200,460],[900,460]]) {
    for (const fovMode of [0,1]) {
        const size=[120.85,7.3,38.68],center=[7,3,-4];
        const camera={fov:Math.PI/4,fovMode,alpha:Math.PI/2,beta:50*Math.PI/180,
                      radius:100,upperRadiusLimit:500,target:new Vector(),
                      stopInterpolation(){},setTarget(v){this.target=v;}};
        canvas={getBoundingClientRect:()=>({width,height})};
        details={model:{getWorldBounds:()=>({size,center})},camera,markSceneMutated(){}};
        fitNewModel();
        cases.push({width,height,size,center,fovMode,radius:camera.radius,
                    target:[camera.target.x,camera.target.y,camera.target.z]});
    }
}
process.stdout.write(JSON.stringify(cases));
"""
        result = subprocess.run([shutil.which("node"), "-e", fit + harness],
                                capture_output=True, text=True, check=True)
        for case in json.loads(result.stdout):
            np.testing.assert_allclose(case["target"], case["center"])
            beta = np.radians(50)
            outward = np.array([0, np.cos(beta), np.sin(beta)])
            right = np.array([1, 0, 0])
            up = np.cross(outward, right)
            aspect = case["width"] / case["height"]
            tan_half = np.tan(np.pi / 8)
            tan_vertical = tan_half / aspect if case["fovMode"] == 1 else tan_half
            tan_horizontal = tan_half if case["fovMode"] == 1 else tan_half * aspect
            for x in (-1, 1):
                for y in (-1, 1):
                    for z in (-1, 1):
                        point = np.array(case["size"]) * [x, y, z] / 2
                        distance = case["radius"] - np.dot(point, outward)
                        self.assertGreater(distance, 0)
                        horizontal = abs(np.dot(point, right) / (distance * tan_horizontal))
                        vertical = abs(np.dot(point, up) / (distance * tan_vertical))
                        self.assertLess(horizontal, 1)
                        self.assertLess(vertical, 1)

    @unittest.skipUnless(shutil.which("node"), "Node.js is needed to exercise async preview readiness")
    def test_save_stays_disabled_until_current_preview_is_rendered_and_after_load_failure(self):
        component = live.preview_component()
        gate = "function saveButton() {" + component.js_on_load.split(
            "function saveButton() {", 1)[1].split("function available()", 1)[0]
        load = "async function loadValue() {" + component.js_on_load.split(
            "async function loadValue() {", 1)[1].split("controls.addEventListener", 1)[0]
        harness = r"""
(async () => {
    let checks=[], deferred;
    let button={tagName:'BUTTON',disabled:false};
    let document={getElementById:()=>button};
    let previewReady=false,requestNumber=0,currentUrl=null,disposed=false;
    let props={value:JSON.stringify({url:'/preview-A.glb'})},status={textContent:''};
    let details={camera:{}},events=[],loadCount=0;
    let viewer={loadModel:async url=>{loadCount++;if(url.includes('failure'))throw Error('load failure');await new Promise(resolve=>deferred=resolve);},resetModel:async()=>{}};
    const ensureViewer=async()=>{};
    const fitNewModel=()=>events.push('fit');
    const resetLayers=()=>{};
    const configureLayers=()=>{};
    const available=()=>{};
    const applyDisplay=()=>events.push('display');
    const requestAnimationFrame=callback=>setImmediate(()=>{events.push('frame');callback(0);});
    const console={error:()=>{}};
    __FUNCTIONS__
    setSaveReadiness(false);
    checks.push(['initial-disabled',button.disabled]);
    const pending=loadValue();
    await Promise.resolve();await Promise.resolve();
    checks.push(['loading-disabled',button.disabled]);
    // Simulate Gradio applying interactive=True before the viewer has finished.
    button.disabled=false;enforceSaveReadiness();
    checks.push(['backend-enable-guarded',button.disabled]);
    const click={target:{closest:()=>true},preventDefault(){this.prevented=true;},stopImmediatePropagation(){this.stopped=true;}};
    blockUnreadySave(click);
    checks.push(['click-capture-guarded',click.prevented===true && click.stopped===true]);
    deferred();await pending;
    checks.push(['ready-enabled',previewReady && !button.disabled]);
    checks.push(['ready-after-fit-display-frame',events.join(',')==='fit,display,frame']);
    await loadValue();
    checks.push(['same-url-does-not-reload',loadCount===1 && !button.disabled]);
    props.value=JSON.stringify({url:'/failure.glb'});await loadValue();
    checks.push(['failed-disabled',!previewReady && button.disabled]);
    button.disabled=false;enforceSaveReadiness();
    checks.push(['failed-server-enable-guarded',button.disabled]);
    props.value=JSON.stringify({url:null});await loadValue();
    checks.push(['reset-disabled',!previewReady && button.disabled]);
    process.stdout.write(JSON.stringify(checks));
})().catch(error=>{process.stderr.write(String(error));process.exitCode=1;});
""".replace("__FUNCTIONS__", gate + load)
        result = subprocess.run([shutil.which("node"), "-e", harness],
                                capture_output=True, text=True, check=True)
        for description, passed in json.loads(result.stdout):
            self.assertTrue(passed, description)


if __name__ == "__main__":
    unittest.main()
