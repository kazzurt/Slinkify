"""Print-layer inspection clips the reviewed scene without changing exports."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

import live_preview as live
import slinky_app as app
from slinky import write_stl
from test_slinky_app import generation_options


def client_function(name):
    """Read a complete function from the actual viewer client, including braces."""
    source = live.preview_component().js_on_load
    match = re.search(rf"^function {re.escape(name)}\([^{{]*\{{", source, re.MULTILINE)
    if match is None:
        raise AssertionError(f"Viewer client function {name} is missing")
    depth, quote, escaped, comment = 0, None, False, None
    offset = match.end() - 1
    while offset < len(source):
        char, pair = source[offset], source[offset:offset + 2]
        if comment == "line":
            if char == "\n":
                comment = None
        elif comment == "block":
            if pair == "*/":
                comment = None
                offset += 1
        elif quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif pair in ("//", "/*"):
            comment = "line" if pair == "//" else "block"
            offset += 1
        elif char in ("'", '"', "`"):
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[match.start():offset + 1]
        offset += 1
    raise AssertionError(f"Viewer client function {name} is incomplete")


class LayerPreviewSnapshotTests(unittest.TestCase):
    def test_payload_supports_default_and_selected_layer_height(self):
        default = json.loads(live.preview_payload("preview.glb", "helix"))
        self.assertEqual(default["layer_height"], 0.2)
        self.assertIsNone(json.loads(live.preview_payload(None, "helix"))["url"])
        custom = json.loads(live.preview_payload("preview.glb", "staggered", layer_height=0.35))
        self.assertEqual(custom["layer_height"], 0.35)

    def test_generated_layer_metadata_and_saved_mesh_use_reviewed_snapshot(self):
        with tempfile.TemporaryDirectory(prefix="slinkify-layer-inspection-") as directory:
            root = Path(directory)
            cache = root / "preview-cache"
            cache.mkdir()
            output = root / "reviewed-export"
            options = generation_options(output, method="Staggered perimeter bridges",
                                         bridge_length=12, layer=0.35, snap_gap=False)
            with mock.patch.object(app, "gr_cache_dir", return_value=str(cache)), \
                    mock.patch.object(app, "write_stl", wraps=app.write_stl) as writer:
                payload, _, downloads, draft, _, token = app.generate_for_ui(**options)
                writer.assert_not_called()
                self.assertIsNone(downloads)
                self.assertFalse(output.exists())
                self.assertEqual(json.loads(payload)["layer_height"], 0.35)
                self.assertEqual(draft["params"]["layer"], 0.35)
                original_vertices, original_faces = draft["vertices"].copy(), draft["faces"].copy()
                expected = root / "full-reviewed-model.stl"
                write_stl(expected, original_vertices, original_faces)
                # Controls may change while the loaded preview is being inspected.
                options["layer"] = 0.1
                with mock.patch.object(app.osl, "build", side_effect=AssertionError("Save must not rebuild")):
                    app.save_preview(draft, str(output), "inspected-model", token)
                self.assertEqual((output / "inspected-model.stl").read_bytes(), expected.read_bytes())
                saved = json.loads((output / "inspected-model.slinky.json").read_text())
                self.assertEqual(saved["params"]["layer"], 0.35)
                np.testing.assert_array_equal(draft["vertices"], original_vertices)
                np.testing.assert_array_equal(draft["faces"], original_faces)
                self.assertFalse(draft["vertices"].flags.writeable)
                self.assertFalse(draft["faces"].flags.writeable)
                app.cleanup_draft(draft)


@unittest.skipUnless(shutil.which("node"), "Node.js is needed to exercise print-layer JavaScript")
class LayerPreviewClientTests(unittest.TestCase):
    def run_client(self, scenario):
        functions = "\n".join(client_function(name) for name in (
            "setLayerPlanes", "resetLayers", "configureLayers", "applyLayerPreview",
            "moveLayer", "showFullModel", "applyDisplay"))
        harness = r"""
class Vector {
    constructor(x=0,y=0,z=0) { this.x=x;this.y=y;this.z=z; }
    clone() { return new Vector(this.x,this.y,this.z); }
    copyFromFloats(x,y,z) { this.x=x;this.y=y;this.z=z;return this; }
}
class RangeControl {
    constructor() { this.disabled=true;this.attributes={};this._min=1;this._max=1;this._value=1; }
    get min() { return String(this._min); }
    set min(value) { this._min=Number(value);this.sanitize(this._value); }
    get max() { return String(this._max); }
    set max(value) { this._max=Number(value);this.sanitize(this._value); }
    get value() { return String(this._value); }
    set value(value) { this.sanitize(Number(value)); }
    sanitize(value) {
        const max=Math.max(this._min,this._max);
        const candidate=Number.isFinite(value) ? value : (this._min+max)/2;
        const stepped=this._min+Math.round(candidate-this._min);
        this._value=Math.max(this._min,Math.min(max,stepped));
    }
    setAttribute(key,value) { this.attributes[key]=value; }
    removeAttribute(key) { delete this.attributes[key]; }
}
const layerSlider = new RangeControl();
const layerNumber = {textContent:''},layerHeight = {textContent:''};
const sectionOnly = {disabled:true,checked:false},fullButton={disabled:true};
let layerState = null;
let bounds = {size:[20,2.1,14],center:[3,-6.2,8],extents:{min:[-7,-7.25,1]}};
let marks = 0,dirty=[];
const meshes = ['laps','base','bridge-exposed','bridge-full'].map(role=>({
    name:'slinky:'+role,material:{},setEnabled(value){this.enabled=value;}}));
const allMeshes = ()=>meshes;
const roleOf = mesh=>mesh.name.split(':')[1];
const colors = new WeakMap();
const boxes = Object.fromEntries(['laps','base','bridges','attachments','transparent','colors','wireframe']
    .map(key=>[key,{checked:['laps','base','bridges','colors'].includes(key),disabled:false}]));
const camera={target:new Vector(3,-6.2,8),alpha:1.4,beta:.8,radius:27};
const details={model:{getWorldBounds:()=>bounds},camera,
    scene:{clipPlane:null,clipPlane2:null,markAllMaterialsAsDirty(value){dirty.push(value);}},
    markSceneMutated(){marks++;}};
function plane(plane) {
    return plane ? {normal:[plane.normal.x,plane.normal.y,plane.normal.z],d:plane.d} : null;
}
function snapshot() {
    return {state:layerState && {...layerState},slider:{disabled:layerSlider.disabled,
        min:layerSlider.min,max:layerSlider.max,value:layerSlider.value,aria:layerSlider.attributes['aria-valuetext']},
        section:{checked:sectionOnly.checked,disabled:sectionOnly.disabled},fullDisabled:fullButton.disabled,
        number:layerNumber.textContent,height:layerHeight.textContent,
        upper:plane(details.scene.clipPlane),lower:plane(details.scene.clipPlane2),
        marks,dirty:[...dirty],camera:JSON.parse(JSON.stringify(camera))};
}
__FUNCTIONS__
__SCENARIO__
""".replace("__FUNCTIONS__", functions).replace("__SCENARIO__", scenario)
        result = subprocess.run([shutil.which("node"), "-e", harness],
                                capture_output=True, text=True, check=True)
        return json.loads(result.stdout)

    def test_build_up_and_single_layer_planes_use_print_height_in_world_coordinates(self):
        result = self.run_client(r"""
configureLayers({layer_height:.2});
const full=snapshot();
layerSlider.value=String(Number(layerSlider.value)-1);moveLayer();
const firstDown=snapshot();
layerSlider.value='3';moveLayer();
const build=snapshot(),upperPlane=details.scene.clipPlane,dirtyBefore=dirty.length;
layerSlider.value='5';moveLayer();
const reused=details.scene.clipPlane===upperPlane && dirty.length===dirtyBefore;
sectionOnly.checked=true;applyLayerPreview();
const band=snapshot();
layerSlider.value='11';moveLayer();
const last=snapshot();
showFullModel();
process.stdout.write(JSON.stringify({full,firstDown,build,reused,band,last,restored:snapshot()}));
""")
        self.assertEqual(result["full"]["state"]["count"], 11)
        self.assertEqual(result["full"]["slider"]["value"], "11")
        self.assertIsNone(result["full"]["upper"])
        self.assertIsNone(result["full"]["lower"])
        self.assertFalse(result["full"]["slider"]["disabled"])
        self.assertEqual(result["firstDown"]["state"]["index"], 10)
        self.assertEqual(result["firstDown"]["slider"]["value"], "10")
        self.assertEqual(result["firstDown"]["height"], "2.00 mm")
        self.assertAlmostEqual(result["firstDown"]["upper"]["d"], -(-7.25 + 2.0))
        self.assertTrue(result["reused"], "Dragging should update an existing plane without shader rebuilds")
        build = result["build"]
        np.testing.assert_allclose(build["upper"]["normal"], [0, 1, 0])
        self.assertAlmostEqual(build["upper"]["d"], -(-7.25 + 0.6))
        self.assertIsNone(build["lower"])
        self.assertEqual(build["number"], "3 / 11")
        self.assertEqual(build["height"], "0.60 mm")
        self.assertIn("Layer 3 of 11", build["slider"]["aria"])

        def distance(plane, print_height):
            point = [1.5, -7.25 + print_height, 8.5]
            return float(np.dot(plane["normal"], point) + plane["d"])

        # Babylon discards positive signed distances. Check the solid retained
        # below the plane independently of the client's cutoff calculation.
        self.assertLess(distance(build["upper"], 0.59), 0)
        self.assertGreater(distance(build["upper"], 0.61), 0)
        band = result["band"]
        np.testing.assert_allclose(band["lower"]["normal"], [0, -1, 0])
        for print_height, retained in ((0.79, False), (0.81, True), (0.99, True), (1.01, False)):
            visible = all(distance(band[key], print_height) <= 0 for key in ("upper", "lower"))
            self.assertEqual(visible, retained, f"Band membership at print Z={print_height}")
        last = result["last"]
        self.assertEqual(last["height"], "2.10 mm")
        self.assertAlmostEqual(-last["upper"]["d"] - last["lower"]["d"], 0.1)
        for print_height, retained in ((1.99, False), (2.05, True), (2.11, False)):
            visible = all(distance(last[key], print_height) <= 0 for key in ("upper", "lower"))
            self.assertEqual(visible, retained)
        self.assertIsNone(result["restored"]["upper"])
        self.assertIsNone(result["restored"]["lower"])
        self.assertFalse(result["restored"]["section"]["checked"])
        self.assertEqual(result["restored"]["state"]["index"], 11)
        for key in ("firstDown", "build", "band", "last", "restored"):
            self.assertEqual(result[key]["camera"], result["full"]["camera"])

    def test_rounding_new_models_missing_metadata_and_reset_are_safe(self):
        result = self.run_client(r"""
const cases=[];
for (const height of [.6000000238,.61,.05,400.00003,1.05]) {
    if (layerState) {sectionOnly.checked=true;layerSlider.value='1';moveLayer();}
    bounds={size:[20,height,14],center:[3,4,8]};
    configureLayers({layer_height:.2});cases.push(snapshot());
}
bounds={size:[20,1.05,14],center:[3,4,8]};configureLayers({});const fallback=snapshot();
layerSlider.value='-100';moveLayer();const low=snapshot();
layerSlider.value='100';moveLayer();const high=snapshot();
sectionOnly.checked=true;applyLayerPreview();resetLayers();const reset=snapshot();
bounds={size:[20,0,14],center:[3,4,8]};configureLayers({layer_height:.2});const empty=snapshot();
bounds={size:[20,2.1,14],center:[3,4,8]};configureLayers({layer_height:Number.MIN_VALUE});
const overflow=snapshot();
process.stdout.write(JSON.stringify({cases,fallback,low,high,reset,empty,overflow}));
""")
        self.assertEqual([case["state"]["count"] for case in result["cases"]], [3, 4, 1, 2000, 6])
        for case, height in zip(result["cases"], (0.6000000238, 0.61, 0.05, 400.00003, 1.05)):
            self.assertAlmostEqual(case["state"]["minY"], 4 - height / 2)
            self.assertEqual(case["state"]["index"], case["state"]["count"])
            self.assertEqual(case["slider"]["value"], str(case["state"]["count"]))
            self.assertIsNone(case["upper"])
            self.assertIsNone(case["lower"])
            self.assertFalse(case["section"]["checked"])
        self.assertEqual(result["fallback"]["state"]["step"], 0.2)
        self.assertEqual(result["low"]["state"]["index"], 1)
        self.assertEqual(result["high"]["state"]["index"], 6)
        for key in ("reset", "empty", "overflow"):
            with self.subTest(state=key):
                state = result[key]
                self.assertIsNone(state["state"])
                self.assertTrue(state["slider"]["disabled"])
                self.assertTrue(state["section"]["disabled"])
                self.assertTrue(state["fullDisabled"])
                self.assertIsNone(state["upper"])
                self.assertIsNone(state["lower"])
                self.assertNotIn("aria", state["slider"])

    def test_feature_visibility_preserves_cutoff_and_camera_and_marks_scene_for_render(self):
        result = self.run_client(r"""
configureLayers({layer_height:.2});layerSlider.value='4';moveLayer();
const before=snapshot();
boxes.laps.checked=false;boxes.attachments.checked=true;boxes.wireframe.checked=true;applyDisplay();
const after=snapshot();
process.stdout.write(JSON.stringify({before,after,enabled:meshes.map(mesh=>mesh.enabled),
    wireframe:details.scene.forceWireframe}));
""")
        self.assertEqual(result["after"]["state"], result["before"]["state"])
        self.assertEqual(result["after"]["upper"], result["before"]["upper"])
        self.assertEqual(result["after"]["camera"], result["before"]["camera"])
        self.assertEqual(result["enabled"], [False, True, False, True])
        self.assertTrue(result["wireframe"])
        self.assertGreater(result["after"]["marks"], result["before"]["marks"])


if __name__ == "__main__":
    unittest.main()
