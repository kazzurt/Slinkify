"""A single local GLB scene with feature visibility and print-layer inspection.

The printable mesh is never changed. Exposed bridge surfaces and full ramps are
separate preview groups; the viewer enables one representation at a time.
"""
from functools import lru_cache
import json
from pathlib import Path
import re
import struct
from urllib.parse import quote

import gradio as gr
import numpy as np

from preview_colors import BRIDGE_COLORS, LAP_COLOR, face_materials, linear_color


BASE_COLOR = "#87909E"


def write_live_preview(path, manifold, bridge_ids, ramps=(), base=0.0):
    """Write independently visible laps, base, exposed bridges, and full ramps.

    Coordinates use the standard right-handed glTF orientation [X, print-Z,
    -profile-Y]. Babylon converts the GLB into its left-handed scene itself;
    pre-reflecting X here would mirror the lettering in the default camera.
    Groups are named ``slinky:<role>[:<palette-name>]`` with flat, unit normals.
    """
    original = manifold.to_mesh()
    vertices = np.asarray(original.vert_properties[:, :3], dtype=np.float64)
    if not len(vertices):
        raise ValueError("The preview mesh is empty.")
    center = (vertices.min(0) + vertices.max(0)) / 2
    ramps = tuple(ramps)
    lap_count = max((int(lap) + 2 for lap, _ in ramps), default=1)
    lap_alpha = 1 - 0.65 ** (1 / (2 * lap_count))
    buffers, views, accessors, meshes, nodes, materials = [], [], [], [], [], []
    offset = 0

    def triangles_of(part):
        mesh = part.to_mesh()
        points = (np.asarray(mesh.vert_properties[:, :3], dtype=np.float64) - center)[:, [0, 2, 1]]
        points *= [1, 1, -1]
        return mesh, points[np.asarray(mesh.tri_verts)]

    def vectors(array, bounds=False):
        nonlocal offset
        array = np.asarray(array, dtype="<f4").reshape(-1, 3)
        data = array.tobytes()
        buffers.append(data)
        views.append(dict(buffer=0, byteOffset=offset, byteLength=len(data), target=34962))
        accessor = dict(bufferView=len(views) - 1, componentType=5126,
                        count=len(array), type="VEC3")
        if bounds:
            accessor.update(min=array.min(0).tolist(), max=array.max(0).tolist())
        accessors.append(accessor)
        offset += len(data)
        return len(accessors) - 1

    def add_group(role, triangles, color, suffix=""):
        if not len(triangles):
            return
        triangles = np.asarray(triangles, dtype="<f4")
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 0
        triangles, normals, lengths = triangles[valid], normals[valid], lengths[valid]
        if not len(triangles):
            return
        normals = np.repeat((normals / lengths[:, None])[:, None, :], 3, axis=1)
        position, normal = vectors(triangles, True), vectors(normals)
        name = f"slinky:{role}" + (f":{suffix}" if suffix else "")
        material = dict(name=name, doubleSided=True, pbrMetallicRoughness=dict(
            baseColorFactor=linear_color(color) + [1.0], metallicFactor=0, roughnessFactor=0.85),
            extras=dict(slinky_role=role, preview_alpha=lap_alpha if role == "laps" else 1.0))
        materials.append(material)
        primitive = dict(attributes=dict(POSITION=position, NORMAL=normal),
                         material=len(materials) - 1, mode=4)
        meshes.append(dict(name=name, primitives=[primitive]))
        nodes.append(dict(name=name, mesh=len(meshes) - 1,
                          extras=dict(slinky_role=role, preview_alpha=lap_alpha if role == "laps" else 1.0)))

    body = manifold
    if base > 0:
        lower = manifold.trim_by_plane((0, 0, -1), -float(base))
        body = manifold.trim_by_plane((0, 0, 1), float(base))
        if not lower.is_empty():
            _, triangles = triangles_of(lower)
            add_group("base", triangles, BASE_COLOR)
    mesh, triangles = triangles_of(body)
    labels = face_materials(mesh, bridge_ids)
    add_group("laps", triangles[labels == 0], LAP_COLOR)
    for label in sorted(set(labels) - {0}):
        name, color = BRIDGE_COLORS[int(label) - 1]
        add_group("bridge-exposed", triangles[labels == label], color, name)
    full_ramps = {}
    for lap, ramp in ramps:
        _, triangles = triangles_of(ramp)
        full_ramps.setdefault(int(lap) % len(BRIDGE_COLORS), []).append(triangles)
    for label, parts in sorted(full_ramps.items()):
        name, color = BRIDGE_COLORS[label]
        add_group("bridge-full", np.concatenate(parts), color, name)
    if not meshes:
        raise ValueError("The preview mesh has no nonzero-area faces.")
    document = dict(asset=dict(version="2.0", generator="Slinkify live feature preview"),
                    scene=0, scenes=[dict(nodes=list(range(len(nodes))))], nodes=nodes,
                    meshes=meshes, materials=materials, buffers=[dict(byteLength=offset)],
                    bufferViews=views, accessors=accessors,
                    extras=dict(lap_count=lap_count, lap_alpha=lap_alpha))
    header = json.dumps(document, separators=(",", ":"), allow_nan=False).encode("utf-8")
    header += b" " * (-len(header) % 4)
    binary = b"".join(buffers)
    with open(path, "wb") as target:
        target.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(header) + 8 + len(binary)))
        target.write(struct.pack("<I4s", len(header), b"JSON"))
        target.write(header)
        target.write(struct.pack("<I4s", len(binary), b"BIN\x00"))
        target.write(binary)


@lru_cache(maxsize=1)
def viewer_asset_name():
    """Find the viewer entry point used by the installed Gradio Model3D."""
    assets = Path(gr.__file__).parent / "templates" / "frontend" / "assets"
    for source in sorted(assets.glob("Canvas3D-*.js")):
        content = source.read_text(encoding="utf-8")
        match = re.search(r"import\([`'\"]\./(lib-[^`'\"]+\.js)[`'\"]\)", content)
        if match and (assets / match[1]).is_file():
            return match[1]
    raise RuntimeError("The installed Gradio bundle has no local 3D viewer module.")


def preview_payload(path, method, layer_height=0.2):
    """Return a quote-safe HTML value; only a new URL causes a mesh load."""
    if not path:
        return json.dumps(dict(url=None, method=method), separators=(",", ":"))
    local_path = Path(path).resolve().as_posix()
    return json.dumps(dict(url="/gradio_api/file=" + quote(local_path, safe="/:"),
                           method=method, layer_height=layer_height),
                      separators=(",", ":"), ensure_ascii=False, allow_nan=False)


_TEMPLATE = """
<div class="slinky-live-viewer">
  <div class="slinky-preview-controls" aria-label="3D preview visibility">
    <div class="slinky-preview-group" role="group" aria-label="Visible features">
      <span class="slinky-preview-group-name">Visible</span>
      <div class="slinky-preview-options">
        <label title="Show or hide lap surfaces in the preview."><input type="checkbox" data-feature="laps" checked disabled> Laps</label>
        <label title="Show or hide bridge connections in the preview."><input type="checkbox" data-feature="bridges" checked disabled> Bridges</label>
        <label title="Show or hide the flat base, when one is included."><input type="checkbox" data-feature="base" checked disabled> Base</label>
      </div>
    </div>
    <div class="slinky-preview-group" role="group" aria-label="Preview display">
      <span class="slinky-preview-group-name">Display</span>
      <div class="slinky-preview-options">
        <label title="Show complete ramp volumes, including attachments buried in the laps. Off shows only exposed bridge surfaces. Preview only."><input type="checkbox" data-feature="attachments" disabled> Full ramps</label>
        <label title="Make the laps transparent to inspect the connections. Preview only."><input type="checkbox" data-feature="transparent" disabled> Transparent laps</label>
        <label title="Color connections by gap, from bottom to top. Preview only."><input type="checkbox" data-feature="colors" checked disabled> Bridge colors</label>
        <label title="Show triangle edges on visible features. Preview only."><input type="checkbox" data-feature="wireframe" disabled> Wireframe</label>
      </div>
    </div>
  </div>
  <div class="slinky-preview-stage">
    <canvas class="slinky-preview-canvas" aria-label="Interactive 3D slinky preview"></canvas>
    <div class="slinky-layer-panel" role="group" aria-label="Layer inspection">
      <span class="slinky-layer-label">Layer</span>
      <output class="slinky-layer-number" aria-live="off">—</output>
      <span class="slinky-layer-height">—</span>
      <input class="slinky-layer-slider" type="range" min="1" max="1" step="1" value="1"
             aria-label="Preview layer" aria-orientation="vertical"
             title="Drag down to remove higher layers. Arrow keys move one print layer; Home goes to the first layer and End shows the top." disabled>
      <label class="slinky-layer-only-label" title="Isolate the selected print layer.">
        <input class="slinky-layer-only" type="checkbox" disabled> Layer only
      </label>
      <button class="slinky-layer-full" type="button" title="Restore all layers." disabled>Full</button>
    </div>
  </div>
  <p class="slinky-preview-status" role="status">Generate a model to preview its features.</p>
  <p class="slinky-preview-hint">Drag to rotate · Scroll to zoom · Right-drag to pan</p>
</div>
"""

_CSS = """
.slinky-live-viewer { border: 1px solid var(--sl-border, var(--border-color-primary, #ddd)); border-radius: 10px; overflow: hidden; }
.slinky-preview-controls { display: grid; gap: 2px; padding: 8px 12px; background: var(--sl-surface, var(--background-fill-secondary, #fafaf9)); border-bottom: 1px solid var(--sl-border, var(--border-color-primary, #ddd)); }
.slinky-preview-group { display: grid; grid-template-columns: 52px minmax(0, 1fr); align-items: start; gap: 12px; }
.slinky-preview-group-name { font-size: 12px; font-weight: 600; line-height: 36px; color: var(--sl-muted, var(--body-text-color-subdued, #6b7280)); }
.slinky-preview-options { display: flex; flex-wrap: wrap; gap: 0 14px; min-width: 0; }
.slinky-preview-controls label { display: inline-flex; align-items: center; gap: 7px; min-height: 36px; margin: 0; font-size: 13px; line-height: 1.35; cursor: pointer; white-space: nowrap; }
.slinky-preview-controls label:has(input:disabled) { color: var(--sl-muted, var(--body-text-color-subdued, #6b7280)); cursor: default; }
.slinky-preview-controls input { accent-color: var(--sl-accent, var(--primary-500, #b45309)); width: 15px; height: 15px; margin: 0; flex: 0 0 auto; }
.slinky-preview-controls input:focus-visible { outline: 2px solid var(--sl-accent, #b45309); outline-offset: 3px; }
.slinky-preview-canvas { display: block; width: 100%; height: 460px; background: #f7f7f7; touch-action: none; }
.slinky-preview-canvas:focus-visible { outline: 2px solid var(--sl-accent, #b45309); outline-offset: -2px; }
.slinky-preview-stage { display: grid; grid-template-columns: minmax(0, 1fr) 84px; }
.slinky-layer-panel { display: flex; flex-direction: column; align-items: center; gap: 5px; padding: 12px 8px; min-width: 0; background: var(--sl-surface, var(--background-fill-secondary, #fafaf9)); border-left: 1px solid var(--sl-border, #ddd); }
.slinky-layer-label { font-size: 12px; font-weight: 600; }
.slinky-layer-number { font-size: 11px; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; text-align: center; }
.slinky-layer-height { font-size: 11px; color: var(--sl-muted, #6b7280); font-variant-numeric: tabular-nums; }
.slinky-layer-slider { writing-mode: vertical-lr; direction: rtl; appearance: auto; align-self: stretch; flex: 1; min-height: 180px; width: 44px; margin: 6px auto; accent-color: var(--sl-accent, #b45309); cursor: ns-resize; touch-action: none; }
.slinky-layer-only-label { display: flex; flex-direction: column; align-items: center; gap: 4px; font-size: 11px; line-height: 1.4; white-space: nowrap; cursor: pointer; }
.slinky-layer-only { accent-color: var(--sl-accent, #b45309); width: 15px; height: 15px; }
.slinky-layer-full { min-height: 36px; width: 100%; border: 1px solid var(--sl-border, #ddd); border-radius: 6px; background: transparent; color: inherit; font-size: 12px; cursor: pointer; }
.slinky-layer-panel :disabled { opacity: .5; cursor: default; }
.slinky-layer-panel :is(input, button):focus-visible { outline: 2px solid var(--sl-accent, #b45309); outline-offset: 3px; }
.slinky-preview-status, .slinky-preview-hint { margin: 0; padding: 6px 12px; font-size: 12px; line-height: 1.5; }
.slinky-preview-hint { color: var(--sl-muted, var(--body-text-color-subdued, #6b7280)); padding-bottom: 10px; }
.slinky-preview-status:empty { display: none; }
"""

_JS = r"""
const canvas = element.querySelector('canvas');
const controls = element.querySelector('.slinky-preview-controls');
const status = element.querySelector('.slinky-preview-status');
const layerSlider = element.querySelector('.slinky-layer-slider');
const layerNumber = element.querySelector('.slinky-layer-number');
const layerHeight = element.querySelector('.slinky-layer-height');
const sectionOnly = element.querySelector('.slinky-layer-only');
const fullButton = element.querySelector('.slinky-layer-full');
const boxes = Object.fromEntries([...controls.querySelectorAll('input')].map(x => [x.dataset.feature, x]));
let details, viewer, ready, disposed = false, requestNumber = 0, currentUrl = null;
let previewReady = false;
let layerState = null;
const colors = new WeakMap();
const roleOf = mesh => mesh.name.startsWith('slinky:') ? mesh.name.split(':')[1] : null;
const allMeshes = () => (details?.model?.assetContainer.meshes ?? []).filter(x => x.getTotalVertices() > 0);

function saveButton() {
    const root = document.getElementById('slinky-save-preview');
    return root?.tagName === 'BUTTON' ? root : root?.querySelector('button');
}

function enforceSaveReadiness() {
    const button = saveButton();
    // Gradio can enable its button output before the asynchronous scene loads.
    // Reapply only the closed gate here, respecting ordinary server button state.
    if (!previewReady && button && !button.disabled) button.disabled = true;
}

function setSaveReadiness(isReady) {
    previewReady = isReady;
    const button = saveButton();
    if (button) button.disabled = !isReady;
}

function blockUnreadySave(event) {
    if (!previewReady && event.target?.closest?.('#slinky-save-preview')) {
        event.preventDefault();
        event.stopImmediatePropagation();
        enforceSaveReadiness();
    }
}

function available() {
    const roles = new Set(allMeshes().map(roleOf));
    boxes.laps.disabled = !roles.has('laps');
    boxes.base.disabled = !roles.has('base');
    boxes.bridges.disabled = !(roles.has('bridge-exposed') || roles.has('bridge-full'));
    boxes.attachments.disabled = !roles.has('bridge-full');
    boxes.transparent.disabled = boxes.laps.disabled;
    boxes.colors.disabled = boxes.bridges.disabled;
    boxes.wireframe.disabled = !allMeshes().length;
}

function applyDisplay() {
    for (const mesh of allMeshes()) {
        const role = roleOf(mesh);
        const full = boxes.attachments.checked && !boxes.attachments.disabled;
        const visible = role === 'laps' ? boxes.laps.checked :
            role === 'base' ? boxes.base.checked :
            role === 'bridge-full' ? boxes.bridges.checked && full :
            role === 'bridge-exposed' ? boxes.bridges.checked && !full : true;
        mesh.setEnabled(visible);
        const material = mesh.material;
        if (!material) continue;
        if (material.albedoColor && !colors.has(material)) colors.set(material, material.albedoColor.clone());
        if (material.albedoColor && role?.startsWith('bridge-')) {
            if (boxes.colors.checked) material.albedoColor.copyFrom(colors.get(material));
            else material.albedoColor.copyFromFloats(__NEUTRAL_COLOR__);
        }
        const transparent = role === 'laps' && boxes.transparent.checked;
        if (transparent) {
            const alpha = material.metadata?.gltf?.extras?.preview_alpha ?? mesh.metadata?.gltf?.extras?.preview_alpha;
            material.alpha = typeof alpha === 'number' ? alpha : 0.035;
            material.transparencyMode = 2;
        } else {
            material.alpha = 1;
            material.transparencyMode = 0;
        }
    }
    if (details) {
        details.scene.forceWireframe = boxes.wireframe.checked;
        details.markSceneMutated();
    }
}

async function ensureViewer() {
    if (!ready) ready = (async () => {
        const moduleUrl = new URL('assets/__VIEWER_MODULE__', document.baseURI).href;
        const { CreateViewerForCanvas } = await import(moduleUrl);
        if (disposed) return;
        const created = await CreateViewerForCanvas(canvas, {
            engine: 'WebGL', clearColor: [.97, .97, .97, 1],
            useRightHandedSystem: false, cameraAutoOrbit: { enabled: false },
            postProcessing: { ssao: 'disabled' },
            onInitialized: d => { details = d; }
        });
        if (disposed) { created.dispose(); return; }
        viewer = created;
        viewer.onModelChanged.add(() => { available(); applyDisplay(); });
        details.camera.lowerRadiusLimit = .1;
        details.camera.onAfterCheckInputsObservable.add(() => {
            details.camera.wheelPrecision = 250 / Math.max(details.camera.radius, .1);
            details.camera.panningSensibility = 10000 / Math.max(details.camera.radius, .1);
        });
    })();
    await ready;
}

function setLayerPlanes(upperY = null, lowerY = null) {
    if (!details) return;
    const scene = details.scene;
    let definesChanged = false;
    for (const [key, height, direction] of [['clipPlane', upperY, 1], ['clipPlane2', lowerY, -1]]) {
        if (height === null) {
            definesChanged ||= !!scene[key];
            scene[key] = null;
        } else if (scene[key]) {
            scene[key].d = -direction * height;
        } else {
            scene[key] = { normal: details.camera.target.clone().copyFromFloats(0, direction, 0),
                           d: -direction * height };
            definesChanged = true;
        }
    }
    // Babylon's PBR shaders need new defines when a clipping plane is added or
    // removed. Moving an existing plane just updates its world-space uniform.
    if (definesChanged) scene.markAllMaterialsAsDirty(16);
    details.markSceneMutated();
}

function resetLayers() {
    layerState = null;
    layerSlider.disabled = sectionOnly.disabled = fullButton.disabled = true;
    layerSlider.min = layerSlider.max = layerSlider.value = '1';
    layerSlider.removeAttribute('aria-valuetext');
    sectionOnly.checked = false;
    layerNumber.textContent = layerHeight.textContent = '—';
    setLayerPlanes();
}

function configureLayers(payload) {
    const bounds = details?.model?.getWorldBounds();
    const height = bounds?.size?.[1];
    const minY = bounds?.extents?.min?.[1] ?? bounds?.center?.[1] - height / 2;
    if (!(height > 0) || !Number.isFinite(height) || !Number.isFinite(minY)) {
        resetLayers();
        return;
    }
    const requestedStep = Number(payload.layer_height);
    const step = Number.isFinite(requestedStep) && requestedStep > 0 ? requestedStep : .2;
    // GLB bounds use float32 coordinates; an exact 0.6 mm solid should have
    // three 0.2 mm layers rather than a phantom fourth from rounding noise.
    const layers = height / step;
    const count = Math.max(1, Math.ceil(layers - Math.max(1, layers) * 1e-6));
    if (!Number.isSafeInteger(count)) { resetLayers(); return; }
    layerState = {minY, height, step, count, index: count};
    layerSlider.min = '1';
    // Set max before value: native range inputs clamp against their current
    // maximum immediately, which is still 1 after resetting the previous model.
    layerSlider.max = String(count);
    layerSlider.value = String(count);
    layerSlider.disabled = sectionOnly.disabled = fullButton.disabled = false;
    sectionOnly.checked = false;
    applyLayerPreview();
}

function applyLayerPreview() {
    if (!layerState) return;
    const {minY, height, step, count, index} = layerState;
    const top = Math.min(index * step, height);
    const lower = Math.min((index - 1) * step, top);
    const full = index === count && !sectionOnly.checked;
    setLayerPlanes(full ? null : minY + top, sectionOnly.checked ? minY + lower : null);
    const digits = step < .01 ? 4 : step < .1 ? 3 : 2;
    layerNumber.textContent = `${index} / ${count}`;
    layerHeight.textContent = `${top.toFixed(digits)} mm`;
    layerSlider.setAttribute('aria-valuetext', `Layer ${index} of ${count}, ${top.toFixed(digits)} millimeters${sectionOnly.checked ? ', layer only' : ''}`);
}

function moveLayer() {
    if (!layerState) return;
    const value = Number(layerSlider.value);
    if (!Number.isFinite(value)) return;
    layerState.index = Math.max(1, Math.min(layerState.count, Math.round(value)));
    layerSlider.value = String(layerState.index);
    applyLayerPreview();
}

function showFullModel() {
    if (!layerState) return;
    sectionOnly.checked = false;
    layerSlider.value = String(layerState.count);
    moveLayer();
}

function fitNewModel() {
    const bounds = details?.model?.getWorldBounds();
    if (!bounds || !bounds.size.every(Number.isFinite)) return;
    const camera = details.camera;
    // Fit the bounding sphere through the narrower field of view. The bundled
    // viewer's default radius uses the model diagonal alone, which clips wide
    // profiles in a portrait-shaped preview. The sphere also fits after rotation.
    const rectangle = canvas.getBoundingClientRect();
    const aspect = Math.max(rectangle.width, 1) / Math.max(rectangle.height, 1);
    const halfFov = camera.fov / 2;
    const vertical = camera.fovMode === 1 ? Math.atan(Math.tan(halfFov) / aspect) : halfFov;
    const horizontal = camera.fovMode === 1 ? halfFov : Math.atan(Math.tan(halfFov) * aspect);
    const sphereRadius = Math.hypot(...bounds.size) / 2;
    const radius = Math.max(.1, sphereRadius / Math.sin(Math.min(vertical, horizontal)) * 1.08);
    camera.stopInterpolation();
    camera.setTarget(camera.target.clone().copyFromFloats(...bounds.center));
    camera.radius = radius;
    camera.upperRadiusLimit = Math.max(camera.upperRadiusLimit || 0, radius * 5);
    details.markSceneMutated();
}

async function loadValue() {
    const request = ++requestNumber;
    setSaveReadiness(false);
    try {
        const payload = typeof props.value === 'string' ? JSON.parse(props.value || '{}') : (props.value || {});
        if (!payload.url) {
            currentUrl = null;
            resetLayers();
            if (viewer) await viewer.resetModel();
            available();
            status.textContent = 'Generate a model to preview its features.';
            return;
        }
        if (payload.url === currentUrl) { setSaveReadiness(true); return; }
        resetLayers();
        status.textContent = 'Loading 3D preview…';
        await ensureViewer();
        if (disposed || request !== requestNumber || !viewer) return;
        await viewer.loadModel(payload.url);
        if (disposed || request !== requestNumber) return;
        currentUrl = payload.url;
        // glTF's LH conversion reflects X; the LH camera at +Z reflects screen
        // X as well. Together they keep the original profile upright and readable.
        details.camera.alpha = Math.PI / 2;
        details.camera.beta = 50 * Math.PI / 180;
        fitNewModel();
        configureLayers(payload);
        available();
        applyDisplay();
        status.textContent = '';
        // Allow the new scene to reach the canvas before enabling its Save button.
        await new Promise(resolve => requestAnimationFrame(resolve));
        if (!disposed && request === requestNumber) setSaveReadiness(true);
    } catch (error) {
        if (!disposed && request === requestNumber) {
            status.textContent = 'Could not load the 3D preview: ' + (error?.message || String(error));
            console.error('Slinky preview:', error);
        }
    }
}

controls.addEventListener('change', applyDisplay);
layerSlider.addEventListener('input', moveLayer);
sectionOnly.addEventListener('change', applyLayerPreview);
fullButton.addEventListener('click', showFullModel);
document.addEventListener('click', blockUnreadySave, true);
watch('value', loadValue);
loadValue();
const cleanup = new MutationObserver(() => {
    if (!element.isConnected) {
        disposed = true;
        requestNumber++;
        cleanup.disconnect();
        controls.removeEventListener('change', applyDisplay);
        layerSlider.removeEventListener('input', moveLayer);
        sectionOnly.removeEventListener('change', applyLayerPreview);
        fullButton.removeEventListener('click', showFullModel);
        document.removeEventListener('click', blockUnreadySave, true);
        viewer?.dispose();
    } else {
        enforceSaveReadiness();
    }
});
cleanup.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['disabled'] });
"""


def preview_component():
    """Create a persistent viewer with browser-only visibility and layer controls."""
    javascript = _JS.replace("__VIEWER_MODULE__", viewer_asset_name()).replace(
        "__NEUTRAL_COLOR__", ",".join(str(x) for x in linear_color(LAP_COLOR)))
    return gr.HTML(value=preview_payload(None, "helix"), html_template=_TEMPLATE,
                   css_template=_CSS, js_on_load=javascript, apply_default_css=False,
                   elem_id="slinky-live-preview", label="3D preview", show_label=False,
                   container=False)
