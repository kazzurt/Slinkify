"""Embedded GLB materials for Gradio's preview, separate from the print STL.

Solid view uses Boolean surface IDs before STL precision cleanup to color exposed
bridge surfaces. See-through view also reveals full ramps and buried attachments.
"""
import json
import struct

import numpy as np


BRIDGE_COLORS = (
    ("orange", "#E56B17"), ("blue", "#167FC4"), ("magenta", "#C43788"),
    ("green", "#27915A"), ("violet", "#7455C9"), ("gold", "#BE8B00"),
)
LAP_COLOR = "#B5BAC2"


def preview_legend():
    swatches = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:5px;white-space:nowrap" '
        f'title="Gap {gap}: {name}"><span aria-hidden="true" '
        f'style="width:8px;height:8px;background:{color};display:inline-block"></span>{gap}</span>'
        for gap, (name, color) in enumerate(BRIDGE_COLORS, 1))
    return ("<div aria-label='Bridge gap colors, bottom to top; repeats every six gaps' "
            "style='display:flex;flex-wrap:wrap;align-items:center;gap:6px 12px;"
            "font-size:12px;line-height:1.5;color:var(--sl-muted,var(--body-text-color-subdued,#6b7280))'>"
            "<span>Bridge gaps, bottom to top</span>" + swatches +
            "<span>Repeat every 6 gaps</span></div>")


def face_materials(mesh, bridge_ids):
    """Material 0 = laps; 1..6 = bridge gap index modulo the palette size."""
    labels = np.zeros(len(mesh.tri_verts), dtype=np.uint8)
    for start, end, original_id in zip(mesh.run_index[:-1], mesh.run_index[1:], mesh.run_original_id):
        if original_id in bridge_ids:
            labels[start // 3:end // 3] = 1 + bridge_ids[original_id] % len(BRIDGE_COLORS)
    return labels


def linear_color(hex_color):
    srgb = np.array([int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4).tolist()


def write_bridge_preview(path, manifold, bridge_ids, see_through=False, ramps=()):
    """Write a GLB with flat normals, embedded buffers and named materials.

    Gradio 6.28's Babylon glTF loader converts right-handed coordinates by
    reflecting X. This preview precompensates so its final coordinates match the
    existing STL preview (x, print-Z, -profile-Y). It is a viewer asset, not an STL.
    """
    mesh = manifold.to_mesh()
    vertices = np.asarray(mesh.vert_properties[:, :3], dtype=np.float64)
    center = (vertices.min(0) + vertices.max(0)) / 2
    vertices = (vertices - center)[:, [0, 2, 1]] * [-1, 1, -1]
    # The preview transform is a reflection; reverse faces before computing normals.
    faces = np.asarray(mesh.tri_verts)[:, ::-1]
    labels = face_materials(mesh, bridge_ids)
    full_ramps = {}
    # Keep lower bridges visible in taller models: accumulated transparency
    # through every lap should stay similar as the number of laps increases.
    lap_count = max((lap + 2 for lap, _ in ramps), default=1)
    lap_alpha = 1 - 0.65 ** (1 / (2 * lap_count))
    if see_through:
        for lap, ramp in ramps:
            part = ramp.to_mesh()
            points = (np.asarray(part.vert_properties[:, :3], dtype=np.float64) - center)[:, [0, 2, 1]] * [-1, 1, -1]
            triangles = points[np.asarray(part.tri_verts)[:, ::-1]]
            full_ramps.setdefault(1 + lap % len(BRIDGE_COLORS), []).append(triangles)
    buffers, views, accessors, primitives, materials = [], [], [], [], []
    offset = 0

    def add_vectors(array, bounds=False):
        nonlocal offset
        array = np.asarray(array, dtype="<f4").reshape(-1, 3)
        data = array.tobytes()
        buffers.append(data)
        views.append(dict(buffer=0, byteOffset=offset, byteLength=len(data), target=34962))
        accessor = dict(bufferView=len(views) - 1, componentType=5126, count=len(array), type="VEC3")
        if bounds:
            accessor.update(min=array.min(0).tolist(), max=array.max(0).tolist())
        accessors.append(accessor)
        offset += len(data)
        return len(accessors) - 1

    for label in sorted(set(labels) | set(full_ramps)):
        if see_through and label > 0:
            if label not in full_ramps:
                continue
            triangles = np.asarray(np.concatenate(full_ramps[label]), dtype="<f4")
        else:
            triangles = np.asarray(vertices[faces[labels == label]], dtype="<f4")
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(cross, axis=1)
        valid = lengths > 0  # Avoid only faces collapsed by the preview's float32 conversion.
        triangles, cross, lengths = triangles[valid], cross[valid], lengths[valid]
        if not len(triangles):
            continue
        normals = np.repeat((cross / lengths[:, None])[:, None, :], 3, axis=1)
        position = add_vectors(triangles, bounds=True)
        normal = add_vectors(normals)
        name, color = ("Laps", LAP_COLOR) if label == 0 else BRIDGE_COLORS[label - 1]
        material = dict(name=name, pbrMetallicRoughness=dict(
            baseColorFactor=linear_color(color) + [lap_alpha if see_through and label == 0 else 1.0],
            metallicFactor=0, roughnessFactor=0.85))
        if see_through and label == 0:
            material.update(alphaMode="BLEND", doubleSided=True)
        materials.append(material)
        primitives.append(dict(attributes=dict(POSITION=position, NORMAL=normal),
                               material=len(materials) - 1, mode=4))
    if not primitives:
        raise ValueError("The preview mesh has no nonzero-area faces.")
    document = dict(asset=dict(version="2.0", generator="Slinkify bridge preview"),
                    scene=0, scenes=[dict(nodes=[0])], nodes=[dict(mesh=0, name="Staggered laps")],
                    meshes=[dict(primitives=primitives)], materials=materials,
                    buffers=[dict(byteLength=offset)], bufferViews=views, accessors=accessors)
    header = json.dumps(document, separators=(",", ":"), allow_nan=False).encode("utf-8")
    header += b" " * (-len(header) % 4)
    binary = b"".join(buffers)
    with open(path, "wb") as target:
        target.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(header) + 8 + len(binary)))
        target.write(struct.pack("<I4s", len(header), b"JSON"))
        target.write(header)
        target.write(struct.pack("<I4s", len(binary), b"BIN\x00"))
        target.write(binary)
