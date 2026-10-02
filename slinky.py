#!/usr/bin/env python3
"""
slinky.py -- sweep a constant 2D profile along a helix -> print-in-place "slinky" STL.

PROFILE SOURCES (pick one)
    --step FILE        a STEP of a straight extrusion (needs cadquery: pip install cadquery).
                       The extrusion axis is detected, the base face outline (with holes) is taken.
    --loops FILE.json  loops saved by --save-loops (no cadquery needed):
                       [{"outer": [[x,y],...], "holes": [[[x,y],...], ...]}]
    --profile "u,v; u,v; ..."   one simple polygon typed in
    (none)             a plain rectangle --width x --thickness

PROFILE -> COIL MAPPING   (profile x -> radial u, profile y -> axial v, both scaled by --scale)
    u : radial offset from the helix mean radius R   (+ = outward)     [--flip-radial reverses]
    v : axial offset, profile bottom = coil bottom   (+ = up)           [--mirror flips]
The profile plane always contains the helix axis (Onshape sweep "Lock profile direction" = axis),
so a flat strip keeps its flat faces perpendicular to the axis -> washer-like coils.
For a right-handed coil the top end face reads left-to-right when +x points INWARD, which is the
default when --step/--loops are used (text is mirrored on the other end, like any extruded logo).

GEOMETRY
    pitch = profile height + gap          (coils rest `gap` apart -> print-in-place clearance)
    Bottom end: the sweep starts one profile-height below the bed and is trimmed at z=0, so the
    first turn is a wedge lying flat on the bed (fully supported).  Top end: trimmed the same way
    unless --no-top-trim, which leaves the last turn's end face intact (shows the profile).
    Resting height H = turns * pitch + profile height (with both trims).

PRINTING (axis vertical, no supports)
    Every layer is an arc of the coil.  The leading  360*layer/pitch  degrees of each layer's arc
    are bridged over `gap` of air onto the coil below -- that is the print-in-place trick.
    Keep gap = an integer number of layer heights (2 layers is the usual starting point) so the
    slicer produces the same clearance all the way round.

USAGE
    python slinky.py                                        # 60 mm coil, 5 x 1 mm strip, 0.4 gap, 40 turns
    python slinky.py --turns 5 --out coupon.stl             # short test coupon
    python slinky.py --profile "-2.5,0; 2.5,0; 2.5,1.2; -2.5,1.2"
    python slinky.py --step DDR.step --scale 0.2 --od 65 --turns 10 --no-top-trim --png --save-loops DDR_profile.json
    python slinky.py --loops DDR_profile.json --scale 0.2 --od 65 --turns 10 --no-top-trim

Requires: numpy, manifold3d   (pip install numpy manifold3d);  matplotlib for --png; cadquery for --step.
"""
import argparse
from collections import deque
import json
import math
import struct
import sys

import numpy as np
import manifold3d as m3d

TAU = 2.0 * math.pi
MAX_LAYER_SAMPLES = 10000
MATERIALS = {  # shear modulus [MPa], density [g/cm^3]  (rough, for the spring estimate only)
    "PLA": (1300.0, 1.24),
    "PETG": (750.0, 1.27),
    "ABS": (800.0, 1.04),
    "NYLON": (600.0, 1.14),
}


# ----------------------------------------------------------------------------- 2D helpers
def parse_profile(text):
    pts = []
    for tok in text.replace("\n", ";").split(";"):
        tok = tok.strip()
        if tok:
            u, v = tok.split(",")
            pts.append((float(u), float(v)))
    if len(pts) < 3:
        raise ValueError("profile needs at least 3 points")
    return np.array(pts, float)


def signed_area(P):
    # Local coordinates avoid cancellation for DXFs far from the drawing origin.
    P = np.asarray(P, float)
    if len(P) < 3:
        return 0.0
    P = P - P[0]
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def centroid(P):
    x, y = P[:, 0], P[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    c = x * yn - xn * y
    A = 0.5 * c.sum()
    return float(((x + xn) * c).sum() / (6 * A)), float(((y + yn) * c).sum() / (6 * A))


def rdp(P, tol):
    """Ramer-Douglas-Peucker on a closed loop (keeps the two most distant points as anchors)."""
    P = np.asarray(P, float)
    if len(P) < 4 or tol <= 0:
        return P
    i0 = 0
    i1 = int(np.argmax(np.linalg.norm(P - P[0], axis=1)))

    def simplify(pts):
        if len(pts) < 3:
            return pts
        a, b = pts[0], pts[-1]
        d = b - a
        L = np.linalg.norm(d)
        if L < 1e-12:
            dist = np.linalg.norm(pts - a, axis=1)
        else:
            dist = np.abs((pts[:, 0] - a[0]) * d[1] - (pts[:, 1] - a[1]) * d[0]) / L
        k = int(np.argmax(dist))
        if dist[k] > tol:
            return np.vstack([simplify(pts[:k + 1])[:-1], simplify(pts[k:])])
        return np.array([a, b])

    seg1 = simplify(P[i0:i1 + 1])
    seg2 = simplify(np.vstack([P[i1:], P[:1]]))
    return np.vstack([seg1[:-1], seg2[:-1]])


def dedupe(P, eps=1e-6):
    P = np.asarray(P, float)
    if len(P) == 0:
        return P.copy()
    keep = [0]
    for i in range(1, len(P)):
        if np.linalg.norm(P[i] - P[keep[-1]]) > eps:
            keep.append(i)
    if len(keep) > 1 and np.linalg.norm(P[keep[-1]] - P[keep[0]]) <= eps:
        keep.pop()
    return P[keep]


def validate_regions(regions):
    """Copy, validate and orient profile loops; reject invalid data before meshing."""
    if not isinstance(regions, (list, tuple)) or not regions:
        raise ValueError("profile must contain at least one region")

    def ring(points, name, clockwise=False):
        try:
            points = np.asarray(points, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}: expected numeric [x, y] points") from exc
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"{name}: expected an array of [x, y] points")
        if not np.isfinite(points).all():
            raise ValueError(f"{name}: coordinates must be finite")
        points = dedupe(points)
        if len(points) < 3:
            raise ValueError(f"{name}: needs at least three distinct points")
        area = signed_area(points)
        if abs(area) <= 1e-12:
            raise ValueError(f"{name}: has zero area; check for collinear or crossing edges")
        if (area < 0) != clockwise:
            points = points[::-1]
        return points.copy()

    result = []
    for i, region in enumerate(regions, 1):
        if not isinstance(region, dict) or "outer" not in region:
            raise ValueError(f"region {i}: missing outer boundary")
        holes = region.get("holes", [])
        if not isinstance(holes, (list, tuple)):
            raise ValueError(f"region {i}: holes must be a list of loops")
        result.append({
            "outer": ring(region["outer"], f"region {i} outer boundary"),
            "holes": [ring(hole, f"region {i} hole {j}", clockwise=True)
                      for j, hole in enumerate(holes, 1)],
        })
    return result


# ----------------------------------------------------------------------------- profile sources
def _plane_basis(normal):
    """An orthonormal in-plane frame, retaining the axis-aligned import orientation."""
    normal = np.asarray(normal, float)
    normal = normal / np.linalg.norm(normal)
    axis = int(np.argmax(np.abs(normal)))
    ix, iy = [i for i in range(3) if i != axis]
    ex, ey = np.eye(3)[ix], np.eye(3)[iy]
    if np.dot(np.cross(ex, ey), normal) < 0:
        ex, ey = ey, ex
    ex = ex - np.dot(ex, normal) * normal
    ex /= np.linalg.norm(ex)
    return ex, np.cross(normal, ex)


def loops_from_step(path, sample=0.25):
    """Outline (+holes) of a straight extrusion in a STEP file, in the extrusion's own base plane."""
    if not math.isfinite(sample) or sample <= 0:
        raise ValueError("STEP sampling distance must be finite and positive")
    import cadquery as cq
    from OCP.BRepTools import BRepTools_WireExplorer
    from OCP.TopAbs import TopAbs_REVERSED
    solids = cq.importers.importStep(path).solids().vals()
    if len(solids) != 1:
        raise ValueError(f"{path}: expected one solid, found {len(solids)}")
    s = solids[0]
    planes = [f for f in s.Faces() if f.geomType() == "PLANE"]
    volume = s.Volume()
    candidates = []
    for candidate in sorted(planes, key=lambda face: face.Area(), reverse=True):
        n = candidate.normalAt()
        for twin in planes:
            if twin is candidate or abs(abs(twin.normalAt().dot(n)) - 1) >= 1e-6:
                continue
            if abs(twin.Area() - candidate.Area()) >= 1e-4 * candidate.Area():
                continue
            depth = abs((twin.Center() - candidate.Center()).dot(n))
            if depth > 1e-9 and abs(volume - candidate.Area() * depth) <= 1e-3 * volume:
                candidates.append((candidate, n, depth))
                break
        if candidates:
            break
    if not candidates:
        raise ValueError("could not identify two parallel end faces of a straight extrusion")
    base, n, depth = candidates[0]
    # 2D frame in the base plane: x along the base's longest bbox direction if axis-aligned, else the
    # world axes projected.  Onshape extrudes are axis aligned, so this picks X/Y (or whatever is in-plane).
    nv = np.array(n.toTuple())
    ax = int(np.argmax(np.abs(nv)))
    ex, ey = _plane_basis(nv)

    def disc(w):
        pts = []
        exp = BRepTools_WireExplorer(w.wrapped)
        while exp.More():
            e = cq.Edge(exp.Current())
            rev = exp.Current().Orientation() == TopAbs_REVERSED
            k = 2 if e.geomType() == "LINE" else max(3, int(math.ceil(e.Length() / sample)) + 1)
            ts = np.linspace(0, 1, k)[:-1]
            if rev:
                ts = 1 - ts
            for t in ts:
                p = np.array(e.positionAt(float(t)).toTuple())
                pts.append((float(p @ ex), float(p @ ey)))
            exp.Next()
        return dedupe(np.array(pts))

    ow = base.outerWire()
    holes = [disc(w) for w in base.Wires() if not w.isSame(ow)]
    print(f"{path}: extrusion depth {depth:.3f} mm along {'XYZ'[ax]}, base area {base.Area():.2f} mm^2, "
          f"{len(holes)} hole(s)")
    return [{"outer": disc(ow), "holes": holes}]


def load_loops(args):
    """Returns a list of regions, each {'outer': (n,2) CCW, 'holes': [(m,2) CW, ...]} in raw profile mm."""
    if args.step:
        regions = loops_from_step(args.step)
    elif args.loops:
        with open(args.loops, encoding="utf-8-sig") as f:
            regions = json.load(f)
        if isinstance(regions, dict) and "regions" in regions:
            regions = regions["regions"]
    elif args.profile:
        regions = [{"outer": parse_profile(args.profile), "holes": []}]
    else:
        w, t = args.width, args.thickness
        regions = [{"outer": np.array([[-w / 2, 0.0], [w / 2, 0.0], [w / 2, t], [-w / 2, t]]), "holes": []}]
    return validate_regions(regions)


def prepare_profile(regions, args):
    """Scale, orient and simplify the raw loops into (u, v) coil coordinates."""
    allpts = np.vstack([r["outer"] for r in regions])
    xmin, xmax = allpts[:, 0].min(), allpts[:, 0].max()
    ymin, ymax = allpts[:, 1].min(), allpts[:, 1].max()
    s = args.scale
    xc = 0.5 * (xmin + xmax)
    inward_default = bool(args.step or args.loops)   # logos read correctly on the top end face with +x inward
    flip = -1.0 if (inward_default != bool(args.flip_radial)) else 1.0
    mirror = -1.0 if args.mirror else 1.0
    yref = ymax if args.mirror else ymin

    def xform(P):
        Q = np.column_stack([flip * (P[:, 0] - xc) * s, mirror * (P[:, 1] - yref) * s])
        Q = rdp(Q, args.tol)
        Q = dedupe(Q, 1e-6)
        return Q

    prof = []
    dropped = 0
    for r in regions:
        o = xform(r["outer"])
        if signed_area(o) < 0:
            o = o[::-1].copy()
        hs = []
        for h in r["holes"]:
            hq = xform(h)
            if abs(signed_area(hq)) < args.min_hole_area or len(hq) < 3:
                dropped += 1
                continue
            if signed_area(hq) > 0:
                hq = hq[::-1].copy()
            hs.append(hq)
        prof.append({"outer": o, "holes": hs})
    if dropped:
        print(f"dropped {dropped} hole(s) smaller than {args.min_hole_area} mm^2 after scaling")
    return prof


# ----------------------------------------------------------------------------- mesh building
def sweep_region(reg, R, pitch, th0, th1, seg_per_turn, z_bottom0, left=False):
    """Sweep one region (outer loop CCW + holes CW) along a helix.  Returns V (n,3), F (m,3) outward."""
    loops = [reg["outer"]] + reg["holes"]
    P = np.vstack(loops)
    n = int(math.ceil((th1 - th0) / TAU * seg_per_turn)) + 1
    th = np.linspace(th0, th1, n)
    r = R + P[:, 0]
    sgn = -1.0 if left else 1.0
    X = r[None, :] * np.cos(th)[:, None]
    Y = sgn * r[None, :] * np.sin(th)[:, None]
    Z = P[:, 1][None, :] + z_bottom0 + pitch * th[:, None] / TAU
    V = np.stack([X, Y, Z], -1).reshape(-1, 3)
    m = len(P)
    faces = []
    off = 0
    i = np.arange(n - 1)[:, None]
    for L in loops:
        k = len(L)
        j = np.arange(k)[None, :]
        jn = (j + 1) % k
        a, b, c, d = i * m + off + j, i * m + off + jn, (i + 1) * m + off + jn, (i + 1) * m + off + j
        faces.append(np.stack([a, b, c], -1).reshape(-1, 3))
        faces.append(np.stack([a, c, d], -1).reshape(-1, 3))
        off += k
    T = np.asarray(m3d.triangulate(loops), np.int64)           # caps: indices into the concatenated loops
    faces.append(T[:, ::-1])                                    # start cap
    faces.append(T + (n - 1) * m)                               # end cap
    F = np.concatenate(faces).astype(np.int64)
    if mesh_volume(V, F) < 0:
        F = F[:, ::-1].copy()
    return V, F


def mesh_volume(V, F):
    t = V[F]
    return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)


def to_manifold(V, F):
    man = m3d.Manifold(m3d.Mesh(vert_properties=V.astype(np.float32), tri_verts=F.astype(np.uint32)))
    if man.status() != m3d.Error.NoError:
        sys.exit(f"sweep mesh is not manifold: {man.status()} (self-intersecting profile? gap <= 0? tol too big?)")
    return man


def from_manifold(man):
    mesh = man.to_mesh()
    return np.asarray(mesh.vert_properties, float)[:, :3], np.asarray(mesh.tri_verts, np.int64)


def write_stl(path, V, F, name=b"slinky"):
    """Write binary STL, omitting faces that collapse at its float32 precision."""
    V, F = np.asarray(V), np.asarray(F)
    if V.ndim != 2 or V.shape[1] != 3 or not np.isfinite(V).all():
        raise ValueError("STL vertices must be finite [x, y, z] points")
    if F.ndim != 2 or F.shape[1] != 3 or not np.issubdtype(F.dtype, np.integer):
        raise ValueError("STL faces must be integer vertex-index triples")
    if len(F) and (F.min() < 0 or F.max() >= len(V)):
        raise ValueError("STL face contains an invalid vertex index")
    if len(F) > 0xffffffff:
        raise ValueError("binary STL supports at most 4,294,967,295 triangles")
    if np.any(np.abs(V) > np.finfo(np.float32).max):
        raise ValueError("STL coordinates exceed the float32 range")
    if isinstance(name, str):
        name = name.encode("utf-8")
    record_type = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    with open(path, "wb") as f:
        f.write(name[:80].ljust(80, b"\0"))
        f.write(struct.pack("<I", 0))
        written = 0
        for start in range(0, len(F), 16384):
            tri = V[F[start:start + 16384]].astype(np.float32)
            # Float64 normals avoid overflow when multiplying large coordinates.
            edges = tri.astype(np.float64)
            nrm = np.cross(edges[:, 1] - edges[:, 0], edges[:, 2] - edges[:, 0])
            # Test the coordinates actually stored in STL. Tiny taper faces may
            # collapse during conversion; retain every face with nonzero area.
            keep = np.any(nrm != 0, axis=1)
            tri, nrm = tri[keep], nrm[keep]
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1)[:, None], 1e-30)
            rec = np.zeros(len(tri), dtype=record_type)
            rec["n"], rec["v"] = nrm, tri
            f.write(memoryview(rec))
            written += len(tri)
        f.seek(80)
        f.write(struct.pack("<I", written))


# ----------------------------------------------------------------------------- build
def build(args):
    regions = load_loops(args)
    prof = prepare_profile(regions, args)
    allpts = np.vstack([r["outer"] for r in prof])
    umin, umax = allpts[:, 0].min(), allpts[:, 0].max()
    vmin, vmax = allpts[:, 1].min(), allpts[:, 1].max()
    width, height = umax - umin, vmax - vmin
    R = (args.od / 2 - umax) if args.od else args.radius
    if R + umin <= 0:
        sys.exit(f"profile reaches past the axis (R={R:.2f} + min u={umin:.2f} <= 0): raise --radius/--od or lower --scale")
    pitch = args.pitch if args.pitch else height + args.gap
    gap = pitch - height
    if gap <= 0:
        sys.exit(f"pitch {pitch} must exceed the profile height {height} (coils would fuse)")
    N = args.turns
    wedge = TAU * height / pitch                     # angle over which a plane trim tapers a coil 0 -> t
    d = TAU * 0.03                                   # overshoot so the end caps are never coplanar with a trim
    bt, tt = not args.no_bottom_trim, not args.no_top_trim
    th0 = -d if bt else 0.0
    th1 = TAU * N + (wedge if bt else 0.0) + (wedge + d if tt else 0.0)
    z0 = -height if bt else 0.0
    H = N * pitch + height                           # top trim plane = top of the last full turn's start
    man = None
    for reg in prof:
        V, F = sweep_region(reg, R, pitch, th0, th1, args.seg, z0, args.left)
        mr = to_manifold(V, F)
        man = mr if man is None else man + mr
    if bt:
        man = man.trim_by_plane((0, 0, 1), 0.0)
    if tt:
        man = man.trim_by_plane((0, 0, -1), -H)
    if man.status() != m3d.Error.NoError or man.is_empty():
        sys.exit(f"build failed: {man.status()}")
    bb = man.bounding_box()
    info = dict(prof=prof, R=R, pitch=pitch, gap=gap, width=width, height=height, N=N,
                H=bb[5], umin=umin, umax=umax, wedge_deg=math.degrees(wedge),
                th_end=th1, bottom_trim=bt, top_trim=tt)
    return man, info


# ----------------------------------------------------------------------------- analysis
def layer_check(man, layer, zmax, gap, slack=0.0):
    """Slice like a slicer does (layer mid-heights).  For each layer: area, the part of it that is not
    over the previous layer (= bridged), the fraction of that bridged region which has solid exactly
    `gap` below it (the landing pad on the coil below), and the deepest air under any of it."""
    if not all(math.isfinite(value) for value in (layer, zmax, gap, slack)):
        raise ValueError("layer-check dimensions must be finite")
    if layer <= 0 or zmax < 0 or gap < 0 or slack < 0:
        raise ValueError("layer height must be positive; height, gap and slack cannot be negative")
    if (zmax - layer / 2) / layer > MAX_LAYER_SAMPLES:
        raise ValueError(f"layer check would exceed {MAX_LAYER_SAMPLES:,} slices; "
                         "increase the layer height or disable the layer check")
    n_gap = int(round(gap / layer))
    n_slack = int(round(slack / layer))          # a tapered edge opens the clearance up by this much
    # Only nearby layers are consulted; retaining every slice exhausted memory on tall coils.
    cs = deque(maxlen=max(61, n_gap + n_slack + 2))
    rows = []
    for k, z in enumerate(np.arange(layer / 2, zmax, layer)):
        cs.append(man.slice(float(z)))
        if k == 0:
            continue
        a = cs[-1].area()
        if a <= 0:
            continue
        unsup = cs[-1] - cs[-2]
        ua = unsup.area()
        if ua <= 1e-6 or k <= n_gap:
            continue
        below = cs[-n_gap - 2]
        for j in range(1, n_slack + 1):                         # material under the tapered edge too
            if k - n_gap - 1 - j >= 0:
                below = below + cs[-n_gap - 2 - j]
        land = (unsup ^ below).area() / ua                      # ^ = intersection in manifold3d
        deepest = None
        for j in range(2, min(k, 60) + 1):
            if (unsup ^ cs[-j - 1]).area() >= 0.99 * ua:
                deepest = j - 1
                break
        rows.append((z, a, ua, land, deepest))
    return rows


def spring_estimate(info, man, material):
    G, rho = MATERIALS[material]
    w, t, R, N = info["width"], info["height"], info["R"], info["N"]
    tt = min(w, t) / max(w, t)                        # thin-rectangle torsion constant (Roark) on the bbox
    J = max(w, t) * min(w, t) ** 3 * (1 / 3 - 0.21 * tt * (1 - tt ** 4 / 12))
    A = sum(abs(signed_area(r["outer"])) - sum(abs(signed_area(h)) for h in r["holes"]) for r in info["prof"])
    fill = A / (w * t)                                # crude: scale J by the solid fraction of the bbox
    J *= fill
    k_turn = G * J / (TAU * R ** 3)                   # N/mm per turn  (torsion-dominated helical spring)
    m_turn = man.volume() * rho * 1e-3 / max(info["th_end"] / TAU, 1e-9)   # g per turn (average)
    wt = m_turn * 9.81e-3
    hang = wt * N * (N - 1) / (2 * k_turn)
    dp_top = (N - 1) * wt / k_turn
    gamma = t * dp_top / (TAU * R ** 2)
    return dict(J=J, k_turn=k_turn, K=k_turn / N, m_turn=m_turn, hang=hang, gamma=gamma,
                tau=G * gamma, mass=man.volume() * rho * 1e-3, area=A)


def report(man, info, args):
    R, pitch, gap, w, t, N, H = (info[k] for k in ("R", "pitch", "gap", "width", "height", "N", "H"))
    umin, umax = info["umin"], info["umax"]
    npts = sum(len(r["outer"]) + sum(len(h) for h in r["holes"]) for r in info["prof"])
    nholes = sum(len(r["holes"]) for r in info["prof"])
    est = spring_estimate(info, man, args.material)
    print(f"profile          : {npts} pts, {nholes} hole(s), {w:.3f} wide (radial) x {t:.3f} tall (axial), "
          f"solid area {est['area']:.2f} mm^2, scale {args.scale}")
    print(f"helix            : mean R {R:.2f}  ->  OD {2*(R+umax):.2f} / ID {2*(R+umin):.2f} mm, "
          f"{'left' if args.left else 'right'}-handed")
    print(f"pitch / gap      : {pitch:.3f} / {gap:.3f} mm   (gap = {gap/args.layer:.2f} layers @ {args.layer} mm"
          + ("" if abs(gap / args.layer - round(gap / args.layer)) < 0.02 else
             "  <-- not a whole number of layers: clearance will vary around the coil") + ")")
    ends = ("wedge" if info["bottom_trim"] else "open end face") + " / " + ("wedge" if info["top_trim"] else "open end face")
    print(f"turns / height   : {N} full turns, ends: {ends} ({info['wedge_deg']:.0f} deg per wedge), height {H:.2f} mm")
    band = 360 * args.layer / pitch
    print(f"per-layer bridge : leading {band:.1f} deg of every layer ({math.radians(band)*(R+umax):.1f} mm at the OD) "
          f"is laid over {gap:.2f} mm of air onto the coil below")
    print(f"mesh             : {man.num_tri()} triangles, watertight ({man.status()}), volume {man.volume()/1000:.2f} cm^3")
    print(f"{args.material} estimate     : mass {est['mass']:.1f} g, k = {est['k_turn']*1000:.1f} N/m per turn, "
          f"{est['K']*1000:.2f} N/m whole spring")
    print(f"                   hanging from one end it stretches ~{est['hang']:.0f} mm; top-turn shear strain "
          f"{100*est['gamma']:.2f} % ({est['tau']:.1f} MPa)  [thin-strip torsion model, bbox {w:.2f}x{t:.2f}]")
    if not args.no_layer_check:
        rows = layer_check(man, args.layer, H, gap)
        if not rows:
            print("slice check      : no bridged layers to measure at this layer height")
            return est, rows
        ua = np.array([r[2] for r in rows])
        land = np.array([r[3] for r in rows])
        deep = [r[4] for r in rows if r[4] is not None]
        print(f"slice check      : {len(rows)} layers sliced; bridged area per layer {ua.min():.0f}..{ua.max():.0f} mm^2; "
              f"{100*land.min():.0f}..{100*land.max():.0f} % of it lands on solid {gap:.2f} mm below"
              + (f"; deepest air under any bridged region {max(deep)*args.layer:.2f} mm" if deep else ""))
        return est, rows
    return est, None


# ----------------------------------------------------------------------------- pictures
def section_segments(V, F, phi_deg):
    """Segments of the mesh cut by the half-plane through the axis at azimuth phi (x>0 after rotation)."""
    c, s = math.cos(-math.radians(phi_deg)), math.sin(-math.radians(phi_deg))
    Vr = V.copy()
    Vr[:, 0], Vr[:, 1] = c * V[:, 0] - s * V[:, 1], s * V[:, 0] + c * V[:, 1]
    tri = Vr[F]
    ids, pts = [], []
    for p, q in ((0, 1), (1, 2), (2, 0)):
        yp, yq = tri[:, p, 1], tri[:, q, 1]
        mk = yp * yq < 0
        tt = yp[mk] / (yp[mk] - yq[mk])
        pts.append(tri[mk, p] + tt[:, None] * (tri[mk, q] - tri[mk, p]))
        ids.append(np.nonzero(mk)[0])
    ids, pts = np.concatenate(ids), np.concatenate(pts)
    o = np.argsort(ids, kind="stable")
    ids, pts = ids[o], pts[o]
    u, st, cnt = np.unique(ids, return_index=True, return_counts=True)
    st = st[cnt == 2]
    A, B = pts[st], pts[st + 1]
    keep = (A[:, 0] > 0) & (B[:, 0] > 0)
    return A[keep][:, [0, 2]], B[keep][:, [0, 2]]


def make_section_png(man, info, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    V, F = from_manifold(man)
    R, H, gap, pitch, t = info["R"], info["H"], info["gap"], info["pitch"], info["height"]
    umin, umax = info["umin"], info["umax"]
    x0, x1 = R + umin - 1.0, R + umax + 1.0
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 5, width_ratios=[1.2, 1, 1, 1, 1], height_ratios=[1, 1])
    axf = fig.add_subplot(gs[:, 0])
    A, B = section_segments(V, F, 180.5)
    axf.add_collection(LineCollection(np.stack([A, B], 1), colors="k", linewidths=0.6))
    axf.axhline(0, color="tab:red", lw=0.8, ls="--")
    axf.set_xlim(x0, x1); axf.set_ylim(-1, H + 1); axf.set_aspect("equal")
    axf.set_title(f"section through the axis @180°\n{info['N']} turns, pitch {pitch:.2f}, gap {gap:.2f}", fontsize=9)
    axf.set_xlabel("r [mm]"); axf.set_ylabel("z [mm]")
    zoom = 3 * pitch + t
    for col, phi in enumerate((0.5, 90.5, 180.5, 270.5), start=1):
        A, B = section_segments(V, F, phi)
        for row, (zlo, zhi, ttl) in enumerate(((-0.5, zoom, "bottom"), (H - zoom, H + 0.5, "top"))):
            ax = fig.add_subplot(gs[row, col])
            ax.add_collection(LineCollection(np.stack([A, B], 1), colors="k", linewidths=1.0))
            ax.axhline(0 if row == 0 else H, color="tab:red", lw=0.8, ls="--")
            ax.set_xlim(x0, x1); ax.set_ylim(zlo, zhi); ax.set_aspect("equal")
            ax.set_title(f"{ttl} @ {phi-0.5:.0f}°", fontsize=9)
            ax.tick_params(labelsize=7)
    fig.suptitle("bed plane = red dashed line; every coil rests on "
                 f"{gap:.2f} mm of air above the one below", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def make_profile_png(info, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    for r in info["prof"]:
        o = np.vstack([r["outer"], r["outer"][:1]])
        ax.fill(o[:, 0], o[:, 1], color="#f0c070")
        ax.plot(o[:, 0], o[:, 1], "k-", lw=1)
        for h in r["holes"]:
            hh = np.vstack([h, h[:1]])
            ax.fill(hh[:, 0], hh[:, 1], color="white")
            ax.plot(hh[:, 0], hh[:, 1], "k-", lw=1)
    ax.axvline(0, color="tab:blue", lw=0.8, ls="--")
    ax.set_aspect("equal"); ax.grid(True, lw=0.3)
    ax.set_xlabel("u = radial offset from mean radius [mm]  (+ outward)")
    ax.set_ylabel("v = axial [mm]")
    ax.set_title(f"coil cross-section as swept ({info['width']:.2f} x {info['height']:.2f} mm)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    return path


def render(man, path, size=900, elev=28.0, azim=-35.0, ortho_zoom=0.92, center=None, extent=None):
    """Tiny orthographic z-buffer renderer (numpy only) -- a clean shaded preview of the STL.
    center/extent (mm) frame a detail instead of the whole part."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    V, F = from_manifold(man)
    if not len(V) or not len(F):
        raise ValueError("cannot render an empty mesh")
    if not isinstance(size, (int, np.integer)) or size <= 0:
        raise ValueError("render size must be a positive integer")
    if not math.isfinite(ortho_zoom) or ortho_zoom <= 0:
        raise ValueError("render zoom must be finite and positive")
    if extent is not None and (not math.isfinite(extent) or extent <= 0):
        raise ValueError("render extent must be finite and positive")
    az, el = math.radians(azim), math.radians(elev)
    Rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, math.cos(el), -math.sin(el)], [0, math.sin(el), math.cos(el)]])
    C = V - ((V.min(0) + V.max(0)) / 2 if center is None else np.asarray(center, float))
    P = C @ Rz.T @ Rx.T                                   # screen x = P[:,0], screen y = P[:,2], depth = P[:,1]
    ss = 2
    W = size * ss
    half = max(np.abs(P[:, 0]).max(), np.abs(P[:, 2]).max()) if extent is None else extent / 2
    scale = ortho_zoom * W / (2 * half)
    sx = P[:, 0] * scale + W / 2
    sy = W / 2 - P[:, 2] * scale
    dep = P[:, 1]
    tri = P[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1)[:, None], 1e-12)
    # The camera looks from -Y. Closed manifold back faces cannot contribute visible pixels.
    visible = n[:, 1] < -1e-12
    F, n = F[visible], n[visible]
    light = np.array([-0.45, -0.7, 0.55]); light /= np.linalg.norm(light)
    shade = 0.28 + 0.72 * np.clip(n @ light, 0, 1) ** 0.9
    base = np.array([0.93, 0.56, 0.16])
    zbuf = np.full((W, W), np.inf, dtype=np.float32)
    img = np.ones((W, W, 3), dtype=np.float32)
    X0, Y0, D0 = sx[F], sy[F], dep[F]
    for k in range(len(F)):
        x, y, d = X0[k], Y0[k], D0[k]
        xa, xb = int(max(0, math.floor(x.min()))), int(min(W - 1, math.ceil(x.max())))
        ya, yb = int(max(0, math.floor(y.min()))), int(min(W - 1, math.ceil(y.max())))
        if xa > xb or ya > yb:
            continue
        gx, gy = np.meshgrid(np.arange(xa, xb + 1) + 0.5, np.arange(ya, yb + 1) + 0.5)
        det = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
        if abs(det) < 1e-12:
            continue
        l1 = ((gx - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (gy - y[0])) / det
        l2 = ((x[1] - x[0]) * (gy - y[0]) - (gx - x[0]) * (y[1] - y[0])) / det
        l0 = 1 - l1 - l2
        inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        if not inside.any():
            continue
        z = l0 * d[0] + l1 * d[1] + l2 * d[2]
        sub = zbuf[ya:yb + 1, xa:xb + 1]
        upd = inside & (z < sub)
        sub[upd] = z[upd]
        img[ya:yb + 1, xa:xb + 1][upd] = base * shade[k]
    img = img.reshape(size, ss, size, ss, 3).mean((1, 3))
    plt.imsave(path, np.clip(img, 0, 1))
    return path


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("profile source")
    src.add_argument("--step", type=str, default="", help="STEP file of a straight extrusion (needs cadquery)")
    src.add_argument("--loops", type=str, default="", help="JSON loops file written by --save-loops")
    src.add_argument("--profile", type=str, default="", help='"u,v; u,v; ..." polygon in mm')
    src.add_argument("--width", type=float, default=5.0, help="default rectangle: radial width [mm]")
    src.add_argument("--thickness", type=float, default=1.0, help="default rectangle: axial thickness [mm]")
    src.add_argument("--save-loops", type=str, default="", help="write the raw (unscaled) loops to this JSON")
    geo = ap.add_argument_group("profile mapping")
    geo.add_argument("--scale", type=float, default=1.0, help="scale factor applied to the profile")
    geo.add_argument("--flip-radial", action="store_true", help="profile +x points outward instead of inward")
    geo.add_argument("--mirror", action="store_true", help="flip the profile upside down (axial)")
    geo.add_argument("--tol", type=float, default=0.02, help="curve simplification tolerance after scaling [mm]")
    geo.add_argument("--min-hole-area", type=float, default=0.5, help="drop holes smaller than this after scaling [mm^2]")
    coil = ap.add_argument_group("coil")
    coil.add_argument("--radius", type=float, default=30.0, help="helix mean radius at u=0 [mm]")
    coil.add_argument("--od", type=float, default=0.0, help="instead of --radius: outer diameter of the coil [mm]")
    coil.add_argument("--gap", type=float, default=0.4, help="air between coils at rest [mm] (use N x layer height)")
    coil.add_argument("--pitch", type=float, default=0.0, help="explicit pitch [mm] (default: profile height + gap)")
    coil.add_argument("--turns", type=int, default=40, help="full-thickness turns")
    coil.add_argument("--left", action="store_true", help="left-handed helix")
    coil.add_argument("--no-bottom-trim", action="store_true", help="keep the raw start face (needs supports to print)")
    coil.add_argument("--no-top-trim", action="store_true", help="keep the raw end face (shows the profile)")
    coil.add_argument("--seg", type=int, default=180, help="sweep stations per turn")
    out = ap.add_argument_group("output / analysis")
    out.add_argument("--layer", type=float, default=0.2, help="layer height you will slice with [mm] (analysis only)")
    out.add_argument("--material", choices=MATERIALS, default="PLA")
    out.add_argument("--no-layer-check", action="store_true", help="skip the slicer-like layer verification")
    out.add_argument("--png", action="store_true", help="write <out>_section/_iso/_end/_profile PNGs")
    out.add_argument("--out", type=str, default="slinky.stl")
    args = ap.parse_args()

    if args.save_loops and args.step:
        regs = loops_from_step(args.step)
        json.dump([{"outer": r["outer"].tolist(), "holes": [h.tolist() for h in r["holes"]]} for r in regs],
                  open(args.save_loops, "w"))
        print("wrote", args.save_loops)

    man, info = build(args)
    V, F = from_manifold(man)
    write_stl(args.out, V, F)
    print(f"wrote {args.out}  ({len(F)} triangles)")
    report(man, info, args)
    if args.png:
        base = args.out[:-4] if args.out.lower().endswith(".stl") else args.out
        print("wrote", make_profile_png(info, args, base + "_profile.png"))
        print("wrote", make_section_png(man, info, args, base + "_section.png"))
        print("wrote", render(man, base + "_iso.png"))
        if not info["top_trim"]:
            # look straight at the top end face: its outward normal is +e_theta(th_end)
            th = info["th_end"]
            az = 180.0 - math.degrees(th) % 360.0
            sgn = -1.0 if args.left else 1.0
            if args.left:
                az = -az
            zc = info["H"] - info["height"] / 2
            ctr = (info["R"] * math.cos(th), sgn * info["R"] * math.sin(th), zc)
            ext = 1.6 * max(info["width"], info["height"])
            print("wrote", render(man, base + "_end.png", elev=10.0, azim=az, center=ctr, extent=ext))


if __name__ == "__main__":
    main()
