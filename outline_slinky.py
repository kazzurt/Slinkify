#!/usr/bin/env python3
"""
outline_slinky.py -- turn ANY extrusion profile into a slinky: the profile's OUTLINE becomes the
coil path (a non-circular helix), and a flat strip is coiled along it with a constant pitch.
Seen from above the slinky has the silhouette of the extrusion; a circle gives an ordinary slinky.

    profile (STEP of a straight extrusion, or a saved loops JSON, or a circle by default)
      -> outer contour = coil path, length L
      -> strip = band of width W just inside the contour (--band inward|centered|outward), T thick
      -> z rises by pitch P per lap:  z = k*P + P*s/L   (s = arc length along the contour)
      -> N laps; the first lap is filled solid down to the bed (full first layer, no wedge),
         the last lap is filled up to a flat top
    pitch P = T + gap   (print-in-place clearance, gap = whole number of layer heights)

Optionally the hole contours (letter counters) become their own nested coils (--holes); keep W
below half the narrowest letter stroke or the bands overlap and the coils fuse.

USAGE
    python outline_slinky.py --loops DDR_profile_loops.json --width 3 --thickness 1 --gap 0.4 --turns 40 --png
    python outline_slinky.py --step DDR.step --scale 1.5 --width 3 --holes --png
    python outline_slinky.py --circle 60 --width 5 --turns 40          # plain 60 mm slinky, for comparison

Requires numpy + manifold3d (and slinky.py next to this file); cadquery only for --step; matplotlib for --png.
"""
import argparse
import json
import math
import sys

import numpy as np
import manifold3d as m3d

from slinky import (loops_from_step, rdp, dedupe, signed_area, write_stl, from_manifold, validate_regions,
                    layer_check, render, MATERIALS, TAU)


# ----------------------------------------------------------------------------- path helpers
def polyline_lengths(P):
    seg = np.roll(P, -1, axis=0) - P
    ln = np.linalg.norm(seg, axis=1)
    return seg, ln, np.concatenate([[0.0], np.cumsum(ln)[:-1]]), float(ln.sum())


def pick_seam(P):
    """Midpoint of the longest straight segment -> (index, point, unit tangent)."""
    seg, ln, _, _ = polyline_lengths(P)
    i = int(np.argmax(ln))
    C0 = P[i] + 0.5 * seg[i]
    T0 = seg[i] / ln[i]
    return i, C0, T0


def start_at(P, i, C0):
    """Rotate the closed polygon so it starts at C0 (inserted on segment i)."""
    Q = np.vstack([P[i + 1:], P[:i + 1]])            # starts at P[i+1], ends with P[i]
    return np.vstack([C0[None, :], Q])              # C0 -> P[i+1] ... P[i] (-> back to C0)


def nearest_s(V, P):
    """Arc length at each closest point, with bounded temporary memory for large profiles."""
    V = np.asarray(V, float)
    P = np.asarray(P, float)
    seg, ln, S, L = polyline_lengths(P)
    result = np.empty(len(V))
    chunk_size = max(1, 131072 // max(1, len(P)))
    for start in range(0, len(V), chunk_size):
        pts = V[start:start + chunk_size]
        W = pts[:, None, :] - P[None, :, :]
        t = np.clip(np.einsum("nmk,mk->nm", W, seg) / np.maximum(ln[None, :] ** 2, 1e-18), 0.0, 1.0)
        W -= t[:, :, None] * seg[None, :, :]
        j = np.argmin(np.einsum("nmk,nmk->nm", W, W), axis=1)
        result[start:start + len(pts)] = S[j] + t[np.arange(len(pts)), j] * ln[j]
    return result, L


# ----------------------------------------------------------------------------- lap stacking
def half_planes(C0, T0, N0, big):
    """The two half-planes either side of the seam line (through C0, along N0)."""
    Hp = m3d.CrossSection([np.array([C0, C0 + big * T0, C0 + big * T0 + big * N0, C0 + big * N0,
                                     C0 - big * N0, C0 + big * T0 - big * N0, C0 + big * T0])])
    Hm = m3d.CrossSection([np.array([C0, C0 + big * N0, C0 - big * T0 + big * N0, C0 - big * T0,
                                     C0 - big * T0 - big * N0, C0 - big * N0])])
    return Hp, Hm


def edge_profile(T, e_bot, e_top, step, kind):
    """Vertical build-up of one lap as [(dz, height, inset), ...] covering [0, T].

    `inset` is how far the 2D region is pulled in (material removed all round the profile edge) for
    that slice, so the lap's top and bottom edges are chamfered ("chamfer", a 45 deg cut) or filleted
    ("round", a quarter circle of radius e).  The taper is built as sub-slabs of `step` (the layer
    height by default), i.e. exactly the staircase the printer will make anyway, and each sub-slab is
    inscribed in the true taper (it takes the largest inset it spans), so the result never has more
    material than the ideal chamfer."""
    if kind == "none" or (e_bot <= 0 and e_top <= 0):
        return [(0.0, T, 0.0)]

    def inset_at(d, e):                       # d = distance in from the edge (0 at the face)
        if d >= e:
            return 0.0
        if kind == "chamfer":
            return e - d
        return e - math.sqrt(max(0.0, e * e - (e - d) ** 2))     # quarter round of radius e

    out = []
    if e_bot > 0:
        n = max(1, int(math.ceil(e_bot / step - 1e-9)))
        h = e_bot / n
        for j in range(n):
            out.append((j * h, h, inset_at(j * h, e_bot)))        # largest inset in the slab
    mid = T - e_bot - e_top
    if mid <= 1e-9:                                   # stack_laps clamps e, so this cannot normally happen
        raise ValueError(f"edge size too large: {e_bot:.2f} + {e_top:.2f} >= sheet thickness {T:.2f}")
    out.append((e_bot, mid, 0.0))
    if e_top > 0:
        n = max(1, int(math.ceil(e_top / step - 1e-9)))
        h = e_top / n
        for j in range(n):
            z_a = T - e_top + j * h
            out.append((z_a, h, inset_at(T - (z_a + h), e_top)))   # measured from the top face
    return out


def stack_laps(cs, C0, T0, N0, big, fsample, T, p, N, base=0.0, top="fill", pitches=None,
               edge="none", edge_size=0.0, edge_step=0.2):
    """Stack N laps of the 2D region `cs` (a CrossSection) using the rise field fsample(points, side)
    in [0, 1].  With a constant pitch p, lap k occupies z in [base + k p + p f, + T]; with a list of
    per-lap `pitches` (a gap-ladder coupon) lap k occupies [base + S_k + p_k f, + T], S_k = sum of the
    earlier pitches, so the helix stays continuous and the gap above lap k ramps from g_k to g_k+1.
    The first lap is filled down to the bed (plus an optional flat plate of height `base`), so the
    bottom is one solid full-profile face and the first helical gap only starts one lap up; the last
    lap is filled up to a flat top (top="fill") or left as a plain lap (top="open").  `edge` tapers
    each lap's top and bottom edges (never the faces that meet the bed or the flat top)."""
    if not isinstance(N, (int, np.integer)) or isinstance(N, bool) or N < 1:
        raise ValueError("turns must be a positive whole number")
    if N > 1000:
        raise ValueError("turns must be 1000 or fewer; reduce the target height or lap count")
    if not math.isfinite(T) or T <= 0 or not math.isfinite(base) or base < 0:
        raise ValueError("thickness must be positive and base must be nonnegative")
    if top not in ("fill", "open") or edge not in ("none", "chamfer", "round"):
        raise ValueError("invalid top or edge mode")
    if not math.isfinite(edge_size) or edge_size < 0 or not math.isfinite(edge_step) or edge_step <= 0:
        raise ValueError("edge size must be nonnegative and edge step must be positive")
    P = list(pitches) if pitches is not None else [p] * N
    if len(P) != N:
        raise ValueError("pitches must have one entry per lap")
    if any(not math.isfinite(pk) or pk < T for pk in P):
        raise ValueError("each pitch must be finite and at least the sheet thickness")
    if N == 1 and top == "fill":
        # Both faces are flat on a filled single lap, so there are no free lap edges to taper.
        return m3d.Manifold.extrude(cs, base + P[0] + T)
    S = [0.0]
    for pk in P:
        S.append(S[-1] + pk)
    e = float(edge_size) if edge != "none" else 0.0
    e_max = 0.4 * T                                   # keep at least 20 % of the sheet at full width
    if e > e_max:
        print(f"note             : taper {e:.2f} mm clamped to {e_max:.2f} mm (40 % of the {T:.2f} mm sheet, "
              f"so {T - 2*e_max:.2f} mm stays full width)")
        e = e_max
    edge_step = max(1e-3, min(edge_step, e if e > 0 else 1.0))
    prof_mid = edge_profile(T, e, e, edge_step, edge)          # a lap in free air: both edges tapered
    prof_bot = edge_profile(T, 0.0, e, edge_step, edge)        # lap 0: underside is filled to the bed
    prof_top = edge_profile(T, e, 0.0, edge_step, edge)        # last lap when filled to a flat top
    insets = sorted({ins for pr in (prof_mid, prof_bot, prof_top) for _, _, ins in pr})
    Hp, Hm = half_planes(C0, T0, N0, big)
    inset_regions = {ins: cs if ins <= 0 else cs.offset(-ins, m3d.JoinType.Round) for ins in insets}
    if any(region.is_empty() for region in inset_regions.values()):
        raise ValueError(f"edge size {e:.2f} mm eats the whole profile -- reduce it or scale up")
    pieces = []
    for side, half in ((+1.0, Hp), (-1.0, Hm)):
        # offset the FULL cross-section first, then cut at the seam, so the seam face stays full height
        parts = {}
        for ins, region in inset_regions.items():
            parts[ins] = region ^ half
        if all(pt.is_empty() for pt in parts.values()):
            continue

        def fv(V, side=side):
            return fsample(V[:, :2], side)

        slab_cache = {}
        # All taper slices must use the same XY triangulation. Warping separately
        # triangulated inset footprints samples the nonlinear rise field on different
        # faces, leaving cracks between adjoining sub-slabs. Warp the full footprint
        # first, then trim it with each inset's vertical mask.
        clip_margin = max(P) + T
        clip_height = base + S[-1] + T + 2 * clip_margin
        inset_masks = {
            ins: m3d.Manifold.extrude(part, clip_height).translate([0, 0, -clip_margin])
            for ins, part in parts.items() if ins > 0 and not part.is_empty()
        }

        def slab(ins, h, place, overlap_low=0.0, overlap_high=0.0):
            """A sub-slab of height h, warped into place by `place(V_z, f) -> z`."""
            part = parts[ins]
            if part.is_empty() or h <= 1e-9:
                return None
            key = h
            if key not in slab_cache:
                slab_cache[key] = [m3d.Manifold.extrude(parts[0.0], h), None, None]
            cached = slab_cache[key]
            prism = cached[0]

            def warp(V):
                # Every lap uses the same XY mesh. Reuse its rise values instead of running
                # nearest-contour searches or harmonic interpolation once per lap.
                if cached[1] is None or not np.array_equal(cached[1], V[:, :2]):
                    cached[1] = V[:, :2].copy()
                    cached[2] = fv(V)
                f = cached[2]
                out = V.copy()
                out[:, 2] = place(V[:, 2], f)
                # Let adjoining internal taper slices overlap by a submicron
                # tolerance. Independently clipped, exactly coincident triangle
                # faces otherwise leave zero-volume Boolean artifacts. The lap's
                # external top/bottom faces and its clearance are unchanged.
                out[:, 2] += np.where(V[:, 2] < h / 2, -overlap_low, overlap_high)
                return out
            m = prism.warp_batch(warp)
            if ins > 0:
                m = m ^ inset_masks[ins]
            if m.status() != m3d.Error.NoError:
                raise ValueError(f"warp failed: {m.status()}")
            return m

        for k in range(N):
            pk, Sk = P[k], S[k]
            if k == 0:
                prof = prof_bot
            elif k == N - 1 and top == "fill":
                prof = prof_top
            else:
                prof = prof_mid
            overlap = min(1e-6, min(height for _, height, _ in prof) * 1e-3)
            for i, (dz, h, ins) in enumerate(prof):
                overlap_low = overlap if i > 0 else 0.0
                overlap_high = overlap if i < len(prof) - 1 else 0.0
                if k == 0 and i == 0:
                    # stretch the lowest element down to the bed: 0 .. base + p0 f + h
                    m = slab(ins, pk + h, lambda z, f, h=h, pk=pk: z * (base + pk * f + h) / (pk + h),
                             overlap_low, overlap_high)
                elif k == N - 1 and top == "fill" and i == len(prof) - 1:
                    # stretch the highest element up to the flat top at base + S_N + T
                    m = slab(ins, pk + h,
                             lambda z, f, h=h, pk=pk, Sk=Sk, dz=dz: base + Sk + pk * f + dz
                             + z * (pk + T - pk * f - dz) / (pk + h), overlap_low, overlap_high)
                else:
                    m = slab(ins, h, lambda z, f, pk=pk, Sk=Sk, dz=dz: z + base + Sk + pk * f + dz,
                             overlap_low, overlap_high)
                if m is not None:
                    pieces.append(m)
    man = m3d.Manifold.batch_boolean(pieces, m3d.OpType.Add)
    if man.status() != m3d.Error.NoError:
        raise ValueError(f"lap stacking failed: {man.status()}")
    return man


# ----------------------------------------------------------------------------- band coil
def coil_from_contour(P, width, thickness, pitch, N, band="inward", left=False, seam_run=1.0,
                      base=0.0, top="fill", pitches=None, edge="none", edge_size=0.0, edge_step=0.2):
    """P: closed CCW polygon (mm).  A strip of `width` along P, N laps, rise = arc length / lap length."""
    P = validate_regions([{"outer": P}])[0]["outer"]
    if not math.isfinite(width) or width <= 0:
        raise ValueError("band width must be positive")
    if band not in ("inward", "outward", "centered"):
        raise ValueError("invalid band mode")
    region = m3d.CrossSection([P])
    if band == "inward":
        inner = region.offset(-width, m3d.JoinType.Round)
        if N > 1 and inner.is_empty():
            raise ValueError("Band width removes the interior opening needed for a continuous spring. "
                             "Reduce band width or increase the profile size.")
        cs = region - inner
    elif band == "outward":
        cs = region.offset(width, m3d.JoinType.Round) - region
    else:
        inner = region.offset(-width / 2, m3d.JoinType.Round)
        if N > 1 and inner.is_empty():
            raise ValueError("Band width removes the interior opening needed for a continuous spring. "
                             "Reduce band width or increase the profile size.")
        cs = region.offset(width / 2, m3d.JoinType.Round) - inner
    if cs.is_empty():
        raise ValueError("band is empty -- check the profile and band width")
    if left:
        P = P[::-1].copy()                            # reverse travel only; material stays CCW
    i, C0, T0 = pick_seam(P)
    N0 = np.array([-T0[1], T0[0]])
    _, ln_all, _, L = polyline_lengths(P)
    seam_run = min(seam_run, 0.4 * float(ln_all[i]))   # only vertices this close to the seam can wrap
    Ps = start_at(P, i, C0)

    def fsample(P2, side):
        s, _ = nearest_s(P2, Ps)
        if side > 0:
            s = np.where(s > L - seam_run, s - L, s)       # vertices on the seam face: s = 0
        else:
            s = np.where(s < seam_run, s + L, s)           # ... or s = L on the other side
        return s / L

    big = 10 * (np.abs(P).max() + width)
    man = stack_laps(cs, C0, T0, N0, big, fsample, thickness, pitch, N, base, top, pitches,
                     edge, edge_size, edge_step)
    return man, L


# ----------------------------------------------------------------------------- staggered perimeter bridges
def contour_arc(P, start, length, spacing=0.5):
    """An open, forward arc of a closed contour, including corners and both ends."""
    seg, ln, S, L = polyline_lengths(P)
    if not 0 < length < L:
        raise ValueError("Bridge length must be shorter than the outline perimeter.")
    start %= L
    end = start + length
    distances = [start, end]
    for shift in (0, L):
        distances.extend(s + shift for s in S if start < s + shift < end)
    distances = sorted(set(distances))
    dense = []
    for a, b in zip(distances, distances[1:]):
        dense.extend(np.linspace(a, b, max(1, math.ceil((b - a) / spacing)), endpoint=False))
    dense.append(end)
    s = np.asarray(dense) % L
    indices = np.minimum(np.searchsorted(S, s, side="right") - 1, len(P) - 1)
    return P[indices] + seg[indices] * ((s - S[indices]) / ln[indices])[:, None]


def open_arc_fraction(points, arc):
    """Closest arc-length fraction on an open arc; never closes the end back to the start."""
    seg = np.diff(arc, axis=0)
    ln = np.linalg.norm(seg, axis=1)
    S = np.concatenate(([0.0], np.cumsum(ln)[:-1]))
    result = np.empty(len(points))
    chunk = max(1, 131072 // len(seg))
    for start in range(0, len(points), chunk):
        q = points[start:start + chunk]
        d = q[:, None, :] - arc[:-1]
        t = np.clip(np.einsum("nmk,mk->nm", d, seg) / ln[None, :] ** 2, 0, 1)
        d -= t[:, :, None] * seg
        nearest = np.argmin(np.einsum("nmk,nmk->nm", d, d), axis=1)
        result[start:start + len(q)] = (S[nearest] + t[np.arange(len(q)), nearest] * ln[nearest]) / ln.sum()
    return result


def perimeter_bridge_patch(cs, arc, depth):
    """Buffer an open perimeter arc and clip it to the actual material footprint."""
    patches = []
    for a, b in zip(arc, arc[1:]):
        normal = np.array([-(b - a)[1], (b - a)[0]])
        normal *= depth / np.linalg.norm(normal)
        patches.append(m3d.CrossSection([np.array([a - normal, b - normal, b + normal, a + normal])]))
    # Round joins keep a ramp connected as it follows a corner. End caps are flat.
    for i in range(1, len(arc) - 1):
        before, after = arc[i] - arc[i - 1], arc[i + 1] - arc[i]
        if abs(before[0] * after[1] - before[1] * after[0]) > 1e-8:
            patches.append(m3d.CrossSection.circle(depth, 24).translate(arc[i]))
    patch = cs ^ m3d.CrossSection.batch_boolean(patches, m3d.OpType.Add)
    if patch.is_empty():
        raise ValueError("A perimeter bridge misses the material. Increase bridge depth, "
                         "change bridge advance, or use Continuous helix.")
    # A buffer can catch a nearby, separate stroke across a notch. Keep one local
    # connected ramp, never turn the detached scrap into an extra inter-lap tie.
    patch = max(patch.decompose(), key=lambda part: part.area())
    # Sampling the curved arc on the mesh boundary avoids one long, coarse ramp face.
    return m3d.CrossSection([densify(polygon, 0.5) for polygon in patch.to_polygons()])


def staggered_laps(cs, P, T, pitch, N, base=0.0, top="fill", pitches=None,
                   edge="none", edge_size=0.0, edge_step=0.1, left=False,
                   advance=25.0, bridge_length=6.0, bridge_depth=2.0, capture_preview=False):
    """Flat sheets joined ONLY by local ramps, advancing along the perimeter.

    These are linked closed laps rather than the original continuous helical ribbon.
    At each inter-lap Z gap only its own local ramp contains material. The print
    has substantially larger unsupported areas and needs its own physical coupon.
    """
    if not 0 < advance < 100:
        raise ValueError("Bridge advance must be greater than 0 and less than 100 percent.")
    if not math.isfinite(bridge_length) or bridge_length <= 0 or not math.isfinite(bridge_depth) or bridge_depth <= 0:
        raise ValueError("Bridge length and depth must be positive finite numbers.")
    if not isinstance(N, (int, np.integer)) or isinstance(N, bool) or not 1 <= N <= 1000:
        raise ValueError("turns must be a whole number from 1 to 1000")
    if top not in ("fill", "open") or edge not in ("none", "round", "chamfer"):
        raise ValueError("invalid top or edge mode")
    if not math.isfinite(T) or T <= 0 or not math.isfinite(base) or base < 0:
        raise ValueError("thickness must be positive and base must be nonnegative")
    if not math.isfinite(edge_size) or edge_size < 0 or not math.isfinite(edge_step) or edge_step <= 0:
        raise ValueError("edge size must be nonnegative and edge step must be positive")
    intervals = list(pitches) if pitches is not None else [pitch] * N
    if len(intervals) != N or any(not math.isfinite(p) or p <= T for p in intervals):
        raise ValueError("each pitch must be finite and greater than sheet thickness")
    i, C0, _ = pick_seam(P)
    path = start_at(P, i, C0)
    if left:
        path = np.vstack([path[0], path[:0:-1]])
    _, _, _, L = polyline_lengths(path)
    if bridge_length >= L:
        raise ValueError("Bridge length must be shorter than the outline perimeter.")
    e = min(edge_size, 0.4 * T) if edge != "none" else 0.0
    # Keep ramps inside the full-height footprint of the tapered sheets. Letting
    # an inclined ramp intersect a curved taper creates near-coincident slivers
    # which can collapse when written at STL's float32 precision.
    ramp_section = cs.offset(-e - 0.02, m3d.JoinType.Round) if e > 0 else cs
    pieces, bridges, bridge_ids, preview_ramps = [], [], {}, []
    inset_cache = {0.0: cs}
    z = base
    for k in range(N):
        profile = edge_profile(T, e if k else 0.0, e if k < N - 1 or top == "open" else 0.0,
                               edge_step, edge)
        for j, (dz, height, inset) in enumerate(profile):
            if inset not in inset_cache:
                inset_cache[inset] = cs.offset(-inset, m3d.JoinType.Round)
            section = inset_cache[inset]
            if section.is_empty():
                raise ValueError("Lap taper removes the material; reduce taper size or enlarge the profile.")
            low = z + dz
            high = low + height
            # Internal sub-slabs overlap by a submicron to prevent zero-volume
            # Boolean shells at exactly coincident taper faces. External faces stay put.
            if j > 0:
                low -= 1e-6
            if j < len(profile) - 1:
                high += 1e-6
            if k == 0 and j == 0:
                low = 0.0
            pieces.append(m3d.Manifold.extrude(section, high - low).translate([0, 0, low]))
        if k < N - 1:
            start = (k * advance / 100.0 * L) % L
            arc = contour_arc(path, start, bridge_length)
            patch = perimeter_bridge_patch(ramp_section, arc, bridge_depth)
            ramp = m3d.Manifold.extrude(patch, T)
            if capture_preview:
                # Boolean surface provenance identifies exposed ramp faces exactly,
                # including curved/corner ramps; no bounding-box color approximation.
                bridge_ids[ramp.original_id()] = k
            pk = intervals[k]

            def warp(vertices, z=z, pk=pk, arc=arc):
                out = vertices.copy()
                out[:, 2] += z + pk * open_arc_fraction(vertices[:, :2], arc)
                return out

            ramp = ramp.warp_batch(warp)
            if ramp.status() != m3d.Error.NoError:
                raise ValueError(f"perimeter ramp failed: {ramp.status()}")
            pieces.append(ramp)
            if capture_preview:
                preview_ramps.append((k, ramp))
            bridges.append(dict(lap=k, start_fraction=start / L, length=bridge_length,
                                depth=bridge_depth, start_xy=arc[0].tolist(), end_xy=arc[-1].tolist(),
                                lower_z=z, upper_z=z + pk, gap=pk - T))
            z += pk
    man = m3d.Manifold.batch_boolean(pieces, m3d.OpType.Add)
    preview = man if capture_preview else None
    # Release Boolean provenance constraints and collapse tiny coplanar faces
    # before float32 STL export. Surface displacement is bounded by this tolerance.
    cleanup_tolerance = max(1e-5, max(abs(value) for value in man.bounding_box()) * np.finfo(np.float32).eps * 2)
    man = man.as_original().simplify(cleanup_tolerance)
    if man.status() != m3d.Error.NoError or man.is_empty() or len(man.decompose()) != len(cs.decompose()):
        raise ValueError("Staggered bridges did not connect every lap. Reduce the taper, increase bridge depth, "
                         "or change bridge advance/length.")
    return man, dict(L=L, bridges=bridges, cleanup_tolerance_mm=cleanup_tolerance,
                     preview_manifold=preview, preview_bridge_ids=bridge_ids, preview_ramps=preview_ramps)


def staggered_from_contour(P, width, thickness, pitch, N, band="inward", left=False,
                           base=0.0, top="fill", pitches=None, edge="none", edge_size=0.0,
                           edge_step=0.1, advance=25.0, bridge_length=6.0, bridge_depth=2.0,
                           capture_preview=False):
    region = m3d.CrossSection([P])
    if band == "inward":
        cs = region - region.offset(-width, m3d.JoinType.Round)
    elif band == "outward":
        cs = region.offset(width, m3d.JoinType.Round) - region
    elif band == "centered":
        cs = region.offset(width / 2, m3d.JoinType.Round) - region.offset(-width / 2, m3d.JoinType.Round)
    else:
        raise ValueError("invalid band mode")
    return staggered_laps(cs, P, thickness, pitch, N, base, top, pitches,
                          edge, edge_size, edge_step, left, advance, bridge_length, bridge_depth, capture_preview)


# ----------------------------------------------------------------------------- full-stroke coil
def densify(P, maxlen):
    if not math.isfinite(maxlen) or maxlen <= 0:
        raise ValueError("maximum segment length must be positive")
    out = []
    for a, b in zip(P, np.roll(P, -1, axis=0)):
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / maxlen)))
        for k in range(n):
            out.append(a + (b - a) * k / n)
    return np.array(out)


def harmonic_rise(outer, holes, h, seam_i=None):
    """Rise field f on the region (outer minus holes), 0..1 once around the outer contour:
    f = s/L on the outer contour, no-flux on the holes, and a slit from the seam point inward to the
    first hole carries the jump 0 -> 1.  Solved as Laplace's equation on a grid of cell size h.
    Returns a sampler f(points, side) plus the seam geometry."""
    from matplotlib.path import Path
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    from scipy.spatial import cKDTree

    if not math.isfinite(h) or h <= 0:
        raise ValueError("rise-field grid spacing must be positive")
    # Check before tracing the slit or allocating arrays: a tiny grid step can otherwise
    # spend minutes in the slit search and exhaust memory during the sparse solve.
    allp = np.vstack([outer] + holes)
    x0, y0 = allp.min(0) - 2 * h + h / 3
    x1, y1 = allp.max(0) + 2 * h
    nx, ny = int(math.ceil((x1 - x0) / h)), int(math.ceil((y1 - y0) / h))
    if nx * ny > 1_000_000:
        suggested = h * math.sqrt(nx * ny / 1_000_000) * 1.05
        raise ValueError(f"rise-field grid would contain {nx * ny:,} cells; increase grid spacing "
                         f"to at least {suggested:.3g} mm or reduce the profile size")
    if seam_i is None:
        i, C0, T0 = pick_seam(outer)
    else:                                              # seam given as (C0, T0): find its segment on this polygon
        C0, T0 = seam_i
        seg, ln, _, _ = polyline_lengths(outer)
        t = np.clip(((C0 - outer) * seg).sum(1) / np.maximum(ln ** 2, 1e-18), 0, 1)
        d2 = ((outer + t[:, None] * seg - C0) ** 2).sum(1)
        i = int(np.argmin(d2))
    N0 = np.array([-T0[1], T0[0]])                     # left normal = inward for a CCW contour
    Ps = start_at(outer, i, C0)
    _, _, _, L = polyline_lengths(Ps)
    # slit: from C0 inward until we leave the region
    paths = [Path(outer)] + [Path(hh) for hh in holes]

    def inside(pts):
        m = paths[0].contains_points(pts)
        for ph in paths[1:]:
            m &= ~ph.contains_points(pts)
        return m
    step = h / 4
    d = step
    while inside(np.array([C0 + d * N0]))[0]:
        d += step
    slit_len = d
    C1 = C0 + slit_len * N0
    hit_hole = any(ph.contains_points(np.array([C0 + (slit_len + step) * N0]))[0] for ph in paths[1:])
    if not hit_hole:
        slit_len *= 0.5                                # no hole behind the seam: stop half way (a pole)
        C1 = C0 + slit_len * N0

    # grid (origin nudged by h/3 so no cell centre can sit exactly on the slit line)
    xs = x0 + (np.arange(nx) + 0.5) * h
    ys = y0 + (np.arange(ny) + 0.5) * h
    GX, GY = np.meshgrid(xs, ys)                        # shape (ny, nx)
    cells = np.column_stack([GX.ravel(), GY.ravel()])
    mask = inside(cells).reshape(ny, nx)
    idx = -np.ones((ny, nx), int)
    idx[mask] = np.arange(mask.sum())
    n = mask.sum()
    if n == 0:
        raise ValueError("rise-field grid misses the profile; reduce grid spacing or enlarge the profile")
    if n > 250_000:
        suggested = h * math.sqrt(n / 250_000) * 1.05
        raise ValueError(f"rise-field solve would contain {n:,} cells; increase grid spacing "
                         f"to about {suggested:.3g} mm or larger")

    # which side of the slit each cell is on, and whether a link crosses the slit segment
    def crosses_slit(pa, pb):
        # segment pa-pb crosses segment C0-C1 ?
        def orient(a, b, c):
            return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
        o1 = orient(C0, C1, pa); o2 = orient(C0, C1, pb)
        o3 = orient(pa, pb, C0); o4 = orient(pa, pb, C1)
        return (o1 * o2 < 0) & (o3 * o4 < 0)

    rows, cols, vals = [], [], []
    diag = np.zeros(n)
    dir_val = np.full(n, np.nan)
    # Dirichlet on the outer boundary cells: f = s/L of the nearest outer point (only if the outer
    # contour is the nearest boundary)
    kd_holes = [cKDTree(densify(hh, h)) for hh in holes]
    cell_pts = cells[mask.ravel()]
    # boundary cells = inside cells with an outside 4-neighbour
    padded = np.pad(mask, 1, constant_values=False)
    bnd = mask & ~(padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
    bnd_ids = idx[bnd]
    bnd_pts = cells[bnd.ravel()]
    # distance to outer vs holes
    kd_outer = cKDTree(densify(outer, h))
    d_o, _ = kd_outer.query(bnd_pts)
    d_h = np.full(len(bnd_pts), np.inf)
    for kd in kd_holes:
        dh, _ = kd.query(bnd_pts)
        d_h = np.minimum(d_h, dh)
    is_outer_bnd = d_o <= d_h
    s_b, _ = nearest_s(bnd_pts[is_outer_bnd], Ps)
    dir_val[bnd_ids[is_outer_bnd]] = s_b / L
    # links
    for (di, dj) in ((0, 1), (1, 0)):
        A = mask[:ny - di, :nx - dj] & mask[di:, dj:]
        ia = idx[:ny - di, :nx - dj][A]
        ib = idx[di:, dj:][A]
        cut = crosses_slit(cell_pts[ia], cell_pts[ib])
        # slit faces: Dirichlet 0 on the +T0 side (s just past the seam), 1 on the other side
        for a_, b_ in ((ia[cut], ib[cut]), (ib[cut], ia[cut])):
            side = (cell_pts[a_] - C0) @ T0
            dir_val[a_] = np.where(side > 0, 0.0, 1.0)
        keep = ~cut
        ia, ib = ia[keep], ib[keep]
        rows += [ia, ib]; cols += [ib, ia]; vals += [np.ones(len(ia)), np.ones(len(ia))]
        np.add.at(diag, ia, 1.0); np.add.at(diag, ib, 1.0)
    rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
    Lap = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    Lap = sparse.diags(diag) - Lap                       # Neumann Laplacian
    isdir = ~np.isnan(dir_val)
    A = Lap.tolil()
    b = np.zeros(n)
    for k in np.nonzero(isdir)[0]:
        A.rows[k] = [k]; A.data[k] = [1.0]; b[k] = dir_val[k]
    f = spsolve(A.tocsr(), b)
    if not np.isfinite(f).all():
        raise ValueError("rise-field solve failed; reduce grid spacing to resolve thin or disconnected strokes")
    F = np.full((ny, nx), np.nan)
    F[mask] = f
    # continuity check away from the slit
    jumps = []
    for (di, dj) in ((0, 1), (1, 0)):
        A2 = mask[:ny - di, :nx - dj] & mask[di:, dj:]
        ia = idx[:ny - di, :nx - dj][A2]; ib = idx[di:, dj:][A2]
        cut = crosses_slit(cell_pts[ia], cell_pts[ib])
        jumps.append(np.abs(f[ia[~cut]] - f[ib[~cut]]))
    max_jump = float(np.concatenate(jumps).max(initial=0.0))
    kd_cells = cKDTree(cell_pts)

    def sample(pts, side=0.0):
        """Bilinear f at 2D points; `side` = +1/-1 nudges the sample across the slit line."""
        q = pts + side * h * T0
        gx = (q[:, 0] - x0) / h - 0.5
        gy = (q[:, 1] - y0) / h - 0.5
        i0 = np.clip(np.floor(gx).astype(int), 0, nx - 2); j0 = np.clip(np.floor(gy).astype(int), 0, ny - 2)
        tx = np.clip(gx - i0, 0, 1); ty = np.clip(gy - j0, 0, 1)
        out = np.zeros(len(pts)); wsum = np.zeros(len(pts))
        for (dj, di, w) in ((0, 0, (1 - tx) * (1 - ty)), (0, 1, tx * (1 - ty)), (1, 0, (1 - tx) * ty), (1, 1, tx * ty)):
            v = F[j0 + dj, i0 + di]
            ok = ~np.isnan(v)
            out[ok] += w[ok] * v[ok]; wsum[ok] += w[ok]
        good = wsum > 1e-9
        out[good] /= wsum[good]
        if (~good).any():
            _, nn = kd_cells.query(q[~good])
            out[~good] = f[nn]
        return out

    return dict(sample=sample, C0=C0, C1=C1, T0=T0, N0=N0, L=L, Ps=Ps, max_jump=max_jump,
                slit_len=slit_len, hit_hole=hit_hole, F=F, xs=xs, ys=ys)


def coil_full_region(outer, holes, thickness, pitch, N, h, left=False, base=0.0, top="fill", pitches=None,
                     edge="none", edge_size=0.0, edge_step=0.2):
    """The whole profile (outer minus holes) becomes the ribbon; rise = harmonic field."""
    validated = validate_regions([{"outer": outer, "holes": holes}])[0]
    outer, holes = validated["outer"], validated["holes"]
    if not holes:
        raise ValueError("Full profile needs an interior opening for a continuous spring. "
                         "Choose Band inward for a solid outline, or add an interior opening.")
    _, C0s, T0s = pick_seam(outer)                     # seam on the longest straight run of the real outline
    outer_d = densify(outer, 1.5)
    holes_d = [densify(hh, 1.5) for hh in holes]
    rise = harmonic_rise(outer_d, holes_d, h, seam_i=(C0s, T0s))
    if not rise["hit_hole"]:
        raise ValueError("The Full profile seam does not reach an interior opening, so it would "
                         "leave a solid connection between laps. Choose Band inward, or adjust "
                         "the profile so an interior opening lies behind its longest edge.")
    C0, T0, N0 = rise["C0"], rise["T0"], rise["N0"]
    region = m3d.CrossSection([outer_d] + holes_d)     # holes are CW -> subtracted
    slit_a, slit_b = 0.0, rise["slit_len"]

    def fsample(P2, side):
        # vertices on the cut line inside the slit span need the value from their own side
        along = (P2 - C0) @ N0
        across = (P2 - C0) @ T0
        on_slit = (np.abs(across) < 1e-4) & (along > slit_a - 1e-4) & (along < slit_b + 1e-4)
        fv = rise["sample"](P2, 0.0)
        if on_slit.any():
            # Both sides must differ by exactly one lap. Grid interpolation just
            # beside the slit only approximates these boundary values and leaves a
            # small step where one lap joins the next.
            fv[on_slit] = 0.0 if side > 0 else 1.0
        return 1.0 - fv if left else fv

    big = 10 * (np.abs(outer).max() + 1)
    man = stack_laps(region, C0, T0, N0, big, fsample, thickness, pitch, N, base, top, pitches,
                     edge, edge_size, edge_step)
    return man, rise


# ----------------------------------------------------------------------------- DXF import
DXF_UNIT_FACTORS = {1: 25.4, 2: 304.8, 4: 1.0, 5: 10.0, 6: 1000.0}     # $INSUNITS -> mm


def _dxf_entities(space):
    """Yield entities, expanding INSERT blocks recursively."""
    for e in space:
        if e.dxftype() == "INSERT":
            try:
                yield from _dxf_entities(list(e.virtual_entities()))
            except Exception:
                continue
        else:
            yield e


def chain_segments(open_segs, join_tol):
    """Join open polylines end-to-end into closed loops (greedy, endpoint distance <= join_tol)."""
    segs = [np.asarray(s, float) for s in open_segs if len(s) >= 2]
    loops, dropped = [], 0
    used = [False] * len(segs)
    for i in range(len(segs)):
        if used[i]:
            continue
        used[i] = True
        chain = segs[i].copy()
        while True:
            if len(chain) > 2 and np.linalg.norm(chain[-1] - chain[0]) <= join_tol:
                loops.append(chain[:-1]); break
            best, best_d, best_rev = None, join_tol, False
            for j in range(len(segs)):
                if used[j]:
                    continue
                d0 = np.linalg.norm(segs[j][0] - chain[-1]); d1 = np.linalg.norm(segs[j][-1] - chain[-1])
                if d0 <= best_d:
                    best, best_d, best_rev = j, d0, False
                if d1 < best_d:
                    best, best_d, best_rev = j, d1, True
            if best is None:
                dropped += 1; break                       # open chain: not a loop
            used[best] = True
            nxt = segs[best][::-1] if best_rev else segs[best]
            chain = np.vstack([chain, nxt[1:]])
    return loops, dropped


def regions_from_loops(loops, min_area=1e-6):
    """Classify closed loops into outers (even nesting depth) and holes (odd) -> regions list."""
    from matplotlib.path import Path
    loops = [dedupe(L) for L in loops]
    loops = [L for L in loops if len(L) >= 3 and abs(signed_area(L)) > min_area]
    paths = [Path(L) for L in loops]
    areas = [abs(signed_area(L)) for L in loops]
    n = len(loops)
    contains = [[False] * n for _ in range(n)]
    for a in range(n):
        for b in range(n):
            if a != b and areas[a] > areas[b]:
                contains[a][b] = bool(paths[a].contains_point(loops[b][0]))
    depth = [sum(contains[a][b] for a in range(n)) for b in range(n)]
    regions = []
    parent_of = {}
    for b in range(n):
        if depth[b] % 2 == 1:                            # hole: parent = smallest containing loop
            cands = [a for a in range(n) if contains[a][b] and depth[a] == depth[b] - 1]
            parent_of[b] = min(cands, key=lambda a: areas[a]) if cands else None
    for a in range(n):
        if depth[a] % 2 == 0:
            o = loops[a] if signed_area(loops[a]) > 0 else loops[a][::-1].copy()
            hs = []
            for b, pa in parent_of.items():
                if pa == a:
                    h = loops[b] if signed_area(loops[b]) < 0 else loops[b][::-1].copy()
                    hs.append(h)
            regions.append({"outer": o, "holes": hs})
    return regions


def loops_from_dxf(path, units="auto", tol=0.02, join_tol=0.05):
    """Closed regions (outer + holes) from a DXF: LINE/ARC/CIRCLE/ELLIPSE/SPLINE/(LW)POLYLINE and
    blocks are flattened with ezdxf, open pieces are chained into loops, nesting gives the holes.
    units: 'auto' (from $INSUNITS), 'mm' or 'in'.  Result is in mm."""
    import ezdxf
    from ezdxf import path as ezpath
    doc = ezdxf.readfile(path)
    ins = int(doc.header.get("$INSUNITS", 0))
    if units == "auto":
        factor = DXF_UNIT_FACTORS.get(ins, 1.0)
        unit_name = {1: "in", 2: "ft", 4: "mm", 5: "cm", 6: "m"}.get(ins, "unitless -> assumed mm")
    else:
        factor = 25.4 if units == "in" else 1.0
        unit_name = units + " (forced)"
    closed, opens, skipped = [], [], 0
    for e in _dxf_entities(doc.modelspace()):
        try:
            p = ezpath.make_path(e)
        except Exception:
            skipped += 1
            continue
        for sp in p.sub_paths():
            pts = np.array([(v.x, v.y) for v in sp.flattening(tol / factor)], float) * factor
            if len(pts) < 2:
                continue
            if sp.is_closed or np.linalg.norm(pts[0] - pts[-1]) <= join_tol:
                closed.append(pts if np.linalg.norm(pts[0] - pts[-1]) > 1e-9 else pts[:-1])
            else:
                opens.append(pts)
    loops, dropped = chain_segments(opens, join_tol)
    loops = closed + loops
    regions = regions_from_loops(loops)
    print(f"{path}: units {unit_name}, {len(closed)} closed + {len(opens)} open entities -> "
          f"{len(loops)} loops, {len(regions)} region(s), {sum(len(r['holes']) for r in regions)} hole(s)"
          + (f", {dropped} open chain(s) ignored" if dropped else "") + (f", {skipped} entities skipped" if skipped else ""))
    if not regions:
        sys.exit("no closed regions found in the DXF")
    return regions


# ----------------------------------------------------------------------------- build
def validate_build_args(args):
    """Reject invalid numeric settings before geometry work or file writes."""
    positive = ("scale", "width", "thickness", "layer")
    nonnegative = ("tol", "gap", "base", "min_hole_area", "edge_size", "edge_step", "gap_ladder")
    for name in positive + nonnegative:
        value = getattr(args, name, 0.0)
        if not isinstance(value, (int, float, np.number)) or not math.isfinite(value):
            raise ValueError(f"{name.replace('_', ' ')} must be a finite number")
        if (name in positive and value <= 0) or (name in nonnegative and value < 0):
            requirement = "positive" if name in positive else "nonnegative"
            raise ValueError(f"{name.replace('_', ' ')} must be {requirement}")
    if not isinstance(args.turns, (int, np.integer)) or isinstance(args.turns, bool) or args.turns < 1:
        raise ValueError("turns must be a positive whole number")
    if args.turns > 1000:
        raise ValueError("turns must be 1000 or fewer; reduce the target height or lap count")
    if args.band not in ("inward", "centered", "outward", "full"):
        raise ValueError("invalid band mode")
    if args.top not in ("fill", "open"):
        raise ValueError("invalid top mode")
    if getattr(args, "edge", "none") not in (None, "none", "chamfer", "round"):
        raise ValueError("invalid edge mode")
    method = getattr(args, "method", "helix")
    if method not in ("helix", "staggered"):
        raise ValueError("invalid construction method")
    if method == "staggered":
        advance = getattr(args, "bridge_advance", 25.0)
        if not math.isfinite(advance) or not 0 < advance < 100:
            raise ValueError("Bridge advance must be greater than 0 and less than 100 percent.")
        for key in ("bridge_length", "bridge_depth"):
            value = getattr(args, key, 6.0 if key == "bridge_length" else 2.0)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Bridge length and depth must be positive finite numbers.")
    if method == "helix" and args.band == "full" and (not math.isfinite(args.grid) or args.grid <= 0):
        raise ValueError("rise-field grid spacing must be positive")
    if not (getattr(args, "regions", None) or args.step or getattr(args, "dxf", "") or args.loops):
        if not math.isfinite(args.circle) or args.circle <= 0:
            raise ValueError("circle diameter must be positive")


def build(args):
    validate_build_args(args)
    if getattr(args, "regions", None) is not None:         # pre-loaded regions (the app passes these)
        regions = args.regions
    elif args.step:
        regions = loops_from_step(args.step)
    elif getattr(args, "dxf", ""):
        regions = loops_from_dxf(args.dxf, getattr(args, "units", "auto"))
    elif args.loops:
        with open(args.loops, encoding="utf-8-sig") as source:
            regions = json.load(source)
        if isinstance(regions, dict) and "regions" in regions:
            regions = regions["regions"]                    # app project files contain the same loops
    else:
        n = 360
        a = np.linspace(0, TAU, n, endpoint=False)
        r = args.circle / 2
        regions = [{"outer": np.column_stack([r * np.cos(a), r * np.sin(a)]), "holes": []}]
    regions = validate_regions(regions)
    if args.save_loops and (args.step or getattr(args, "dxf", "")):
        with open(args.save_loops, "w", encoding="utf-8") as target:
            json.dump([{"outer": r["outer"].tolist(), "holes": [h.tolist() for h in r["holes"]]}
                       for r in regions], target)
        print("wrote", args.save_loops)

    allpts = np.vstack([r["outer"] for r in regions])
    ctr = (allpts.min(0) + allpts.max(0)) / 2
    s = args.scale

    def prep(P):
        Q = (np.asarray(P, float) - ctr) * s
        Q = dedupe(rdp(Q, args.tol))
        if len(Q) < 3 or abs(signed_area(Q)) <= 1e-12:
            raise ValueError("outline simplification removed a closed loop; reduce outline tolerance")
        return Q

    T, W, g = args.thickness, args.width, args.gap
    pitch = T + g
    N = args.turns
    edge = getattr(args, "edge", "none") or "none"
    e_size = float(getattr(args, "edge_size", 0.0) or 0.0)
    e_step = float(getattr(args, "edge_step", 0.0) or 0.0) or float(args.layer) / 2
    if e_size <= 0:
        edge = "none"
    if edge == "none":
        e_size = 0.0
    if e_size > 0.4 * T:                              # keep at least 20 % of the sheet at full width
        print(f"note             : taper {e_size:.2f} mm clamped to {0.4*T:.2f} mm "
              f"(40 % of the {T:.2f} mm sheet)")
        e_size = 0.4 * T
    ladder = float(getattr(args, "gap_ladder", 0.0) or 0.0)
    pitches = [T + g + k * ladder for k in range(N)] if ladder > 0 else None   # gap-ladder coupon
    coils, lengths = [], []
    rise_info = None
    method = getattr(args, "method", "helix")
    bridge_sets = []
    preview_coils, preview_bridge_ids, preview_ramps = [], {}, []

    def collect_bridges(details):
        bridge_sets.append(details["bridges"])
        if details["preview_manifold"] is not None:
            preview_coils.append(details["preview_manifold"])
            preview_bridge_ids.update(details["preview_bridge_ids"])
            preview_ramps.extend(details["preview_ramps"])

    bridge_options = dict(advance=getattr(args, "bridge_advance", 25.0),
                          bridge_length=getattr(args, "bridge_length", 6.0),
                          bridge_depth=getattr(args, "bridge_depth", 2.0),
                          capture_preview=getattr(args, "capture_preview", False))
    for r in regions:
        P = prep(r["outer"])
        if signed_area(P) < 0:
            P = P[::-1].copy()
        if args.band == "full":
            hs = [prep(hh) for hh in r["holes"] if abs(signed_area(hh)) * s * s >= args.min_hole_area]
            if method == "staggered":
                section = m3d.CrossSection([P] + hs)
                man, details = staggered_laps(section, P, T, pitch, N, args.base, args.top, pitches,
                                               edge, e_size, e_step, args.left, **bridge_options)
                coils.append(man); lengths.append(details["L"]); collect_bridges(details)
                continue
            man, rise_info = coil_full_region(P, hs, T, pitch, N, args.grid, args.left, args.base, args.top,
                                              pitches, edge, e_size, e_step)
            coils.append(man); lengths.append(rise_info["L"])
            print(f"rise field       : grid {args.grid} mm, slit {rise_info['slit_len']:.2f} mm from the seam "
                  f"{'into a hole' if rise_info['hit_hole'] else 'to a pole (no hole behind the seam)'}, "
                  f"largest cell-to-cell jump off the slit {rise_info['max_jump']*pitch:.4f} mm")
            continue
        if method == "staggered":
            man, details = staggered_from_contour(P, W, T, pitch, N, args.band, args.left,
                base=args.base, top=args.top, pitches=pitches, edge=edge, edge_size=e_size,
                edge_step=e_step, **bridge_options)
            L = details["L"]
            collect_bridges(details)
        else:
            man, L = coil_from_contour(P, W, T, pitch, N, args.band, args.left, base=args.base, top=args.top,
                                       pitches=pitches, edge=edge, edge_size=e_size, edge_step=e_step)
        coils.append(man); lengths.append(L)
        if args.holes:
            for h in r["holes"]:
                if abs(signed_area(h)) * s * s < args.min_hole_area:
                    continue
                Q = prep(h)
                if signed_area(Q) < 0:
                    Q = Q[::-1].copy()
                hb = {"inward": "outward", "outward": "inward", "centered": "centered"}[args.band]
                if method == "staggered":
                    mh, details = staggered_from_contour(Q, W, T, pitch, N, hb, args.left,
                        base=args.base, top=args.top, pitches=pitches, edge=edge, edge_size=e_size,
                        edge_step=e_step, **bridge_options)
                    Lh = details["L"]
                    collect_bridges(details)
                else:
                    mh, Lh = coil_from_contour(Q, W, T, pitch, N, hb, args.left, base=args.base, top=args.top,
                                               pitches=pitches, edge=edge, edge_size=e_size, edge_step=e_step)
                coils.append(mh); lengths.append(Lh)
    for a in range(len(coils)):                      # nested coils must not touch each other
        for b in range(a + 1, len(coils)):
            bb_a, bb_b = coils[a].bounding_box(), coils[b].bounding_box()
            if any(bb_a[d + 3] < bb_b[d] or bb_b[d + 3] < bb_a[d] for d in range(3)):
                continue                            # disjoint components need no expensive intersection
            vol = (coils[a] ^ coils[b]).volume()
            if vol > 1e-6:
                print(f"WARNING: coil {a} and coil {b} overlap ({vol:.2f} mm^3) -- they will fuse; "
                      f"reduce --width below half the narrowest stroke or drop --holes", file=sys.stderr)
    man = m3d.Manifold.batch_boolean(coils, m3d.OpType.Add)
    if man.status() != m3d.Error.NoError or man.is_empty():
        raise ValueError(f"build failed: {man.status()}")
    bb = man.bounding_box()
    P0 = prep(regions[0]["outer"])
    if signed_area(P0) < 0:
        P0 = P0[::-1].copy()
    preview = (preview_coils[0] if len(preview_coils) == 1 else
               m3d.Manifold.batch_boolean(preview_coils, m3d.OpType.Add) if preview_coils else None)
    return man, dict(pitch=pitch, gap=g, W=W, T=T, N=N, H=bb[5], L=lengths, bb=bb, path=P0, pitches=pitches,
                     footprint=(bb[3] - bb[0], bb[4] - bb[1]), ncoils=len(coils), rise=rise_info,
                     edge=edge, edge_size=e_size, edge_step=e_step, method=method, bridge_sets=bridge_sets,
                     preview_manifold=preview, preview_bridge_ids=preview_bridge_ids, preview_ramps=preview_ramps)


def lap_compliance(P, W, T, G, offset=0.0):
    """Axial deflection per unit force for one lap of a flat strip coiled along closed path P:
    integral of d^2/(G J) (torsion, lever arm d = distance from the centroid to the local tangent line)
    plus (r.T)^2/(E I) (weak-axis bending) along the path.  Circle -> 2 pi R^3/(G J)."""
    seg, ln, _, L = polyline_lengths(P)
    mid = P + 0.5 * seg
    c = (mid * ln[:, None]).sum(0) / L
    Tv = seg / ln[:, None]
    r = mid - c
    d = np.abs(r[:, 0] * Tv[:, 1] - r[:, 1] * Tv[:, 0])           # |r x T|
    rt = (r * Tv).sum(1)                                            # r . T
    tt = min(W, T) / max(W, T)
    J = max(W, T) * min(W, T) ** 3 * (1 / 3 - 0.21 * tt * (1 - tt ** 4 / 12))
    I = W * T ** 3 / 12
    E = 2.6 * G
    return float(np.sum((d ** 2 / (G * J) + rt ** 2 / (E * I)) * ln))   # mm/N per lap


def report(man, info, args):
    p, g, W, T, N, H = (info[k] for k in ("pitch", "gap", "W", "T", "N", "H"))
    G, rho = MATERIALS[args.material]
    print(f"coil path        : {info['ncoils']} coil(s); lap length(s) {', '.join(f'{L:.1f}' for L in info['L'])} mm; "
          f"footprint {info['footprint'][0]:.1f} x {info['footprint'][1]:.1f} mm, scale {args.scale}")
    print(f"strip            : {'full profile' if args.band == 'full' else f'{W:.2f} wide ({args.band})'} x {T:.2f} thick")
    staggered = info.get("method", "helix") == "staggered"
    if staggered:
        print(f"method           : staggered perimeter bridges; {getattr(args, 'bridge_advance', 25.0):g}% perimeter advance per gap; "
              f"{getattr(args, 'bridge_length', 6.0):g} mm ramp length, {getattr(args, 'bridge_depth', 2.0):g} mm depth")
        length = getattr(args, "bridge_length", 6.0)
        last_pitch = T + g + max(0, N - 2) * float(getattr(args, "gap_ladder", 0.0) or 0.0)
        angle = math.degrees(math.atan2(p, length))
        last_angle = math.degrees(math.atan2(last_pitch, length))
        print(f"ramp grade       : rise = sheet + gap; {p:.2f} mm / {length:g} mm = {angle:.1f} degrees from horizontal"
              + (f"; last ramp {last_angle:.1f} degrees" if N > 2 and last_pitch > p else "")
              + "; length is perimeter arc length and does not scale with the profile")
        print("construction     : linked flat laps, not a continuous helical strand; the gap is intentionally "
              "closed only at each local ramp")
        print("print caution    : experimental; flat laps have larger unsupported areas. Review the layer check "
              "and slicer preview, then print a short coupon before a tall model.")
    else:
        print("method           : continuous helix")
    if info.get("edge", "none") != "none":
        e = info["edge_size"]
        print(f"edge             : {info['edge']} {e:.2f} mm top and bottom (staircase of {info['edge_step']:.2f} mm); "
              f"the middle {T - 2*e:.2f} mm of the sheet keeps full width and faces the next lap across {g:.2f} mm; "
              f"the outer {e:.2f} mm of every stroke pulls back, so its clearance opens to {g + 2*e:.2f} mm")
    if info.get("pitches") and (not staggered or N > 1):
        gaps = [pk - T for pk in info["pitches"]]
        if staggered:
            gaps = gaps[:-1]
        print("gap ladder       : gap above lap k " + ("is " if staggered else "ramps ") + ", ".join(f"{gk:.2f}" for gk in gaps) + " mm "
              f"({gaps[0]/args.layer:.1f} to {gaps[-1]/args.layer:.1f} layers @ {args.layer} mm); "
              "note which lap frees up cleanly")
    print(f"pitch / gap      : {p:.3f} / {g:.3f} mm   (gap = {g/args.layer:.2f} layers @ {args.layer} mm"
          + ("" if abs(g / args.layer - round(g / args.layer)) < 0.02 else
             "  <-- not a whole number of layers: clearance will vary along the coil") + ")")
    print(f"turns / height   : {N} laps; first lap filled to the bed"
          + (f" on a {args.base:.2f} mm plate" if args.base > 0 else "")
          + (", last lap filled to a flat top" if args.top == "fill" else ", open top")
          + f"; height {H:.2f} mm")
    L0 = info["L"][0]
    band = args.layer / p * L0
    if not staggered:
        print(f"per-layer bridge : leading {band:.1f} mm of path ({100*args.layer/p:.1f} % of a lap) in every layer is "
              f"laid over {g:.2f} mm of air onto the lap below")
    print(f"mesh             : {man.num_tri()} triangles, watertight ({man.status()}), volume {man.volume()/1000:.2f} cm^3")
    mass = man.volume() * rho * 1e-3
    if staggered:
        print(f"{args.material} estimate     : mass {mass:.1f} g; helix stiffness/hanging estimates do not apply to linked flat laps")
    elif args.band == "full":                                        # effective strip width = area / lap length
        W = man.volume() / (N + 2 * T / p) / T / info["L"][0]
    if not staggered:
        k_turn = 1.0 / lap_compliance(info["path"], W, T, G)          # outer coil only
        m_turn = mass / (N + 2 * T / p)
        wt = m_turn * 9.81e-3
        hang = wt * N * (N - 1) / (2 * k_turn)
        print(f"{args.material} estimate     : mass {mass:.1f} g; outer coil ~{k_turn*1000:.1f} N/m per lap "
              f"({k_turn*1000/N:.2f} N/m whole), hangs ~{hang:.0f} mm from one end  [torsion+bending of the strip, rough]")
    if not args.no_layer_check and info.get("pitches"):
        print("slice check      : skipped for gap ladder; variable-pitch layer checks are not supported")
    elif not args.no_layer_check:
        rows = layer_check(man, args.layer, H, g, 2 * info.get("edge_size", 0.0))
        if not rows:
            print("slice check      : no bridged layers to measure at this layer height")
            return
        ua = np.array([r[2] for r in rows]); land = np.array([r[3] for r in rows])
        deep = [r[4] for r in rows if r[4] is not None]
        e2 = info.get("edge_size", 0.0) if info.get("edge", "none") != "none" else 0.0
        span = f"{g:.2f}" if e2 <= 0 else f"{g:.2f}-{g + 2*e2:.2f}"
        print(f"slice check      : {len(rows)} layers; bridged area per layer {ua.min():.0f}..{ua.max():.0f} mm^2; "
              f"{100*land.min():.0f}..{100*land.max():.0f} % of it lands on solid {span} mm below"
              + (f"; deepest air under any bridged region {max(deep)*args.layer:.2f} mm" if deep else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", default="", help="STEP of a straight extrusion (needs cadquery)")
    ap.add_argument("--dxf", default="", help="DXF of the profile (closed loops; needs ezdxf)")
    ap.add_argument("--units", choices=["auto", "mm", "in"], default="auto", help="DXF units if the file doesn't say")
    ap.add_argument("--loops", default="", help="loops JSON written by slinky.py/outline_slinky.py --save-loops")
    ap.add_argument("--circle", type=float, default=60.0, help="no file: use a circle of this diameter [mm]")
    ap.add_argument("--save-loops", default="")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=0.02, help="outline simplification tolerance after scaling [mm]")
    ap.add_argument("--width", type=float, default=3.0, help="strip width in the plane [mm]")
    ap.add_argument("--method", choices=["helix", "staggered"], default="helix",
                    help="continuous helix, or experimental flat laps with staggered perimeter ramps")
    ap.add_argument("--bridge-advance", type=float, default=25.0, help="staggered: perimeter advance per gap [%], 0 < value < 100")
    ap.add_argument("--bridge-length", type=float, default=6.0, help="staggered: ramp length along the perimeter [mm]")
    ap.add_argument("--bridge-depth", type=float, default=2.0, help="staggered: ramp depth into the local material [mm]")
    ap.add_argument("--thickness", type=float, default=1.0, help="strip thickness (axial) [mm]")
    ap.add_argument("--band", choices=["inward", "centered", "outward", "full"], default="inward",
                    help="where the strip sits relative to the outline; 'full' = the whole profile "
                         "(holes included) is the ribbon, rise = harmonic field")
    ap.add_argument("--grid", type=float, default=0.15, help="--band full: cell size of the rise-field solve [mm]")
    ap.add_argument("--gap", type=float, default=0.4)
    ap.add_argument("--turns", type=int, default=40)
    ap.add_argument("--edge", choices=["none", "chamfer", "round"], default="none",
                    help="taper the top and bottom edges of every lap so coils meet on a narrow ridge")
    ap.add_argument("--edge-size", type=float, default=0.3, help="chamfer/fillet size [mm]")
    ap.add_argument("--edge-step", type=float, default=0.0, help="taper staircase step [mm] (default: half the layer height)")
    ap.add_argument("--gap-ladder", type=float, default=0.0,
                    help="test coupon: gap grows by this much every lap (e.g. 0.2 -> 0.4, 0.6, 0.8 ...)")
    ap.add_argument("--base", type=float, default=0.0,
                    help="extra flat plate under the first lap [mm] (the first lap is always filled to the bed)")
    ap.add_argument("--top", choices=["fill", "open"], default="fill",
                    help="fill the last lap up to a flat top (default) or leave it a plain lap")
    ap.add_argument("--left", action="store_true", help="wind the other way")
    ap.add_argument("--holes", action="store_true", help="also coil the hole contours (letter counters)")
    ap.add_argument("--min-hole-area", type=float, default=2.0)
    ap.add_argument("--layer", type=float, default=0.2)
    ap.add_argument("--material", choices=MATERIALS, default="PLA")
    ap.add_argument("--no-layer-check", action="store_true")
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--out", default="outline_slinky.stl")
    args = ap.parse_args()

    try:
        man, info = build(args)
    except (ValueError, OverflowError) as exc:
        ap.error(str(exc))
    V, F = from_manifold(man)
    write_stl(args.out, V, F)
    print(f"wrote {args.out}  ({len(F)} triangles)")
    report(man, info, args)
    if args.png:
        base = args.out[:-4] if args.out.lower().endswith(".stl") else args.out
        print("wrote", render(man, base + "_iso.png", elev=30, azim=-30))
        print("wrote", render(man, base + "_top.png", elev=90, azim=0))
        print("wrote", render(man, base + "_front.png", elev=8, azim=0))


if __name__ == "__main__":
    main()
