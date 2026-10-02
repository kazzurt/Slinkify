#!/usr/bin/env python3
"""
Slinkify -- local web UI for outline_slinky.py.

    python slinky_app.py            (opens http://127.0.0.1:7860 in your browser)
    python slinky_app.py --port 8000 --no-browser

Import a DXF / STEP / loops-JSON profile, set the size and height (inches or mm), laps, sheet
thickness and print-in-place gap, pick the strip mode, generate a preview, then save the reviewed
STL and params JSON separately. Runs entirely on this machine.
"""
import argparse
import atexit
import contextlib
import errno
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import types
import uuid

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

import numpy as np
import gradio as gr

import outline_slinky as osl
from parameter_help import attach_parameter_help
from preview_colors import preview_legend, write_bridge_preview
from live_preview import preview_component, preview_payload, write_live_preview
from ui_styles import APP_CSS, APP_THEME
from slinky import signed_area, write_stl, from_manifold, MATERIALS, validate_regions, MAX_LAYER_SAMPLES

DEFAULT_OUT = os.path.dirname(APP_DIR) if os.path.basename(APP_DIR).lower() == "slinky_app" else APP_DIR
DEFAULT_PROFILE = os.path.join(APP_DIR, "DDR_profile.dxf")     # loaded at start-up when it is there
IN = 25.4
MAX_LAPS = 1000
METHODS = {"Continuous helix": "helix", "Staggered perimeter bridges": "staggered"}
STAGGERED_NOTE = ("**Experimental bridges:** flat laps have large unsupported spans. "
                  "Test a short coupon before a full print.")
# stdout/stderr redirection and matplotlib are process-global. Serialize geometry jobs,
# including calls made outside Gradio, while keeping the inexpensive live readout free.
GEOMETRY_LOCK = threading.RLock()
DRAFT_CACHES = set()


# ----------------------------------------------------------------------------- profile loading
def load_profile(file, units):
    """-> (state dict, info markdown, preview PNG path)"""
    try:
        with GEOMETRY_LOCK:
            return _load_profile(file, units)
    except (Exception, SystemExit) as exc:
        # Clear the previous shape on failure: Generate must never silently use it.
        return None, f"**Could not load this profile.** {exc}", None


def _load_profile(file, units):
    if file is None:
        return None, "Pick a DXF, STEP or loops JSON first.", None
    path = file if isinstance(file, str) else file.name
    ext = os.path.splitext(path)[1].lower()
    buf = io.StringIO()
    params = None
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        if ext == ".dxf":
            regions = osl.loops_from_dxf(path, units)
        elif ext in (".step", ".stp"):
            regions = osl.loops_from_step(path)
        elif ext == ".json":
            with open(path, encoding="utf-8-sig") as source:
                data = json.load(source)
            if isinstance(data, dict) and "regions" in data:          # a .slinky.json written by this app
                params = data.get("params")
                if params is not None and not isinstance(params, dict):
                    raise ValueError("Saved project settings must be a JSON object.")
                data = data["regions"]
            regions = data
        else:
            return None, f"Unsupported file type {ext}", None
    regions = validate_regions(regions)
    allpts = np.vstack([r["outer"] for r in regions])
    lo, hi = allpts.min(0), allpts.max(0)
    w, h = hi - lo
    strokes = [min_stroke(r) for r in regions]
    state = {
        "name": os.path.splitext(os.path.basename(path))[0].removesuffix(".slinky"),
        "file_type": ext,
        "params": params,
        "regions": [{"outer": r["outer"].tolist(), "holes": [hh.tolist() for hh in r["holes"]]} for r in regions],
        "width_mm": float(w), "height_mm": float(h),
        "area_mm2": float(sum(abs(signed_area(r["outer"])) - sum(abs(signed_area(hh)) for hh in r["holes"]) for r in regions)),
        "min_stroke_mm": float(min(strokes)) if strokes else 0.0,
    }
    nholes = sum(len(r["holes"]) for r in regions)
    info = (f"**{os.path.basename(path)}** — {len(regions)} region(s), {nholes} hole(s); "
            f"{w:.2f} × {h:.2f} mm ({w/IN:.3f} × {h/IN:.3f} in) at 1:1, solid area {state['area_mm2']:.1f} mm², "
            f"narrowest stroke ≈ {state['min_stroke_mm']:.2f} mm")
    import_log = buf.getvalue().strip().replace(path, os.path.basename(path))
    if import_log:
        info += "\n\n`" + import_log + "`"
    if params is not None:
        info += "\n\nSaved print settings restored. The output folder stays unchanged."
    png = profile_png(regions, os.path.join(gr_cache_dir(), f"profile_{uuid.uuid4().hex}.png"))
    return state, info, png


def min_stroke(region):
    """Narrowest significant stroke of a region: 2 x the smallest radius at which a morphological
    opening (inward then outward offset) removes more than 1.5 % of the area.  Bisection; rough."""
    import manifold3d as m3d
    cs = m3d.CrossSection([np.asarray(region["outer"])] + [np.asarray(h) for h in region["holes"]])
    A = cs.area()
    lo, hi = 0.0, 0.5 * max(np.ptp(np.asarray(region["outer"]), axis=0))
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        opened = cs.offset(-mid, m3d.JoinType.Round).offset(mid, m3d.JoinType.Round)
        if opened.area() < 0.985 * A:
            hi = mid
        else:
            lo = mid
    return 2 * hi


def gr_cache_dir():
    d = os.path.join(APP_DIR, ".slinky_cache")
    os.makedirs(d, exist_ok=True)
    return d


def same_file_contents(a, b):
    if os.path.getsize(a) != os.path.getsize(b):
        return False
    def digest(path):
        with open(path, "rb") as source:
            return hashlib.file_digest(source, "sha256").digest()
    return digest(a) == digest(b)


def disposable_result(folder):
    """Only remove cache copies after checking their saved originals still match."""
    projects = [entry.path for entry in os.scandir(folder) if entry.is_file() and entry.name.endswith(".slinky.json")]
    if len(projects) != 1:
        return False
    with open(projects[0], encoding="utf-8") as source:
        original = json.load(source)["params"]["out"]
    if not isinstance(original, str):
        return False
    original_project = os.path.splitext(original)[0] + ".slinky.json"
    cached_stl = os.path.join(folder, os.path.basename(original))
    return same_file_contents(projects[0], original_project) and same_file_contents(cached_stl, original)


def process_is_running(pid):
    """Conservatively identify abandoned preview caches without touching a process."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE, read-only wait
        if not handle:
            return ctypes.get_last_error() != 87  # Missing PID; access denied stays protected.
        try:
            return kernel.WaitForSingleObject(handle, 0) != 0
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def purge_cache(max_bytes=192 * 1024 * 1024):
    """Bound duplicate result storage; preserve the two newest results and originals."""
    d = os.path.realpath(gr_cache_dir())
    results = []
    for f in os.scandir(d):
        try:
            if f.is_symlink():
                continue
            target = os.path.realpath(f.path)
            if os.path.dirname(target) != d:
                continue
            if f.is_dir() and f.name.startswith("result-"):
                size = sum(os.path.getsize(os.path.join(root, name))
                           for root, _, names in os.walk(target) for name in names)
                results.append((f.stat().st_mtime, size, target))
            elif f.is_dir() and (owner := re.match(r"draft-(\d+)-", f.name)):
                # Handle interrupted server runs without deleting another live
                # session's draft or anything containing unrecognized files.
                if (target not in DRAFT_CACHES and time.time() - f.stat().st_mtime >= 86400
                        and not process_is_running(int(owner[1]))):
                    entries = list(os.scandir(target))
                    if all(entry.is_file(follow_symlinks=False) and entry.name == "preview.glb" for entry in entries):
                        shutil.rmtree(target)
            elif f.is_file() and time.time() - f.stat().st_mtime >= 7 * 86400:
                os.remove(target)
        except OSError:
            pass
    results.sort(reverse=True)
    total = sum(size for _, size, _ in results)
    for modified, size, target in reversed(results[2:]):
        if total <= max_bytes and time.time() - modified < 7 * 86400:
            continue
        try:
            if disposable_result(target):
                shutil.rmtree(target)
                total -= size
        except (OSError, ValueError, KeyError, TypeError):
            pass


def require_disk_space(directory, needed_bytes):
    """Allow room for saved output, preview and Gradio's download copies."""
    ancestor = os.path.abspath(directory)
    while not os.path.isdir(ancestor):
        parent = os.path.dirname(ancestor)
        if parent == ancestor:
            break
        ancestor = parent
    free = shutil.disk_usage(ancestor).free
    required = needed_bytes + 64 * 1024 * 1024
    if free < required:
        raise gr.Error(f"Not enough free disk space for this action. "
                       f"About {required / (1024 * 1024):.0f} MB is needed; "
                       f"{free / (1024 * 1024):.0f} MB is available. Free some space, then try again.")


def cleanup_draft(draft):
    """Remove only a registered draft directly inside the current cache root."""
    if not isinstance(draft, dict) or not isinstance(draft.get("cache_dir"), str):
        return False
    with GEOMETRY_LOCK:
        path = draft["cache_dir"]
        target, root = os.path.realpath(path), os.path.realpath(gr_cache_dir())
        if (target not in DRAFT_CACHES or os.path.islink(path) or os.path.dirname(target) != root
                or not os.path.basename(target).startswith("draft-")):
            return False
        try:
            shutil.rmtree(target)
        except FileNotFoundError:
            pass
        except OSError:
            return False
        DRAFT_CACHES.discard(target)
        return True


def cleanup_session_drafts():
    for path in list(DRAFT_CACHES):
        cleanup_draft(dict(cache_dir=path))


atexit.register(cleanup_session_drafts)


def reviewed_description(params):
    description = (f"{params['profile']}: {params['turns']} laps, {params['scale'] * 100:g}% XY, "
                   f"{params['thickness']:g} mm sheets, {params['gap']:g} mm gaps")
    if params["method"] == "staggered":
        description += f", staggered bridges with {params['bridge_length']:g} mm ramps"
    else:
        description += ", continuous helix"
    return description


def profile_png(regions, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for r in regions:
        o = np.vstack([r["outer"], r["outer"][:1]])
        ax.fill(o[:, 0], o[:, 1], color="#f0c070"); ax.plot(o[:, 0], o[:, 1], "k-", lw=1)
        for hh in r["holes"]:
            hq = np.vstack([hh, hh[:1]])
            ax.fill(hq[:, 0], hq[:, 1], color="white"); ax.plot(hq[:, 0], hq[:, 1], "k-", lw=1)
    ax.set_aspect("equal"); ax.grid(True, lw=0.3); ax.set_xlabel("mm")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


# ----------------------------------------------------------------------------- derived numbers
def num(v, default=0.0, minimum=None):
    """A Gradio Number sends None while its box is empty (every time you clear one to retype it).
    Coerce to a float so the live readout never blows up mid-keystroke."""
    try:
        x = float(v)
    except (TypeError, ValueError, OverflowError):
        return default
    if x != x or x in (float("inf"), float("-inf")):
        return default
    return x if minimum is None else max(x, minimum)


def derive(state, size_by, size_val, height_by, height_val, thickness, gap, snap_gap, layer, base,
           ladder=False, ladder_step=0.2, method="Continuous helix"):
    """Resolve scale, laps, pitch and height from the UI choices."""
    thickness = num(thickness, D["thickness"])
    gap = num(gap, D["gap"])
    layer = num(layer, D["layer"])
    thickness = thickness if thickness > 0 else D["thickness"]
    gap = gap if gap > 0 else D["gap"]
    layer = layer if layer > 0 else D["layer"]
    base = num(base, 0.0, 0.0)
    ladder_step = num(ladder_step, 0.0, 0.0)
    size_val = num(size_val, 1.0)
    height_val = num(height_val, 1.0)
    if snap_gap and layer > 0:
        gap = max(1, round(gap / layer)) * layer
        if ladder:
            ladder_step = max(1, round(ladder_step / layer)) * layer
    step = ladder_step if ladder else 0.0
    pitch = thickness + gap
    extra_lap = 1 if method == "Staggered perimeter bridges" else 0

    def height_of(n):
        intervals = n - extra_lap
        return base + intervals * pitch + step * intervals * (intervals - 1) / 2 + thickness

    if state is None:
        return dict(scale=1.0, gap=gap, pitch=pitch, laps=max(1, int(round(height_val))) if height_by == "Laps" else 0,
                    H=0.0, step=step)
    if size_by == "Scale factor":
        scale = float(size_val)
    elif size_by == "Width (in)":
        scale = float(size_val) * IN / state["width_mm"]
    else:
        scale = float(size_val) / state["width_mm"]
    if height_by == "Laps":
        laps = max(1, int(round(height_val)))
    else:
        Hmm = float(height_val) * (IN if height_by == "Height (in)" else 1.0)
        laps = max(1, int(round((Hmm - base - thickness) / pitch)) + extra_lap)
        if step > 0:
            # Solve the quadratic once, then choose the closest attainable height.
            target = max(0.0, Hmm - base - thickness)
            b = pitch - step / 2
            root = math.hypot(b, math.sqrt(target) * math.sqrt(2 * step))
            n = 2 * target / (root + b) if b >= 0 and root + b else (root - b) / step
            lower = max(1, math.floor(n) + extra_lap)
            laps = min((lower, lower + 1), key=lambda count: abs(height_of(count) - Hmm))
    H = height_of(laps)
    return dict(scale=scale, gap=gap, pitch=pitch, laps=laps, H=H, step=step)


def readout(*a, **kw):
    """Wrapper: the live readout runs on every keystroke, so it must never raise a red toast."""
    try:
        return _readout(*a, **kw)
    except (ValueError, OverflowError, ZeroDivisionError):
        return "Enter valid size and print settings to see the resulting dimensions."


def bridge_readout(method, thickness, gap, snap_gap, layer, ladder, ladder_step, bridge_length):
    """Live nominal grade, using the same effective gap/snapping as generation."""
    if method != "Staggered perimeter bridges":
        return ""
    thickness, gap, layer, bridge_length = [num(value, -1) for value in (thickness, gap, layer, bridge_length)]
    if any(value <= 0 for value in (thickness, gap, layer, bridge_length)) or not math.isfinite(thickness + gap):
        return "Enter positive sheet, gap, layer height and bridge length values to see the ramp angle."
    d = derive(None, "Scale factor", 1, "Laps", 1, thickness, gap, snap_gap, layer, 0, ladder, ladder_step, method)
    angle = math.degrees(math.atan2(d["pitch"], bridge_length))
    return (f"**Ramp: {angle:.1f}°** · {d['pitch']:.2f} mm rise / {bridge_length:g} mm run"
            + (" Gap ladder makes higher ramps steeper." if d["step"] > 0 else ""))


def _readout(state, size_by, size_val, height_by, height_val, thickness, gap, snap_gap, layer, base,
             ladder=False, ladder_step=0.2, edge="None", edge_size=0.0, method="Continuous helix"):
    if state is None:
        return "Load a profile to see the resulting size."
    for label, value in (("Size", size_val), ("Height / laps", height_val), ("Sheet thickness", thickness),
                         ("Gap", gap), ("Layer height", layer)):
        if num(value, -1) <= 0:
            return f"Enter a positive {label.lower()} to see the resulting dimensions."
    thickness = num(thickness, D["thickness"])
    layer = num(layer, 0.2)
    edge_size = num(edge_size, 0.0, 0.0)
    d = derive(state, size_by, size_val, height_by, height_val, thickness, gap, snap_gap, layer, base, ladder, ladder_step, method)
    w, h = state["width_mm"] * d["scale"], state["height_mm"] * d["scale"]
    stroke = state["min_stroke_mm"] * d["scale"]
    warn = ""
    if d["laps"] > MAX_LAPS:
        warn += f"\n\n**Check:** {d['laps']:,} laps exceeds the {MAX_LAPS:,}-lap limit. Reduce the height."
    if layer > 0 and abs(d["gap"] / layer - round(d["gap"] / layer)) > 0.02:
        warn += f"\n\n**Check:** the {d['gap']:.2f} mm gap is not a whole number of layers; clearance will vary."
    if stroke < 1.0:
        warn += f"\n\n**Check:** the narrowest stroke is {stroke:.2f} mm. Increase the size for easier printing."
    last_gap = max(0, d["laps"] - (2 if method == "Staggered perimeter bridges" else 1))
    gap_txt = (f"Gap {d['gap']:.2f} mm ({d['gap']/layer:g} layers)" if d["step"] <= 0 else
               f"Gap ladder {d['gap']:.2f}–{d['gap'] + last_gap*d['step']:.2f} mm")
    if edge != "None" and edge_size > 0:
        e_eff = min(edge_size, 0.4 * thickness)
        gap_txt += (f"; {edge.lower()} taper {e_eff:.2f} mm"
                    + (" (limited to 40% of the sheet)" if e_eff < edge_size - 1e-9 else ""))
    if thickness / layer < 3 - 1e-6:
        warn += (f"\n\n**Check:** this sheet is only {thickness/layer:.1f} layers thick. "
                 "Use at least three layers per sheet.")
    if method == "Staggered perimeter bridges":
        warn += "\n\n" + STAGGERED_NOTE
    return (f"Footprint **{w:.1f} × {h:.1f} mm** · Height **{d['H']:.1f} mm** · **{d['laps']} laps**  \n"
            f"Sheet {thickness:.2f} mm · {gap_txt}" + warn)


# ----------------------------------------------------------------------------- generate
def require_number(label, value, minimum=0.0, inclusive=False):
    number = num(value, float("nan"))
    if not math.isfinite(number) or (number < minimum if inclusive else number <= minimum):
        rule = f"at least {minimum:g}" if inclusive else f"greater than {minimum:g}"
        raise gr.Error(f"{label} must be a finite number {rule}.")
    return number


def output_filename(value, fallback):
    name = str(value or "").strip() or fallback
    if not name.lower().endswith(".stl"):
        name += ".stl"
    stem = name[:-4]
    if (not stem or stem.endswith((" ", ".")) or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
            or re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", stem)):
        raise gr.Error("Enter a file name such as my_slinky.stl. Put the folder in Output folder.")
    return name


@contextlib.contextmanager
def output_pair(directory, name):
    """Reserve a new STL/project pair; never replace an existing result."""
    stem = name[:-4]
    suffix = 1
    while True:
        path = os.path.join(directory, f"{stem}{'' if suffix == 1 else '-' + str(suffix)}.stl")
        project = os.path.splitext(path)[0] + ".slinky.json"
        try:
            stl_file = open(path, "xb")
        except FileExistsError:
            suffix += 1
            continue
        stl_file.close()
        try:
            project_file = open(project, "x", encoding="utf-8")
        except FileExistsError:
            os.remove(path)
            suffix += 1
            continue
        except BaseException:
            os.remove(path)
            raise
        project_file.close()
        break
    try:
        yield path, project
    except BaseException:
        for created in (path, project):
            try:
                os.remove(created)
            except OSError:
                pass
        raise


def generate(state, size_by, size_val, height_by, height_val, strip, band_width, holes_as_coils,
             thickness, layer, gap, snap_gap, base, top, hand, material, grid, tol, min_hole,
             slice_check, out_dir, out_name, ladder, ladder_step, edge, edge_size,
             method="Continuous helix", bridge_advance=25.0, bridge_length=6.0, bridge_depth=2.0,
             see_through=True, live_preview=False, preview_only=False, progress=gr.Progress()):
    with GEOMETRY_LOCK:
        return _generate(state, size_by, size_val, height_by, height_val, strip, band_width, holes_as_coils,
                         thickness, layer, gap, snap_gap, base, top, hand, material, grid, tol, min_hole,
                         slice_check, out_dir, out_name, ladder, ladder_step, edge, edge_size, progress,
                         method, bridge_advance, bridge_length, bridge_depth, see_through, live_preview, preview_only)


def _generate(state, size_by, size_val, height_by, height_val, strip, band_width, holes_as_coils,
              thickness, layer, gap, snap_gap, base, top, hand, material, grid, tol, min_hole,
              slice_check, out_dir, out_name, ladder, ladder_step, edge, edge_size, progress,
              method, bridge_advance, bridge_length, bridge_depth, see_through, live_preview, preview_only=False):
    if state is None:
        raise gr.Error("Load a profile first.")
    size_val = require_number("Size", size_val)
    height_val = require_number("Height / laps", height_val)
    thickness = require_number("Sheet thickness", thickness)
    layer = require_number("Layer height", layer)
    gap = require_number("Gap between laps", gap)
    base = require_number("Extra base plate", base, inclusive=True)
    band_width = require_number("Band width", band_width) if strip != "Full profile" else num(band_width, 3.0, 1e-3)
    grid = require_number("Rise-field grid", grid) if strip == "Full profile" and method == "Continuous helix" else num(grid, 0.15, 1e-3)
    tol = require_number("Curve tolerance", tol)
    min_hole = require_number("Minimum hole area", min_hole, inclusive=True)
    edge_size = require_number("Taper size", edge_size) if edge != "None" else 0.0
    ladder_step = require_number("Ladder step", ladder_step) if ladder else 0.0
    if size_by not in ("Scale factor", "Width (in)", "Width (mm)") or height_by not in ("Laps", "Height (in)", "Height (mm)"):
        raise gr.Error("Choose a valid size and height mode.")
    if strip not in ("Full profile", "Band, inward", "Band, centered", "Band, outward"):
        raise gr.Error("Choose a valid strip mode.")
    if method not in METHODS:
        raise gr.Error("Choose a valid construction method.")
    if method == "Staggered perimeter bridges":
        bridge_advance = require_number("Bridge advance", bridge_advance)
        if bridge_advance >= 100:
            raise gr.Error("Bridge advance must be greater than 0 and less than 100 percent.")
        bridge_length = require_number("Bridge length", bridge_length)
        bridge_depth = require_number("Bridge depth", bridge_depth)
    else:
        bridge_advance = num(bridge_advance, 25.0)
        bridge_length = num(bridge_length, 6.0)
        bridge_depth = num(bridge_depth, 2.0)
    if edge not in ("None", "Round", "Chamfer") or top not in ("Flat fill", "Open") or hand not in ("Right", "Left") or material not in MATERIALS:
        raise gr.Error("Choose valid taper, top, winding and material settings.")
    try:
        d = derive(state, size_by, size_val, height_by, height_val, thickness, gap, snap_gap, layer, base, ladder, ladder_step, method)
    except (ValueError, OverflowError, ZeroDivisionError):
        raise gr.Error("The requested dimensions are too large. Reduce the size or height.") from None
    if d["laps"] > MAX_LAPS:
        raise gr.Error(f"This would create {d['laps']:,} laps. Use {MAX_LAPS:,} laps or fewer per model.")
    if not all(math.isfinite(v) for v in (d["H"], d["scale"], state["width_mm"] * d["scale"], state["height_mm"] * d["scale"])):
        raise gr.Error("The requested dimensions are too large. Reduce the size or height.")
    if slice_check and not d["step"] and (d["H"] - layer / 2) / layer > MAX_LAYER_SAMPLES:
        raise gr.Error(f"The layer check would exceed {MAX_LAYER_SAMPLES:,} sampled layers. "
                       "Increase the layer height or turn off Run the layer-by-layer print check.")
    band = {"Full profile": "full", "Band, inward": "inward", "Band, centered": "centered", "Band, outward": "outward"}[strip]
    kind = "gapladder" if d["step"] > 0 else "slinky"
    if method == "Staggered perimeter bridges":
        kind = "staggered_gapladder" if d["step"] > 0 else "staggered"
    default_name = f"{state['name']}_{kind}_{d['laps']}t"
    if preview_only:
        out_path = ""
    else:
        out_dir = os.path.abspath(os.path.expanduser((out_dir or "").strip() or DEFAULT_OUT))
        name = output_filename(out_name, default_name)
        out_path = os.path.join(out_dir, name)
    args = types.SimpleNamespace(
        regions=state["regions"], step="", dxf="", loops="", circle=60.0, save_loops="",
        scale=d["scale"], tol=tol, width=band_width, thickness=thickness, band=band,
        gap=d["gap"], turns=int(d["laps"]), base=base, top="fill" if top == "Flat fill" else "open",
        left=(hand == "Left"), holes=bool(holes_as_coils), min_hole_area=min_hole, layer=layer,
        material=material, no_layer_check=not slice_check, png=False, out=out_path, grid=grid, units="auto",
        gap_ladder=d["step"],
        edge={"None": "none", "Chamfer": "chamfer", "Round": "round"}[edge], edge_size=edge_size,
        edge_step=0.0, method=METHODS[method], bridge_advance=bridge_advance,
        bridge_length=bridge_length, bridge_depth=bridge_depth, capture_preview=True)
    buf = io.StringIO()
    purge_cache()
    progress(0.05, desc="building laps")
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            man, info = osl.build(args)
            progress(0.6, desc="preparing mesh" if preview_only else "preparing STL")
            V, F = from_manifold(man)
            progress(0.7, desc="checking layers" if slice_check else "report")
            osl.report(man, info, args)
    except (SystemExit, ValueError, RuntimeError, MemoryError, OverflowError) as exc:
        raise gr.Error(f"Could not build this shape: {exc}") from None
    # params next to the STL, so any result can be regenerated exactly
    params = {k: v for k, v in vars(args).items() if k not in ("regions", "capture_preview")}
    params.update(dict(profile=state["name"], size_by=size_by, size_val=size_val, height_by=height_by,
                       height_val=height_val, snap_gap=bool(snap_gap), laps=d["laps"], height_mm=float(info["H"]),
                       footprint_mm=[float(v) for v in info["footprint"]]))
    if preview_only:
        # Keep the reviewed print mesh in session memory. Generate writes only
        # a viewer asset, never an STL/project or the selected output directory.
        cache_root = gr_cache_dir()
        require_disk_space(cache_root, 80 * len(F))
        result_dir = os.path.realpath(tempfile.mkdtemp(prefix=f"draft-{os.getpid()}-", dir=cache_root))
        DRAFT_CACHES.add(result_dir)
        draft = dict(cache_dir=result_dir)
        preview_path = os.path.join(result_dir, "preview.glb")
        progress(0.9, desc="preview")
        try:
            write_live_preview(preview_path, info["preview_manifold"] if info["preview_manifold"] is not None else man,
                               info["preview_bridge_ids"], info["preview_ramps"], base=base)
        except (OSError, MemoryError, ValueError, RuntimeError) as exc:
            cleanup_draft(draft)
            if isinstance(exc, OSError) and (exc.errno == errno.ENOSPC or getattr(exc, "winerror", None) == 112):
                raise gr.Error("The disk filled up while making the preview. Free some space, then Generate preview again.") from None
            raise gr.Error(f"Could not make the 3D preview: {exc}") from None
        description = reviewed_description(params)
        vertices, faces = V.copy(), F.copy()
        vertices.flags.writeable = faces.flags.writeable = False
        draft.update(vertices=vertices, faces=faces,
                     params=json.loads(json.dumps(params, allow_nan=False)),
                     regions=json.loads(json.dumps(state["regions"], allow_nan=False)),
                     default_name=default_name, description=description, preview_path=preview_path,
                     token=uuid.uuid4().hex)
        build_report = (buf.getvalue().strip() + f"\n\nReviewed preview: {description}\n"
                        f"Total time {time.perf_counter() - t0:.1f} s")
        report = (build_report + "\n"
                  "STL and project have not been saved. Save STL & project saves this preview, "
                  "even if geometry settings have since changed.")
        draft["report"] = build_report
        progress(1.0, desc="preview ready")
        return preview_path, report, draft
    # STL, cache copy, framework download copy and approximately 80 bytes per
    # preview face. Check both destinations if the export is on another drive.
    require_disk_space(out_dir, 84 + 50 * len(F))
    require_disk_space(gr_cache_dir(), 2 * (84 + 50 * len(F)) + 80 * len(F))
    try:
        os.makedirs(out_dir, exist_ok=True)
        with output_pair(out_dir, name) as (out_path, project_path):
            params["out"] = out_path
            write_stl(out_path, V, F)
            with open(project_path, "w", encoding="utf-8") as f:
                json.dump({"params": params, "regions": state["regions"]}, f, allow_nan=False)
    except (OSError, ValueError) as exc:
        raise gr.Error(f"Could not save the result in {out_dir}: {exc}") from None
    buf.write(f"\nSaved STL: {out_path}\nSaved project: {project_path}\n")
    progress(0.9, desc="preview")
    preview, downloads = None, []
    try:
        result_dir = tempfile.mkdtemp(prefix="result-", dir=gr_cache_dir())
        for source in (out_path, project_path):
            destination = os.path.join(result_dir, os.path.basename(source))
            shutil.copyfile(source, destination)
            downloads.append(destination)
        viewer_dir = os.path.join(result_dir, "viewer")
        os.mkdir(viewer_dir)
        if live_preview:
            preview_path = os.path.join(viewer_dir, "preview.glb")
            write_live_preview(preview_path, info["preview_manifold"] if info["preview_manifold"] is not None else man,
                               info["preview_bridge_ids"], info["preview_ramps"], base=base)
            buf.write("\nPreview visibility updates directly in the browser; changing it does not rebuild or write the STL.\n")
        elif info["preview_manifold"] is not None:
            preview_path = os.path.join(viewer_dir, "preview.glb")
            write_bridge_preview(preview_path, info["preview_manifold"], info["preview_bridge_ids"],
                                 see_through=see_through, ramps=info["preview_ramps"])
            buf.write("\nPreview colors: gray laps; bridges from the bottom: orange, blue, magenta, "
                      "green, violet, gold (repeat every six gaps). Print STL geometry is unchanged.\n")
            if see_through:
                buf.write("Preview view: see-through laps with full ramps and buried attachments visible.\n")
        else:
            preview_path = os.path.join(viewer_dir, "preview.stl")
            # Gradio's STL loader swaps Y/Z. Pre-flip profile Y to make final
            # coordinates (x, z, -y), with print Z up and the profile upright.
            Vp = V * np.array([1.0, -1.0, 1.0])
            write_stl(preview_path, Vp - (Vp.min(0) + Vp.max(0)) / 2, F[:, ::-1])
        preview = preview_path
    except (OSError, MemoryError, ValueError, RuntimeError) as exc:
        if isinstance(exc, OSError) and (exc.errno == errno.ENOSPC or getattr(exc, "winerror", None) == 112):
            buf.write("\nThe STL and project were saved, but the disk filled up while saving the preview. "
                      "Free some disk space and Generate again.\n")
            downloads = []  # Don't ask Gradio to make another copy on a full disk.
        else:
            buf.write(f"\nPreview/download copy unavailable: {exc}. Your saved files are still in the output folder.\n")
    purge_cache()
    report = buf.getvalue().strip() + f"\n\nTotal time {time.perf_counter() - t0:.1f} s"
    progress(1.0, desc="saved")
    return preview, report, downloads or None


def generate_for_ui(state, size_by, size_val, height_by, height_val, strip, band_width, holes_as_coils,
                    thickness, layer, gap, snap_gap, base, top, hand, material, grid, tol, min_hole,
                    slice_check, out_dir, out_name, ladder, ladder_step, edge, edge_size,
                    method="Continuous helix", bridge_advance=25.0, bridge_length=6.0, bridge_depth=2.0,
                    previous_draft=None, progress=gr.Progress()):
    """Build a reviewed draft; output names and folders are validated only on Save."""
    with GEOMETRY_LOCK:
        preview, report, draft = generate(
            state, size_by, size_val, height_by, height_val, strip, band_width, holes_as_coils,
            thickness, layer, gap, snap_gap, base, top, hand, material, grid, tol, min_hole,
            slice_check, out_dir, out_name, ladder, ladder_step, edge, edge_size,
            method, bridge_advance, bridge_length, bridge_depth,
            live_preview=True, preview_only=True, progress=progress)
        payload = preview_payload(preview, method)
        cleanup_draft(previous_draft)
        return payload, report, None, draft, gr.update(interactive=True), draft["token"]


def save_preview(draft, out_dir, out_name, reviewed_token=None, progress=gr.Progress()):
    """Save the exact reviewed mesh without rebuilding current geometry settings."""
    if not isinstance(draft, dict) or not all(key in draft for key in ("vertices", "faces", "params", "regions", "report")):
        raise gr.Error("Generate a preview first, then use Save STL & project.")
    with GEOMETRY_LOCK:
        # State is resolved when a queued job runs; this ordinary client input
        # records which displayed draft the user clicked Save on.
        if reviewed_token is not None and reviewed_token != draft.get("token"):
            raise gr.Error("The preview changed while Save was queued. Review the current preview, then Save again.")
        out_dir = os.path.abspath(os.path.expanduser((out_dir or "").strip() or DEFAULT_OUT))
        name = output_filename(out_name, draft["default_name"])
        V, F = draft["vertices"], draft["faces"]
        purge_cache()
        require_disk_space(out_dir, 84 + 50 * len(F))
        require_disk_space(gr_cache_dir(), 2 * (84 + 50 * len(F)))
        params = json.loads(json.dumps(draft["params"], allow_nan=False))
        progress(0.2, desc="saving reviewed preview")
        try:
            os.makedirs(out_dir, exist_ok=True)
            with output_pair(out_dir, name) as (out_path, project_path):
                params["out"] = out_path
                write_stl(out_path, V, F)
                with open(project_path, "w", encoding="utf-8") as target:
                    json.dump(dict(params=params, regions=draft["regions"]), target, allow_nan=False)
        except (OSError, ValueError) as exc:
            raise gr.Error(f"Could not save the reviewed preview in {out_dir}: {exc}") from None
        report = (draft["report"] + f"\n\nSaved reviewed preview: {draft['description']}\n"
                  f"Saved STL: {out_path}\nSaved project: {project_path}")
        downloads = []
        try:
            result_dir = tempfile.mkdtemp(prefix="result-", dir=gr_cache_dir())
            for source in (out_path, project_path):
                destination = os.path.join(result_dir, os.path.basename(source))
                shutil.copyfile(source, destination)
                downloads.append(destination)
        except OSError as exc:
            downloads = []
            report += f"\nDownloads unavailable: {exc}. The saved files are in the output folder."
        purge_cache()
        progress(1.0, desc="saved")
        return report, downloads or None


# ----------------------------------------------------------------------------- UI
# Shared by startup controls, readouts and Reset. Preserve the successful PETG
# lap/gap settings, with the user's latest 150% profile and 30-turn size.
D = dict(size_by="Scale factor", size_val=1.5, height_by="Laps", height_val=30, strip="Full profile",
         band_width=3.0, thickness=0.8, layer=0.2, gap=0.4, snap=False, base=0.5, top="Flat fill",
         hand="Right", material="PETG", edge="Round", edge_size=0.3, ladder=False, ladder_step=0.2,
         holes=False, grid=0.15, tol=0.02, min_hole=2.0, slice_check=True,
         method="Continuous helix", bridge_advance=25.0, bridge_length=6.0, bridge_depth=2.0)
SETTING_KEYS = list(D)


def saved_settings(state):
    """Restore geometry controls, never file paths supplied by an imported JSON file."""
    p = state.get("params") if state else None
    if p is None:
        return None
    raw_choices = dict(band=("full", "inward", "centered", "outward"),
                       top=("fill", "open"), edge=("round", "chamfer", "none"), method=("helix", "staggered"))
    for key, choices in raw_choices.items():
        if key in p and p[key] not in choices:
            raise ValueError(f"Saved {key} has an unsupported value: {p[key]!r}.")
    for key in ("left", "holes", "snap_gap", "no_layer_check"):
        if key in p and not isinstance(p[key], bool):
            raise ValueError(f"Saved {key} must be true or false.")
    settings = D.copy()
    mapping = dict(band_width="width", holes="holes", thickness="thickness", layer="layer", gap="gap",
                   base="base", grid="grid", tol="tol", min_hole="min_hole_area", edge_size="edge_size",
                   bridge_advance="bridge_advance", bridge_length="bridge_length", bridge_depth="bridge_depth")
    for key, source in mapping.items():
        if source in p:
            settings[key] = p[source]
    settings.update(
        size_by=p.get("size_by", "Scale factor"), size_val=p.get("size_val", p.get("scale", 1.0)),
        height_by=p.get("height_by", "Laps"), height_val=p.get("height_val", p.get("turns", 6)),
        strip={"full": "Full profile", "inward": "Band, inward", "centered": "Band, centered", "outward": "Band, outward"}.get(p.get("band"), "Full profile"),
        top="Open" if p.get("top") == "open" else "Flat fill", hand="Left" if p.get("left") else "Right",
        material=p.get("material", "PETG"), snap=p.get("snap_gap", False),
        edge={"round": "Round", "chamfer": "Chamfer", "none": "None"}.get(p.get("edge"), "None"),
        ladder=num(p.get("gap_ladder")) > 0, ladder_step=p.get("gap_ladder", 0.2),
        slice_check=not p.get("no_layer_check", False),
        method={"helix": "Continuous helix", "staggered": "Staggered perimeter bridges"}[p.get("method", "helix")])
    choices = dict(size_by=("Scale factor", "Width (in)", "Width (mm)"),
                   height_by=("Laps", "Height (in)", "Height (mm)"), material=MATERIALS)
    for key, allowed in choices.items():
        if settings[key] not in allowed:
            raise ValueError(f"Saved project has an unsupported {key}: {settings[key]!r}.")
    for key, default in D.items():
        value = settings[key]
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise ValueError(f"Saved {key} must be true or false.")
        elif isinstance(default, (int, float)):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"Saved {key} must be a finite number.")
    return settings


def load_into_ui(file, units):
    state, info, png = load_profile(file, units)
    try:
        settings = saved_settings(state)
    except (ValueError, TypeError, OverflowError) as exc:
        return (None, f"**Could not load this project.** {exc}", None,
                *[gr.skip() for _ in SETTING_KEYS], gr.skip(), gr.skip())
    values = [settings[key] for key in SETTING_KEYS] if settings else [gr.skip() for _ in SETTING_KEYS]
    return (state, info, png, *values,
            settings["size_by"] if settings else gr.skip(), settings["height_by"] if settings else gr.skip())


def convert_size_mode(state, old_mode, new_mode, value):
    if not state or num(value) <= 0:
        return gr.update(label=new_mode), new_mode
    width = num(value) * (state["width_mm"] if old_mode == "Scale factor" else IN if old_mode == "Width (in)" else 1)
    value = width / (state["width_mm"] if new_mode == "Scale factor" else IN if new_mode == "Width (in)" else 1)
    return gr.update(value=value, label=new_mode), new_mode


def convert_height_mode(state, old_mode, new_mode, value, thickness, gap, snap, layer, base, ladder, step,
                        method="Continuous helix"):
    try:
        if num(value, -1) <= 0:
            return gr.update(label=new_mode), new_mode
        # Height depends on print settings even when the profile has been cleared.
        d = derive(state or {"width_mm": 1.0}, "Scale factor", 1.0, old_mode, value,
                   thickness, gap, snap, layer, base, ladder, step, method)
        if new_mode == "Laps":
            converted = d["laps"]
        else:
            height = d["H"] if old_mode == "Laps" else num(value) * (IN if old_mode == "Height (in)" else 1)
            converted = height / (IN if new_mode == "Height (in)" else 1)
        if not math.isfinite(converted):
            return gr.update(label=new_mode), new_mode
    except (ValueError, OverflowError, ZeroDivisionError):
        return gr.update(label=new_mode), new_mode
    return gr.update(value=converted, label=new_mode), new_mode


def make_ui():
    init_state, init_info, init_png = (None, "Load a profile to begin.", None)
    if os.path.exists(DEFAULT_PROFILE):
        try:
            init_state, init_info, init_png = load_profile(DEFAULT_PROFILE, "auto")
        except Exception as exc:                       # a bad sample file must not stop the app
            init_info = f"Could not load {os.path.basename(DEFAULT_PROFILE)}: {exc}"
    init_summary = readout(init_state, D["size_by"], D["size_val"], D["height_by"], D["height_val"],
                           D["thickness"], D["gap"], D["snap"], D["layer"], D["base"],
                           D["ladder"], D["ladder_step"], D["edge"], D["edge_size"], D["method"])
    with gr.Blocks(title="Slinkify", analytics_enabled=False) as demo:
        gr.Markdown("# Slinkify\nDesign a slinky from any 2D profile.", elem_id="slinkify-header")
        state = gr.State(init_state)
        draft_state = gr.State(None, delete_callback=cleanup_draft)
        draft_token = gr.Textbox(value="", visible=False)
        size_mode = gr.State(D["size_by"])
        height_mode = gr.State(D["height_by"])
        with gr.Row(elem_id="slinkify-workspace"):
            with gr.Column(scale=1, min_width=340, elem_id="slinkify-settings"):
                with gr.Column(min_width=0, elem_classes="sl-section"):
                    gr.Markdown("## Profile", elem_classes="sl-section-heading")
                    file = gr.File(label="Profile file", file_types=[".dxf", ".step", ".stp", ".json"],
                                   value=DEFAULT_PROFILE if os.path.exists(DEFAULT_PROFILE) else None)
                    with gr.Row(elem_classes="sl-form-row"):
                        units = gr.Dropdown(["auto", "mm", "in"], value="auto", label="DXF units")
                        load_btn = gr.Button("Reload profile", variant="secondary")
                    with gr.Accordion("Profile details", open=False, elem_classes="sl-disclosure"):
                        info = gr.Markdown(init_info, elem_id="slinkify-profile-details")
                        prof_img = gr.Image(label="Profile outline", type="filepath", height=220, value=init_png,
                                            interactive=False, buttons=["fullscreen"])
                with gr.Column(min_width=0, elem_classes="sl-section"):
                    gr.Markdown("## Size", elem_classes="sl-section-heading")
                    with gr.Row(elem_classes="sl-form-row"):
                        size_by = gr.Dropdown(["Scale factor", "Width (in)", "Width (mm)"], value=D["size_by"], label="Size by")
                        size_val = gr.Number(value=D["size_val"], label=D["size_by"], precision=4, minimum=0)
                    with gr.Row(elem_classes="sl-form-row"):
                        height_by = gr.Dropdown(["Laps", "Height (in)", "Height (mm)"], value=D["height_by"], label="Height by")
                        height_val = gr.Number(value=D["height_val"], label=D["height_by"], precision=3, minimum=0)
                    reset_btn = gr.Button("Reset PETG defaults", variant="secondary", size="sm")
                with gr.Column(min_width=0, elem_classes="sl-section"):
                    gr.Markdown("## Construction", elem_classes="sl-section-heading")
                    method = gr.Radio(list(METHODS), value=D["method"], label="Method")
                    with gr.Column(min_width=0, visible=D["method"] == "Staggered perimeter bridges", elem_classes="sl-section") as bridge_controls:
                        with gr.Row(elem_classes="sl-form-row"):
                            bridge_advance = gr.Number(value=D["bridge_advance"], label="Advance (%)", precision=2, interactive=False, min_width=90)
                            bridge_length = gr.Number(value=D["bridge_length"], label="Length (mm)", precision=2, interactive=False, min_width=90)
                            bridge_depth = gr.Number(value=D["bridge_depth"], label="Depth (mm)", precision=2, interactive=False, min_width=90)
                        ramp_summary = gr.Markdown("", elem_id="slinkify-ramp-summary")
                    strip = gr.Dropdown(["Full profile", "Band, inward", "Band, centered", "Band, outward"],
                                        value=D["strip"], label="Cross-section")
                    with gr.Column(min_width=0, visible=D["strip"] != "Full profile", elem_classes="sl-section") as band_controls:
                        band_width = gr.Number(value=D["band_width"], label="Band width (mm)", precision=2, interactive=False)
                        holes_as_coils = gr.Checkbox(value=D["holes"], label="Nested coils for holes", interactive=False)
                with gr.Column(min_width=0, elem_classes="sl-section"):
                    gr.Markdown("## Print", elem_classes="sl-section-heading")
                    with gr.Row(elem_classes="sl-form-row"):
                        thickness = gr.Number(value=D["thickness"], label="Sheet thickness (mm)", precision=3)
                        layer = gr.Number(value=D["layer"], label="Layer height (mm)", precision=3)
                    with gr.Row(elem_classes="sl-form-row"):
                        gap = gr.Number(value=D["gap"], label="Lap gap (mm)", precision=3)
                        snap_gap = gr.Checkbox(value=D["snap"], label="Snap gap to whole layers")
                with gr.Accordion("Print options", open=False, elem_classes="sl-disclosure"):
                    with gr.Row(elem_classes="sl-form-row"):
                        edge = gr.Radio(["Round", "Chamfer", "None"], value=D["edge"], label="Lap edge taper")
                        edge_size = gr.Number(value=D["edge_size"], label="Taper size (mm)", precision=3)
                    with gr.Row(elem_classes="sl-form-row"):
                        base = gr.Number(value=D["base"], label="Base thickness (mm)", precision=2)
                        top = gr.Radio(["Flat fill", "Open"], value=D["top"], label="Top")
                    with gr.Row(elem_classes="sl-form-row"):
                        hand = gr.Radio(["Right", "Left"], value=D["hand"], label="Winding")
                        material = gr.Dropdown(list(MATERIALS), value=D["material"], label="Material estimate")
                    slice_check = gr.Checkbox(value=D["slice_check"], label="Check every print layer (slower)")
                with gr.Accordion("Advanced geometry", open=False, elem_classes="sl-disclosure"):
                    with gr.Row(elem_classes="sl-form-row"):
                        grid = gr.Number(value=D["grid"], label="Rise-field grid (mm)", precision=3)
                        tol = gr.Number(value=D["tol"], label="Curve tolerance (mm)", precision=3)
                        min_hole = gr.Number(value=D["min_hole"], label="Ignore holes under (mm²)", precision=2)
                    with gr.Row(elem_classes="sl-form-row"):
                        ladder = gr.Checkbox(value=D["ladder"], label="Gap ladder coupon")
                        ladder_step = gr.Number(value=D["ladder_step"], label="Ladder step (mm)", precision=3, interactive=False)
            with gr.Column(scale=2, min_width=500, elem_id="slinkify-preview-panel"):
                gr.Markdown("## Preview", elem_id="slinkify-preview-heading")
                summary = gr.Markdown(init_summary, elem_id="slinkify-dimensions")
                with gr.Row(elem_classes="sl-form-row"):
                    gen_btn = gr.Button("Generate preview", variant="primary", interactive=init_state is not None,
                                        elem_id="slinkify-generate-preview")
                    save_btn = gr.Button("Save STL & project", variant="secondary", interactive=False,
                                         elem_id="slinky-save-preview")
                gr.Markdown("Generate to apply changes. Save exports the last preview.", elem_id="slinkify-preview-note")
                model = preview_component()
                gr.HTML(preview_legend())
                with gr.Accordion("Export settings", open=False, elem_classes="sl-disclosure"):
                    out_dir = gr.Textbox(value=DEFAULT_OUT, label="Output folder")
                    out_name = gr.Textbox(value="", label="File name", placeholder="Automatic",
                                          info="Existing files are kept; duplicate names receive a number.")
                with gr.Accordion("Saved files", open=False, elem_classes="sl-disclosure") as saved_files:
                    download = gr.File(label="STL & project", file_count="multiple", interactive=False)
                with gr.Accordion("Build details", open=False, elem_classes="sl-disclosure"):
                    report = gr.Textbox(label="Build report", lines=8, max_lines=30, interactive=False,
                                        elem_id="slinkify-build-report")

        setting_components = dict(size_by=size_by, size_val=size_val, height_by=height_by, height_val=height_val,
                                  strip=strip, band_width=band_width, holes=holes_as_coils, thickness=thickness,
                                  layer=layer, gap=gap, snap=snap_gap, base=base, top=top, hand=hand,
                                  material=material, grid=grid, tol=tol, min_hole=min_hole, slice_check=slice_check,
                                  ladder=ladder, ladder_step=ladder_step, edge=edge, edge_size=edge_size,
                                  method=method, bridge_advance=bridge_advance, bridge_length=bridge_length,
                                  bridge_depth=bridge_depth)
        settings_outputs = [setting_components[key] for key in SETTING_KEYS]
        help_components = dict(profile=file, units=units, **setting_components, out_dir=out_dir,
                               out_name=out_name)
        demo.load(fn=None, inputs=None, outputs=None, js=attach_parameter_help(help_components))
        gr.on([load_btn.click, file.change, units.change], load_into_ui, [file, units],
              [state, info, prof_img, *settings_outputs, size_mode, height_mode],
              concurrency_id="geometry", concurrency_limit=1, trigger_mode="always_last",
              show_progress_on=[info, prof_img])
        reset_btn.click(lambda: (*[D[key] for key in SETTING_KEYS], D["size_by"], D["height_by"]),
                        outputs=[*settings_outputs, size_mode, height_mode], queue=False)
        size_by.input(convert_size_mode, [state, size_mode, size_by, size_val], [size_val, size_mode], queue=False)
        height_by.input(convert_height_mode, [state, height_mode, height_by, height_val, thickness, gap,
                                             snap_gap, layer, base, ladder, ladder_step, method], [height_val, height_mode], queue=False)
        size_by.change(lambda mode: gr.update(label=mode), size_by, size_val, queue=False)
        height_by.change(lambda mode: gr.update(label=mode), height_by, height_val, queue=False)
        gr.on([strip.change, method.change],
              lambda strip_mode, construction: (gr.update(interactive=strip_mode != "Full profile"),
                                                gr.update(interactive=strip_mode != "Full profile"),
                                                gr.update(interactive=strip_mode == "Full profile" and construction == "Continuous helix"),
                                                gr.update(visible=strip_mode != "Full profile")),
              [strip, method], [band_width, holes_as_coils, grid, band_controls], queue=False)
        method.change(lambda construction: [*[gr.update(interactive=construction == "Staggered perimeter bridges") for _ in range(3)],
                                           gr.update(visible=construction == "Staggered perimeter bridges")],
                      method, [bridge_advance, bridge_length, bridge_depth, bridge_controls], queue=False)
        edge.change(lambda mode: gr.update(interactive=mode != "None"), edge, edge_size, queue=False)
        ladder.change(lambda enabled: gr.update(interactive=enabled), ladder, ladder_step, queue=False)
        state.change(lambda value: (gr.update(interactive=value is not None),
                                    gr.update(interactive=bool(value and value.get("file_type") == ".dxf"))),
                     state, [gen_btn, units], queue=False)
        live = [state, size_by, size_val, height_by, height_val, thickness, gap, snap_gap, layer, base,
                ladder, ladder_step, edge, edge_size, method]
        gr.on([component.change for component in live], readout, live, summary,
              queue=False, trigger_mode="always_last", show_progress="hidden")
        ramp_live = [method, thickness, gap, snap_gap, layer, ladder, ladder_step, bridge_length]
        gr.on([component.change for component in ramp_live], bridge_readout, ramp_live, ramp_summary,
              queue=False, trigger_mode="always_last", show_progress="hidden")
        gen_btn.click(generate_for_ui,
                      [state, size_by, size_val, height_by, height_val, strip, band_width, holes_as_coils,
                       thickness, layer, gap, snap_gap, base, top, hand, material, grid, tol, min_hole,
                       slice_check, out_dir, out_name, ladder, ladder_step, edge, edge_size,
                       method, bridge_advance, bridge_length, bridge_depth, draft_state],
                      [model, report, download, draft_state, save_btn, draft_token], concurrency_id="geometry", concurrency_limit=1,
                      trigger_mode="once", show_progress_on=[model, report])
        save_btn.click(save_preview, [draft_state, out_dir, out_name, draft_token], [report, download],
                       concurrency_id="geometry", concurrency_limit=1, trigger_mode="once",
                       show_progress_on=[report, download]).success(lambda: gr.update(open=True), outputs=saved_files, queue=False)
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    purge_cache()
    demo = make_ui()
    demo.launch(server_name="127.0.0.1", server_port=a.port, inbrowser=not a.no_browser, show_error=True,
                share=False, allowed_paths=[gr_cache_dir(), DEFAULT_PROFILE],
                theme=APP_THEME, css=APP_CSS, footer_links=["settings"])


if __name__ == "__main__":
    main()
