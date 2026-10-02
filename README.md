# Slinkify

Slinkify turns closed 2D profiles into helical coils or linked laps. Run locally in your browser, inspect the 3D model, and export an STL with a reloadable project.

## Quick start

Requires **Python 3.10 or newer**. The required packages, including **Gradio 6.28–6.x**, are listed in `requirements.txt`.

```shell
git clone https://github.com/kazzurt/Slinkify.git
cd Slinkify
python -m pip install -r requirements.txt
python slinky_app.py
```

The app opens at [127.0.0.1:7860](http://127.0.0.1:7860). Keep the terminal open; stop the server with `Ctrl+C`.

On Windows, double-click **`run_slinky_app.bat`** to find a compatible Python environment, install missing dependencies, and open the app.

If the port is busy, run `python slinky_app.py --port 7861`. Add `--no-browser` to start the server without opening a browser tab.

STEP import also requires the optional `cadquery` package:

```shell
python -m pip install cadquery
```

DXF and JSON import work with the standard requirements.

## Preview, then save

1. **Load a profile.** The included `DDR_profile.dxf` loads automatically. Import another DXF, a straight-extrusion STEP, loops JSON, or `.slinky.json` project. Projects restore the profile and geometry settings. DXF units can be detected automatically or set to millimeters or inches.
2. **Choose geometry.** Set XY scale or target width, construction method, and lap count or target height. Target heights use the closest attainable lap count. Hover over controls for help.
3. **Generate preview.** Inspect the model; open **Build details** for the report. This creates a temporary viewer asset without saving an STL or project in the output folder.
4. **Save STL & project.** Open **Export settings** to select the output folder and filename, then use Save beside Generate preview. It exports the exact previewed geometry without rebuilding it. **Saved files** opens with both downloads; numbered suffixes preserve existing files.

Generate another preview after changing geometry. Save uses the stored preview; visibility and camera changes do not affect exports.

## Construction methods

**Continuous helix** is the original method and startup default: a continuous strand with a helical gap. **Full profile** forms broad sheets; inward, centered, or outward bands form narrower ribbons. Full profile requires an interior opening reached by the seam. For solid outlines, use a band that leaves an opening. Band modes can add separate coils around holes.

**Staggered perimeter bridges** joins flat laps with short ramps. Successive connections advance along the perimeter: **25% advance cycles through four positions**. Advance measures arc length, not an angle around the center; winding reverses its direction.

Staggered bridges are experimental: flat laps create larger unsupported areas, and spring motion and durability require physical testing. Inspect the slicer preview and print a six-lap coupon before a larger model.

## Geometry controls

Unless explicitly labeled otherwise, lengths are in millimeters. Scaling changes the profile in XY; sheet thickness, gap, taper, base, and ramp dimensions retain their millimeter values.

| Control | Effect |
| --- | --- |
| Sheet thickness | Vertical thickness of each lap. |
| Layer height | Layer-check sampling, gap snapping, and taper steps. Set the matching value in your slicer. |
| Lap gap | Designed clearance. Pitch is sheet thickness plus effective gap. |
| Lap edge taper | Rounds or chamfers lap edges; reduces the full-width part of the sheet. |
| Base thickness | Adds a flat base below the first lap. |
| Advance (%) | Moves successive staggered connections around the perimeter. |
| Length (mm) / Depth (mm) | Sets the ramp's perimeter run and its reach into the local material. |
| Gap ladder coupon | Increases the gap at each lap to compare clearances in one short print. |

The nominal staggered ramp angle from horizontal is:

```text
angle = atan((sheet thickness + effective gap) / bridge length)
```

With a **0.8 mm sheet and 0.4 mm gap**, the rise is 1.2 mm: **6 mm length gives 11.3°; 12 mm gives 5.7°**. Length measures the perimeter run. Corners, clipping, and taper can change local slope; gap ladders steepen later ramps. The live readout shows the nominal grade.

Startup defaults use **1.5× XY scale, 30 laps**, 0.8 mm sheets, 0.4 mm gaps, and 0.2 mm layers. **PETG** is selected for report estimates. Choose filament, temperatures, and process settings separately in your slicer when using PLA, PETG, or another material.

## Inspect the connections

Seven checkboxes update the loaded 3D scene immediately and preserve the camera:

- **Laps**, **Bridges**, and **Base** show or hide those features.
- **Full ramps** reveals complete ramp volumes, including attachments buried in the laps. Off shows exposed bridge surfaces.
- **Transparent laps** lets you see connections through the sheets.
- **Bridge colors** distinguishes connections by gap, repeating every six gaps.
- **Wireframe** shows triangle edges.

Unavailable features are disabled. Use Full ramps with Transparent laps, or hide Laps, to inspect connections without creating STL files.

The vertical **Layer** slider on the preview's right edge cuts the model at print-layer heights. Drag down to hide higher layers; the readout shows the selected layer and height above the bed. **Layer only** isolates one layer, and **Full** restores the complete model. Arrow keys move one layer; Home selects the first and End selects the top.

Layer inspection uses the layer height from the generated preview. It preserves the camera and feature visibility, and Save still exports the complete model. These are geometry cross-sections; your slicer creates the actual print toolpaths.

## Checks and troubleshooting

**Check every print layer**, under **Print options**, samples constant-pitch models and skips gap ladders. Mesh validation and support checks help inspect a design; review the sliced model and test clearance with your printer and filament.

Free-space errors report the room needed and available. Previews use a temporary cache; saving also needs output-folder space. Free space and retry. Failed saves remove incomplete STL/project pairs and retain the preview for another attempt.

For full-profile helixes, an overly fine rise-field grid can exceed the allocation limit; follow the suggested larger value. Limits are 1,000 laps and 10,000 layer-check samples.

Run the regression suite from the repository folder:

```shell
python -m unittest discover -s tests -v
```

The geometry engine is also available as a CLI. For example:

```shell
python outline_slinky.py --dxf DDR_profile.dxf --band full --scale 1.5 --thickness 0.8 --gap 0.4 --turns 30 --out example.stl
```

Use `python outline_slinky.py --help` for all engine options.
