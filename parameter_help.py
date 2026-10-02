"""Hover and screen-reader descriptions for Slinkify's editable controls."""
import json


PARAMETER_HELP = {
    "see_through": (
        "Make preview laps transparent immediately. Enable Reveal ramp attachments to show the "
        "complete ramps, including material buried inside the sheets. Turn off for solid laps. "
        "Visibility changes keep the camera view and do not rebuild the print STL."
    ),
    "method": (
        "Continuous helix retains the original coil. Staggered perimeter bridges makes flat laps "
        "joined by local ramps that advance around the outline, distributing the bending points. "
        "It is a chain of linked laps rather than a continuous helical strand. Experimental: flat laps "
        "have larger unsupported areas, so inspect the sliced model and print a short coupon first."
    ),
    "bridge_advance": (
        "How far the next connection moves along the outline's arc length, as a percentage of its "
        "perimeter. 25% visits four positions; 20% visits five. Winding reverses travel. "
        "Enter a value greater than 0 and less than 100. Used only for Staggered perimeter bridges."
    ),
    "bridge_length": (
        "Length of each connecting ramp along the perimeter, in millimeters. A longer ramp rises more "
        "gently but connects a larger area and can be stiffer. Nominal angle from horizontal is "
        "atan((sheet thickness + effective gap) / length). 0.8 + 0.4 mm over 6 mm gives 11.3 degrees; "
        "over 12 mm gives 5.7 degrees. Length does not scale with the profile. Must be shorter than the perimeter. "
        "Used only for Staggered perimeter bridges."
    ),
    "bridge_depth": (
        "How far each ramp extends across the local material footprint, in millimeters. The ramp is "
        "clipped to the actual shape. Deeper ramps generally make stronger, stiffer connections. "
        "Used only for Staggered perimeter bridges."
    ),
    "profile": (
        "Choose the closed 2D outline to turn into a coil: DXF, a STEP straight extrusion, "
        "or loops JSON. A .slinky.json project restores both the profile and its print settings. "
        "The output folder stays unchanged."
    ),
    "units": (
        "How to interpret distances in a DXF. Auto uses the file's unit header and assumes "
        "millimeters if units are missing. Choose mm or in to override it. Changing this "
        "reloads the DXF; it does not apply to STEP or JSON files."
    ),
    "size_by": (
        "Choose how to size the profile: multiply its original dimensions, or set its width "
        "in inches or millimeters. Switching modes converts the current value. The profile's "
        "proportions stay the same."
    ),
    "size_val": (
        "The value for Size by. Scale factor 1 keeps the original size; 1.5 makes the profile's "
        "width and height 50% larger. In a Width mode, enter the desired profile width in the selected units. "
        "Sheet thickness, gap and band width are set separately."
    ),
    "height_by": (
        "Set the coil's height using a number of laps or a target height in inches or millimeters. "
        "Target heights choose the nearest whole lap count, so the resulting height can differ "
        "slightly. Switching modes converts the current value."
    ),
    "height_val": (
        "The number of laps, or the target height in the units selected by Height by. More laps "
        "make a taller, longer coil and increase print time and material use. The height includes "
        "the sheet, gaps and any extra base plate. Maximum: 1,000 laps."
    ),
    "strip": (
        "Full profile uses the entire shape, including its holes, for each lap. Band modes use "
        "a ribbon along the outline: inward stays inside it, centered straddles it, and outward "
        "extends outside it. In Continuous helix, Full profile needs an interior opening reached by the spiral seam; "
        "use a band mode for solid shapes. Centered and outward bands can exceed the displayed profile footprint."
    ),
    "band_width": (
        "Width of the ribbon measured across the outline in band modes. Wider bands use more "
        "material and are generally stiffer. With nested hole coils, keep this below roughly half "
        "the narrowest profile stroke to avoid overlapping coils. Unused in Full profile mode."
    ),
    "holes": (
        "In band modes, also build separate coils around the profile's interior holes. Wide "
        "outer and inner bands can overlap and fuse; check the generation report. In Full profile "
        "mode, holes are already part of the shape, so this option is unused."
    ),
    "thickness": (
        "Vertical thickness of each coil lap. Thicker sheets are generally stiffer; thinner sheets "
        "are more flexible but harder to bridge reliably. Aim for at least about three slicer "
        "layers per sheet; the default 0.8 mm sheet spans four nominal 0.2 mm layers. "
        "Thickness plus gap sets the starting pitch."
    ),
    "layer": (
        "Your slicer's intended layer height. Match it to the actual print profile. It controls "
        "gap snapping, taper steps and the layer-by-layer check. Smaller layers shorten the "
        "leading bridge region in Continuous helix but increase the number of layers to print and check. "
        "Staggered flat laps still have large unsupported areas."
    ),
    "gap": (
        "Vertical air clearance between neighboring laps, before any extra edge clearance from "
        "the taper. Continuous helix leaves an open spiral gap; Staggered perimeter bridges intentionally "
        "closes each gap at its local ramp. Slight bonding can still occur during "
        "printing. Too little clearance can weld laps together; larger gaps can be harder to bridge. "
        "Use a short coupon to tune it for your printer and filament."
    ),
    "snap": (
        "Round the gap to the nearest whole number of layer heights, with a minimum of one layer. "
        "This also rounds an enabled ladder step. The live readout shows the effective gap used "
        "for generation; the entered number stays visible in the field."
    ),
    "edge": (
        "Shape the top and bottom edges of each lap to reduce contact near the perimeter. "
        "Round gives a curved taper, Chamfer gives a straight bevel, and None keeps square edges. "
        "The center of the sheet retains the normal gap; bed and flat-top faces stay flat."
    ),
    "edge_size": (
        "Radius of a rounded taper or size of a chamfer, in millimeters. Larger values remove "
        "more material near each lap edge and leave less full-width sheet. The effective size "
        "is capped at 40% of sheet thickness. Unused when Lap edge taper is None."
    ),
    "base": (
        "Thickness of an extra solid plate below the first lap. Zero adds no extra plate; "
        "the first lap is already filled down to the bed. Increasing this adds base stiffness, "
        "height and material."
    ),
    "top": (
        "Flat fill fills the last lap up to a level top surface. Open leaves the last lap "
        "following the rising helix, with a sloped end. In Staggered perimeter bridges, both tops "
        "are level: Open retains the final lap's top-edge taper. This changes the top geometry rather "
        "than the selected number of laps."
    ),
    "hand": (
        "Choose the direction in which the coil winds as it rises. Left reverses Right. "
        "This reverses the helix or the perimeter bridge advance without mirroring the imported profile or changing "
        "the requested size, thickness or gaps."
    ),
    "material": (
        "Material used for the report's approximate mass, stiffness and hanging-length estimates. "
        "Stiffness and hanging-length estimates are unavailable for Staggered perimeter bridges. "
        "Changing it does not change the mesh or automatically set thickness, gap, temperature "
        "or other slicer settings."
    ),
    "grid": (
        "Spacing of the grid used to calculate the rise through a Full profile. Smaller values "
        "resolve narrow strokes better but require more memory and calculation. Larger values "
        "are faster but can miss thin features. Unused in band modes and Staggered perimeter bridges."
    ),
    "tol": (
        "Tolerance for simplifying the scaled profile outline, in millimeters. Smaller values "
        "retain more contour detail and create more geometry. Larger values reduce complexity "
        "but can remove small details or collapse narrow features."
    ),
    "min_hole": (
        "Ignore holes smaller than this area, measured in square millimeters after scaling. "
        "In Full profile mode, ignored holes are filled; in band modes, their nested coils "
        "are omitted. Use zero to retain all valid holes."
    ),
    "ladder": (
        "Make a clearance-test coupon whose gap grows with each lap. Print a short coil to "
        "compare which clearances separate best. The gap changes continuously along the helix. "
        "The layer-by-layer support check is not available for this variable-pitch mode."
    ),
    "ladder_step": (
        "How much to increase the gap per lap when the gap ladder is enabled. For example, "
        "a 0.4 mm starting gap and 0.2 mm step give successive gap levels of 0.4, 0.6, 0.8 mm, "
        "and so on. Snap gap to whole layers also rounds this step."
    ),
    "slice_check": (
        "After building, inspect sliced layers to estimate how bridged regions land on material "
        "below. This adds processing time and is limited to 10,000 sampled layers. Gap ladders "
        "are explicitly skipped. It is a geometric check, not a guarantee of print success."
    ),
    "out_dir": (
        "Local folder used by Save STL & project for the last generated preview. Generating a "
        "preview does not save files here. Missing folders are created when saving. Leave blank "
        "to use the default folder above this app. This folder "
        "is not changed when you import a saved project or reset print settings."
    ),
    "out_name": (
        "File name used by Save STL & project; the saved project uses the same base name. Leave blank "
        "to name it from the profile and lap count. The .stl extension is added automatically. "
        "Existing files are kept and a numbered suffix is added. Enter the folder separately."
    ),
}


def attach_parameter_help(components):
    """Use native hover tooltips and equivalent descriptions on keyboard controls.

    A small observer reapplies descriptions when Gradio replaces controls after a
    mode change. It observes child replacement only, not our own attribute changes.
    """
    descriptions = {}
    for key, component in components.items():
        component.elem_id = f"slinky-setting-{key}"
        descriptions[component.elem_id] = PARAMETER_HELP[key]
    return """() => {
        const descriptions = %s;
        const root = document.querySelector('.gradio-container');
        if (!root) return;
        if (root.slinkyHelpObserver) root.slinkyHelpObserver.disconnect();
        const applyHelp = () => {
            for (const [id, description] of Object.entries(descriptions)) {
                const field = document.getElementById(id);
                if (!field) continue;
                field.setAttribute('title', description);
                for (const control of field.querySelectorAll(
                    'input, textarea, select, button, [role="combobox"], [role="radiogroup"]'
                )) {
                    control.setAttribute('title', description);
                    control.setAttribute('aria-description', description);
                }
            }
        };
        let pending = false;
        const observer = new MutationObserver(() => {
            if (pending) return;
            pending = true;
            requestAnimationFrame(() => { pending = false; applyHelp(); });
        });
        observer.observe(root, {childList: true, subtree: true});
        root.slinkyHelpObserver = observer;
        applyHelp();
    }""" % json.dumps(descriptions)
