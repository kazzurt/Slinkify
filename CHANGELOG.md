# Slinkify changelog

## Layer inspection

- Vertical preview slider clips the model at its generated print-layer heights, with layer and height readouts.
- Layer only isolates one print layer; Full restores the complete preview. Mouse, touch and keyboard controls work locally without rebuilding geometry.
- Camera, feature visibility and complete STL/project exports are preserved while inspecting a cross-section.

## Initial public release

- Local browser app for DXF, straight-extrusion STEP, loops JSON, and saved-project profiles.
- Continuous helix construction with full-profile and band modes; experimental flat laps with staggered perimeter bridges.
- Size, lap count, thickness, clearance, taper, base, and gap-ladder controls with contextual help.
- Preview generation separated from STL/project export. Save exports the stored preview geometry, preserving it when current controls change.
- Seven immediate preview controls for feature visibility, complete ramps, transparency, connection colors, and wireframe.
- Corrected profile orientation and initial camera fitting in the 3D viewer.
- Free-space checks, temporary-preview cleanup, preservation of earlier exports, and rollback of incomplete saves.
- Windows launcher and regression tests for geometry, import/export, preview behavior, and storage handling.
