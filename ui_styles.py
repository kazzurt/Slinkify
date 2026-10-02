"""Shared visual tokens and layout for Slinkify's local design workspace."""
import gradio as gr


APP_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=["Segoe UI", "system-ui", "sans-serif"],
).set(
    body_background_fill="#f6f6f4", body_background_fill_dark="#14171b",
    body_text_color="#242a31", body_text_color_dark="#e9edf1",
    body_text_color_subdued="#58636f", body_text_color_subdued_dark="#b2bdc9",
    background_fill_primary="#ffffff", background_fill_primary_dark="#1c2127",
    background_fill_secondary="#f0f1ee", background_fill_secondary_dark="#22282f",
    border_color_primary="#d8dce1", border_color_primary_dark="#3a424c",
    block_background_fill="transparent", block_background_fill_dark="transparent",
    block_border_width="0px", block_border_width_dark="0px",
    block_label_background_fill="transparent", block_label_background_fill_dark="transparent",
    block_label_border_width="0px", block_label_border_width_dark="0px",
    block_label_text_size="13px", block_label_text_weight="500",
    block_label_text_color="#3f4853", block_label_text_color_dark="#c6cfd9",
    block_info_text_color="#58636f", block_info_text_color_dark="#b2bdc9",
    panel_background_fill="transparent", panel_background_fill_dark="transparent",
    panel_border_width="0px", panel_border_width_dark="0px",
    input_background_fill="#ffffff", input_background_fill_dark="#1c2127",
    input_border_color="#cbd1d8", input_border_color_dark="#414b57",
    input_border_color_focus="#a74a0e", input_border_color_focus_dark="#e99b39",
    input_placeholder_color="#66717e", input_placeholder_color_dark="#b2bdc9",
    input_text_size="14px", input_radius="8px", input_padding="10px 12px",
    layout_gap="16px", form_gap_width="12px", block_padding="0px",
    button_large_radius="8px", button_large_text_size="14px",
    button_large_text_weight="600", button_large_padding="12px 18px",
    button_small_radius="8px", button_small_text_size="13px",
    button_primary_background_fill="#a74a0e", button_primary_background_fill_dark="#e99b39",
    button_primary_background_fill_hover="#873b0b", button_primary_background_fill_hover_dark="#f0ae55",
    button_primary_text_color="#ffffff", button_primary_text_color_dark="#1b2026",
    button_primary_border_color="#a74a0e", button_primary_border_color_dark="#e99b39",
    button_secondary_background_fill="#ffffff", button_secondary_background_fill_dark="#22282f",
    button_secondary_background_fill_hover="#ebeeef", button_secondary_background_fill_hover_dark="#303842",
    button_secondary_text_color="#3f4853", button_secondary_text_color_dark="#e9edf1",
    button_secondary_border_color="#cbd1d8", button_secondary_border_color_dark="#596574",
    checkbox_label_background_fill="transparent", checkbox_label_background_fill_dark="transparent",
    checkbox_label_border_color="#cbd1d8", checkbox_label_border_color_dark="#414b57",
    checkbox_label_background_fill_selected="#fff0e3", checkbox_label_background_fill_selected_dark="#362b1f",
    checkbox_label_border_color_selected="#a74a0e", checkbox_label_border_color_selected_dark="#e99b39",
    checkbox_label_padding="9px 12px", checkbox_label_text_size="13px",
    link_text_color="#9a4308", link_text_color_dark="#f0ae55",
    shadow_drop="none", shadow_inset="none", block_shadow="none", block_shadow_dark="none",
)


APP_CSS = """
.gradio-container {
    --sl-accent: #a74a0e;
    --sl-muted: #58636f;
    --sl-surface: #f0f1ee;
    --sl-border: #d8dce1;
    max-width: 1320px !important;
    margin: 0 auto;
    padding: clamp(18px, 2vw, 28px) clamp(14px, 2.2vw, 32px) 32px !important;
}
.gradio-container > .main { padding: 0 !important; }
.gradio-container .form { background: transparent; border: 0; gap: 12px; }
.dark .gradio-container {
    --sl-accent: #e99b39;
    --sl-muted: #b2bdc9;
    --sl-surface: #22282f;
    --sl-border: #3a424c;
}
.gradio-container ::selection { background: #ead0b6; color: #242a31; }
.gradio-container input, .gradio-container textarea { caret-color: var(--sl-accent); }
.gradio-container a { text-underline-offset: 3px; }
.gradio-container :is(button, input, select, textarea, [tabindex]):focus-visible {
    outline: 2px solid var(--sl-accent);
    outline-offset: 3px;
}
#slinkify-header { padding: 0 0 24px; border-bottom: 1px solid var(--sl-border); margin-bottom: 8px; }
#slinkify-header h1 { margin: 0 0 6px; font-size: 28px; line-height: 1.2; letter-spacing: -.02em; font-weight: 650; }
#slinkify-header p { margin: 0; color: var(--sl-muted); font-size: 15px; }
#slinkify-workspace { gap: 32px; align-items: flex-start; }
#slinkify-settings { gap: 24px; }
#slinkify-settings .sl-section { gap: 12px; }
#slinky-setting-profile { border: 1px solid var(--sl-border) !important; border-radius: 8px; padding: 12px; }
#slinky-setting-profile .file-preview-holder { border: 0; }
.sl-section { border: 0 !important; border-radius: 0 !important; background: transparent !important; padding: 0 !important; }
.sl-section-heading h2 { font-size: 16px; line-height: 1.4; font-weight: 600; margin: 0 0 2px; }
.sl-section-heading { padding-bottom: 2px; }
.sl-form-row { gap: 12px; }
#slinkify-preview-panel { gap: 12px; }
#slinkify-preview-heading h2 { margin: 0; font-size: 20px; font-weight: 600; line-height: 1.3; }
#slinkify-dimensions { font-size: 13px; color: var(--sl-muted); line-height: 1.65; }
#slinkify-dimensions p { margin: 0; }
#slinkify-dimensions strong { color: var(--body-text-color); font-weight: 600; font-variant-numeric: tabular-nums; }
#slinkify-ramp-summary { font-size: 12px; line-height: 1.5; color: var(--sl-muted); }
#slinkify-preview-note { font-size: 12px; line-height: 1.5; color: var(--sl-muted); }
#slinkify-preview-note p { margin: 0; }
#slinkify-generate-preview, #slinky-save-preview { min-height: 44px; }
.sl-disclosure { border: 1px solid var(--sl-border) !important; border-radius: 8px !important; background: transparent !important; }
.sl-disclosure > button { min-height: 42px; padding: 10px 12px !important; font-size: 13px !important; font-weight: 500; }
.sl-disclosure [data-testid="accordion-content"] { padding: 12px; }
#slinkify-build-report textarea { font-family: ui-monospace, Consolas, monospace; font-size: 12px; line-height: 1.6; }
#slinkify-profile-details { color: var(--sl-muted); font-size: 12px; overflow-wrap: anywhere; }
#slinkify-profile-details p { margin: 0 0 8px; }
#slinkify-settings input[type=number] { font-variant-numeric: tabular-nums; }
@media (min-width: 1100px) and (min-height: 850px) {
    #slinkify-preview-panel { position: sticky; top: 24px; }
}
@media (max-width: 899px) {
    #slinkify-workspace { flex-direction: column; gap: 28px; }
    #slinkify-preview-panel { order: -1; min-width: 0 !important; width: 100%; }
    #slinkify-settings { min-width: 0 !important; width: 100%; }
    #slinkify-header { padding-bottom: 18px; }
}
@media (max-width: 480px) {
    .sl-form-row { gap: 10px; }
    #slinkify-header h1 { font-size: 25px; }
}
"""
