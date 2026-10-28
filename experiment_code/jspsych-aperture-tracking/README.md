# jspsych-aperture-tracking

A jsPsych extension that combines a spotlight/aperture overlay with mouse tracking. Participants move their mouse to reveal trial content through a circular window, while all movements are recorded — producing a rich dataset of visual information search behavior.

Replaces the need to use `@jspsych/extension-mouse-tracking` separately; this extension includes equivalent mouse tracking functionality built in.

## Installation

```bash
npm install jspsych-aperture-tracking
```

Or copy `dist/index.js` into your project.

## Quick Start

```js
import ApertureTrackingExtension from "jspsych-aperture-tracking";

const jsPsych = initJsPsych({
  extensions: [
    {
      type: ApertureTrackingExtension,
      params: { minimum_sample_time: 50 }, // ms between samples
    },
  ],
});

const trial = {
  type: jsPsychHtmlButtonResponse,
  stimulus: '<div id="left">A</div><div id="right">B</div>',
  choices: ["Left", "Right"],
  extensions: [
    {
      type: ApertureTrackingExtension,
      params: {
        aperture_radius: 100, // px, clear area radius
        blur_amount: 20, // px, backdrop blur
        grayscale: true, // grayscale the obscured area
        overlay_opacity: 0.6, // darkness of the overlay
        edge_softness: 20, // px, gradient transition zone
        targets: ["#left", "#right"], // elements to record bounding boxes for
        events: ["mousemove"], // which events to track
      },
    },
  ],
};
```

## Parameters

### `initJsPsych` initialization params

| Parameter | Type | Default | Description |
|---|---|---|---|
| `minimum_sample_time` | number | `0` | Minimum ms between recorded mousemove samples. Set to `50` for ~20Hz sampling. |

### Per-trial params

| Parameter | Type | Default | Description |
|---|---|---|---|
| `aperture` | boolean | `true` | Enable/disable the aperture overlay. Set to `false` for mouse-tracking-only mode. |
| `aperture_radius` | number | `100` | Radius (px) of the clear viewing window. |
| `blur_amount` | number | `20` | Backdrop blur strength (px) outside the aperture. |
| `grayscale` | boolean | `true` | Apply grayscale filter to the obscured area. |
| `overlay_opacity` | number | `0.6` | Opacity of the dark overlay (0-1). |
| `edge_softness` | number | `20` | Width (px) of the gradient transition from clear to obscured. |
| `targets` | string[] | `[]` | CSS selectors for elements whose bounding boxes to record. |
| `events` | string[] | `["mousemove"]` | Mouse events to track: `"mousemove"`, `"mousedown"`, `"mouseup"`. |

## Output Data

Each trial produces:

- **`mouse_tracking_data`**: Array of `{x, y, t, event}` objects. `x`/`y` are viewport coordinates, `t` is ms since trial start.
- **`mouse_tracking_targets`**: Object mapping CSS selectors to their `getBoundingClientRect()` values at render time.

## Mouse-Tracking-Only Mode

Set `aperture: false` to disable the overlay and use this purely as a mouse tracker (drop-in replacement for `@jspsych/extension-mouse-tracking`):

```js
extensions: [{
  type: ApertureTrackingExtension,
  params: { aperture: false, targets: ["#stim"] }
}]
```

## Example

See `example/index.html` for a working demo.

## Building

```bash
bun install
bun run build
```
