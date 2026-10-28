import { ParameterType } from "jspsych";

const version = "0.1.0";

/**
 * jsPsych Aperture Tracking Extension
 *
 * Creates a spotlight/aperture overlay that obscures the trial display.
 * Participants must move their mouse to reveal content through a circular
 * window that follows the cursor. All mouse movements are recorded,
 * producing a rich dataset of visual attention/information search behavior.
 *
 * Usage:
 *   const jsPsych = initJsPsych({
 *     extensions: [{ type: ApertureTrackingExtension, params: { minimum_sample_time: 50 } }]
 *   });
 *
 *   const trial = {
 *     type: jsPsychSomePlugin,
 *     extensions: [{
 *       type: ApertureTrackingExtension,
 *       params: {
 *         aperture_radius: 100,
 *         blur_amount: 20,
 *         grayscale: true,
 *         overlay_opacity: 0.6,
 *         edge_softness: 20,
 *         targets: ["#stimulus-1", "#stimulus-2"],
 *         events: ["mousemove", "mousedown", "mouseup"]
 *       }
 *     }]
 *   };
 */
class ApertureTrackingExtension {
  constructor(jsPsych) {
    this.jsPsych = jsPsych;

    this.initialize = async ({ minimum_sample_time = 0 } = {}) => {
      this.domObserver = new MutationObserver(this.mutationObserverCallback);
      this.minimumSampleTime = minimum_sample_time;
    };

    this.on_start = (params) => {
      params = params || {};

      // Mouse tracking state
      this.currentTrialData = [];
      this.currentTrialTargets = new Map();
      this.currentTrialSelectors = params.targets || [];
      this.lastSampleTime = null;
      this.eventsToTrack = params.events || ["mousemove"];

      // Aperture params (with defaults)
      this.apertureRadius = params.aperture_radius ?? 100;
      this.blurAmount = params.blur_amount ?? 20;
      this.grayscale = params.grayscale ?? true;
      this.overlayOpacity = params.overlay_opacity ?? 0.6;
      this.edgeSoftness = params.edge_softness ?? 20;
      this.apertureEnabled = params.aperture ?? true;

      this.domObserver.observe(this.jsPsych.getDisplayElement(), {
        childList: true,
      });
    };

    this.on_load = () => {
      this.currentTrialStartTime = performance.now();

      // Start aperture overlay
      if (this.apertureEnabled) {
        this._startAperture();
      }

      // Bind mouse event listeners
      if (this.eventsToTrack.includes("mousemove")) {
        window.addEventListener("mousemove", this._mouseMoveHandler);
      }
      if (this.eventsToTrack.includes("mousedown")) {
        window.addEventListener("mousedown", this._mouseDownHandler);
      }
      if (this.eventsToTrack.includes("mouseup")) {
        window.addEventListener("mouseup", this._mouseUpHandler);
      }
    };

    this.on_finish = () => {
      // Clean up aperture
      if (this.apertureEnabled) {
        this._stopAperture();
      }

      // Clean up observers and listeners
      this.domObserver.disconnect();
      if (this.eventsToTrack.includes("mousemove")) {
        window.removeEventListener("mousemove", this._mouseMoveHandler);
      }
      if (this.eventsToTrack.includes("mousedown")) {
        window.removeEventListener("mousedown", this._mouseDownHandler);
      }
      if (this.eventsToTrack.includes("mouseup")) {
        window.removeEventListener("mouseup", this._mouseUpHandler);
      }

      return {
        mouse_tracking_data: this.currentTrialData,
        mouse_tracking_targets: Object.fromEntries(
          this.currentTrialTargets.entries()
        ),
      };
    };

    // --- Aperture management ---

    this._startAperture = () => {
      const innerRadius = this.apertureRadius;
      const outerRadius = this.apertureRadius + this.edgeSoftness;

      const backdropFilters = [`blur(${this.blurAmount}px)`];
      if (this.grayscale) {
        backdropFilters.push("grayscale(100%)");
      }

      const spotlight = document.createElement("div");
      spotlight.id = "jspsych-aperture-overlay";
      Object.assign(spotlight.style, {
        position: "fixed",
        top: "0",
        left: "0",
        width: "100vw",
        height: "100vh",
        backgroundColor: `rgba(0, 0, 0, ${this.overlayOpacity})`,
        backdropFilter: backdropFilters.join(" "),
        webkitBackdropFilter: backdropFilters.join(" "),
        mask: `radial-gradient(circle at var(--aperture-x) var(--aperture-y), transparent ${innerRadius}px, black ${outerRadius}px)`,
        webkitMask: `radial-gradient(circle at var(--aperture-x) var(--aperture-y), transparent ${innerRadius}px, black ${outerRadius}px)`,
        zIndex: "9999",
        pointerEvents: "none",
      });

      document.body.appendChild(spotlight);
      this._spotlightEl = spotlight;

      // Separate listener for the aperture visual (always tracks mousemove for the overlay)
      this._apertureMouseMoveHandler = (e) => {
        spotlight.style.setProperty("--aperture-x", e.clientX + "px");
        spotlight.style.setProperty("--aperture-y", e.clientY + "px");
      };
      document.addEventListener("mousemove", this._apertureMouseMoveHandler);
    };

    this._stopAperture = () => {
      if (this._spotlightEl) {
        this._spotlightEl.remove();
        this._spotlightEl = null;
      }
      if (this._apertureMouseMoveHandler) {
        document.removeEventListener(
          "mousemove",
          this._apertureMouseMoveHandler
        );
        this._apertureMouseMoveHandler = null;
      }
    };

    // --- Mouse tracking handlers ---

    this._mouseMoveHandler = ({ clientX: x, clientY: y }) => {
      const event_time = performance.now();
      const t = Math.round(event_time - this.currentTrialStartTime);
      if (
        this.lastSampleTime === null ||
        event_time - this.lastSampleTime >= this.minimumSampleTime
      ) {
        this.lastSampleTime = event_time;
        this.currentTrialData.push({ x, y, t, event: "mousemove" });
      }
    };

    this._mouseUpHandler = ({ clientX: x, clientY: y }) => {
      const event_time = performance.now();
      const t = Math.round(event_time - this.currentTrialStartTime);
      this.currentTrialData.push({ x, y, t, event: "mouseup" });
    };

    this._mouseDownHandler = ({ clientX: x, clientY: y }) => {
      const event_time = performance.now();
      const t = Math.round(event_time - this.currentTrialStartTime);
      this.currentTrialData.push({ x, y, t, event: "mousedown" });
    };

    // --- DOM mutation observer for target tracking ---

    this.mutationObserverCallback = (mutationsList, observer) => {
      for (const selector of this.currentTrialSelectors) {
        if (!this.currentTrialTargets.has(selector)) {
          const target =
            this.jsPsych.getDisplayElement().querySelector(selector);
          if (target) {
            this.currentTrialTargets.set(
              selector,
              target.getBoundingClientRect()
            );
          }
        }
      }
    };
  }

  static info = {
    name: "aperture-tracking",
    version,
    data: {
      /** Array of {x, y, t, event} objects recording mouse position over time. */
      mouse_tracking_data: {
        type: ParameterType.COMPLEX,
        array: true,
        nested: {
          x: { type: ParameterType.INT },
          y: { type: ParameterType.INT },
          t: { type: ParameterType.INT },
          event: { type: ParameterType.STRING },
        },
      },
      /** Bounding rectangles of target elements specified via the `targets` parameter. */
      mouse_tracking_targets: {
        type: ParameterType.COMPLEX,
        nested: {
          x: { type: ParameterType.INT },
          y: { type: ParameterType.INT },
          width: { type: ParameterType.INT },
          height: { type: ParameterType.INT },
          top: { type: ParameterType.INT },
          bottom: { type: ParameterType.INT },
          left: { type: ParameterType.INT },
          right: { type: ParameterType.INT },
        },
      },
    },
  };
}

export default ApertureTrackingExtension;
