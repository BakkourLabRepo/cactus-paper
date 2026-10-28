// src/experiment.js

import { initJsPsych } from "jspsych";
import { createTimeline } from "./experiment-setup.js";
import preload from "@jspsych/plugin-preload";
// Import everything needed for preloading from stimulus-helpers
import {
  setShapeImages,
  fullImgPaths,
  preloadImages,
} from "./stimulus-helpers.js";
import jsPsychExtensionMouseTracking from "@jspsych/extension-mouse-tracking";
// 1. Correctly define the preload trial
const preloadTrial = {
  type: preload,
  images: fullImgPaths, // Use the full array of 360 image paths
  // images: ['/assets/example_choice_trial.png'],
  // auto_preload: true,
  show_detailed_errors: true, // Corrected typo: "error" is singular
  on_start: () => {
    // Use on_load, which runs AFTER images are loaded
    const loadedImages = preloadImages();
    setShapeImages(loadedImages);
    console.log(loadedImages);
  },
  message: "<p>Loading experiment, please wait...</p>",
};

// 2. Initialize jsPsych
const jsPsych = initJsPsych({
  extensions: [
    {
      type: ExtensionMouseTracking,
      params: { minimum_sample_time: 50 },
    },
  ],
  on_finish: () => {
    jsPsych.data.displayData();
  },
});
// console.log(loadedImages)

// 3. Create the main timeline from your setup file
let timeline = createTimeline(jsPsych);

// 4. Add the preload trial to the very beginning of the main timeline
timeline.unshift(preloadTrial);

// 5. Run the entire timeline, not just the preload trial
jsPsych.run(timeline);
