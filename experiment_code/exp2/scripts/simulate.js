// 1. Import necessary modules
import { initJsPsych } from "jspsych";
import { createTimeline } from "../src/experiment-setup.js";
import fs from "fs"; // Node.js module to save files
import jsPsychExtensionMouseTracking from "@jspsych/extension-mouse-tracking";
// 2. Initialize jsPsych in Node.js (no display_element needed)
const jsPsych = initJsPsych({
  //   extensions: [
  //   {
  //     type: jsPsychExtensionMouseTracking,
  //     params: { minimum_sample_time: 50 },
  //   },
  // ],
});

const runSimulation = async (timeline) => {
  console.log("Starting simulation...");
  console.log(`Timeline has ${timeline.length} items`);

  try {
    await jsPsych.simulate(timeline, "data-only");
    console.log("Simulation completed!");
    console.log(`Total trials collected: ${jsPsych.data.get().count()}`);

    // Save data to a CSV file
    const csvData = jsPsych.data.get().csv();
    fs.writeFileSync("simulation_data.csv", csvData);
    console.log("Simulation data saved to simulation_data.csv");
  } catch (error) {
    console.error("Error during simulation:", error);
    throw error;
  }
};

console.log("Creating timeline...");
let timeline = createTimeline(jsPsych);
console.log("Timeline created, starting simulation...");
await runSimulation(timeline);
console.log("Done!");
