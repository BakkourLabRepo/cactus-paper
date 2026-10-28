import fs from "fs/promises";

const filePath = "./dist/experiment.js";

console.log(`Starting post-build modifications for ${filePath}...`);

// Read the built file
let content = await fs.readFile(filePath, "utf-8");

// 1. Define the replacements as an array of [find, replace] pairs.
// This is the correct way to preserve RegExp objects.
const replacements = [
  // === Remove import statements ===
  [new RegExp('^import\\s+.*?\\s+from\\s+.*?;\s*$', "gm"), ""],

  // === Correct Plugin Types ===  
  [/jsPsych2/g, "jsPsych"],
  [
    /type:\s*([a-zA-Z]+),/g,
    (match, pluginName) => {
      if (pluginName.startsWith('jsPsych')) {
        return match;
      }
      const capitalized = pluginName.charAt(0).toUpperCase() + pluginName.slice(1);
      return `type: jsPsych${capitalized},`;
    }
  ],
  [/= lab\(L/g, "= d3.lab\(L"],
  [/= rgb/g, "= d3.rgb"],
  [
    /type: jsPsychExtensionMouseTracking2/g,
    "type: jsPsychExtensionMouseTracking",
  ],
  [/ParameterType2/g, "ParameterType"],
  [/ParameterType3/g, "ParameterType"],
  [new RegExp("(['\"`])/assets/", "g"), "$1"],
  [
    /type: jsPsychExtensionMouseTracking/g,
    "type: MouseTrackingExtension",
  ],
  [
    /type: jsPsychMouseTrackingExtension/g,
    "type: MouseTrackingExtension",
  ],
];

// 2. Apply all replacements using a simple for...of loop
for (const [find, replaceWith] of replacements) {
  content = content.replace(find, replaceWith);
}

// 3. Clean up extra newlines that might be left from removing imports
content = content.replace(/(\r\n|\n\r|\r|\n){3,}/g, "\n\n");

// 4. Write the modified content back to the file
await fs.writeFile(filePath, content, "utf-8");

console.log("Post-build script finished successfully!");
