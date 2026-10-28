// =======================================================================
// INTEGRATION EXAMPLE: How to use trial-generation.js in experiment-setup.js
// =======================================================================
//
// Add this import at the top of experiment-setup.js:
//
//   import { generateChoiceTrials, addDiscriminabilityMetadata } from "./trial-generation.js";
//
// Then replace the current choice trial generation section (lines ~240-268) with:
// =======================================================================

// === CHOICE TRIALS (CONTROLLED DIVERGENT/NON-DIVERGENT) ===

// Target proportion of divergent trials (35-40% as per pre-registration)
const TARGET_DIVERGENT_PROPORTION = 0.38;

// Generate controlled choice trials
let choiceArr = generateChoiceTrials(
  EXPERIMENT_PARAMS.nChoiceTrials,      // 100 trials
  TARGET_DIVERGENT_PROPORTION,           // ~38 divergent, ~62 non-divergent
  EXPERIMENT_PARAMS.values,              // {shape: [...], color: [...], line: [...]}
  dims,                                  // Randomized dimension order for this participant
  EXPERIMENT_PARAMS.categories,          // Category boundaries
  (arr, n) => jsPsych.randomization.sampleWithoutReplacement(arr, n),  // Sampling function
  (arr) => jsPsych.randomization.shuffle(arr)                          // Shuffle function
);

// Add discriminability metadata for analysis
choiceArr = addDiscriminabilityMetadata(choiceArr, dims);

// Log trial composition for verification
const nDivergent = choiceArr.filter(t => t.trialType === 'divergent').length;
const nNonDivergent = choiceArr.filter(t => t.trialType === 'non-divergent').length;
console.log(`Generated ${nDivergent} divergent trials (${(nDivergent/choiceArr.length*100).toFixed(1)}%)`);
console.log(`Generated ${nNonDivergent} non-divergent trials (${(nNonDivergent/choiceArr.length*100).toFixed(1)}%)`);

// === END REPLACEMENT SECTION ===


// =======================================================================
// ALSO UPDATE: The on_finish handler to save the new metadata
// =======================================================================
// In your choiceTrial object, update the on_finish to include trial type metadata:

const choiceTrial = {
  type: canvasButtonResponse,
  canvas_size: [800, 860],
  stimulus: (canvas) => {
    const parent = canvas.parentNode;
    const container = document.createElement('div');
    container.id = 'stimulus-container';
    container.style.position = 'relative';
    parent.insertBefore(container, canvas);
    container.appendChild(canvas);
    drawChoiceTrial(canvas);
  },
  choices: ["left", "right"],
  prompt: jsPsych.timelineVariable("prompt"),
  trial_duration: EXPERIMENT_PARAMS.choiceMaxDuration,
  on_finish: () => {
    stopSpotlightEffect();
    
    let trialAttributes = {};
    let trialObject = {};
    trialObject.left = jsPsych.timelineVariable("left");
    trialObject.right = jsPsych.timelineVariable("right");
    
    // Existing attribute logging
    for (let side in trialObject) {
      let total_rew = 0;
      for (let position in trialObject[side]) {
        for (let attribute in trialObject[side][position]) {
          let key = `choice_trial_${side}_${position}_${attribute}`;
          trialAttributes[key] = trialObject[side][position][attribute];
          if (attribute == "value") {
            total_rew = total_rew + trialObject[side][position][attribute];
          }
        }
      }
      let key = `choice_trial_${side}_total_rew`;
      trialAttributes[key] = total_rew;
    }
    
    // NEW: Add trial type metadata for analysis
    trialAttributes.trial_type = jsPsych.timelineVariable("trialType");
    trialAttributes.categorical_winner = jsPsych.timelineVariable("categoricalWinner");
    trialAttributes.relational_winner = jsPsych.timelineVariable("relationalWinner");
    trialAttributes.critical_dim = jsPsych.timelineVariable("criticalDim");
    trialAttributes.left_sum = jsPsych.timelineVariable("leftSum");
    trialAttributes.right_sum = jsPsych.timelineVariable("rightSum");
    trialAttributes.left_wins = jsPsych.timelineVariable("leftWins");
    trialAttributes.right_wins = jsPsych.timelineVariable("rightWins");
    
    // NEW: Add discriminability data
    const discrim = jsPsych.timelineVariable("discriminability");
    if (discrim) {
      for (let dim in discrim) {
        trialAttributes[`discriminability_${dim}`] = discrim[dim];
      }
    }
    trialAttributes.critical_dim_discriminability = jsPsych.timelineVariable("criticalDimDiscriminability");
    
    jsPsych.data.get().addToLast(trialAttributes);
  },
  on_load: startSpotlightEffect,
  extensions: [{ 
    type: jsPsychExtensionMouseTracking, 
    params: {targets: ['#left-top','#left-mid','#left-bot','#right-top','#right-mid','#right-bot']}
  }],
};


// =======================================================================
// IMPORTANT: Update config.js categories array
// =======================================================================
// Your current config has 6 categories but only 5 values per dimension.
// The categories array should have 5 entries to match:
//
// categories: [
//   { min_val: 1, max_val: 45 },    // Category 0: 0 or 100 points
//   { min_val: 46, max_val: 90 },   // Category 1: 25 or 75 points  
//   { min_val: 91, max_val: 135 },  // Category 2: 50 points
//   { min_val: 136, max_val: 180 }, // Category 3: 75 or 25 points
//   { min_val: 181, max_val: 230 }, // Category 4: 100 or 0 points
// ],
//
// Adjust the boundaries to match your intended category structure.
// The current array has 6 entries with the 5th being a duplicate of the 4th.


// =======================================================================
// VERIFICATION: Quick sanity check you can run in console
// =======================================================================

const verifyTrialGeneration = (trials) => {
  console.log("=== TRIAL GENERATION VERIFICATION ===");
  
  // Count trial types
  const types = { divergent: 0, 'non-divergent': 0, tie: 0 };
  trials.forEach(t => types[t.trialType]++);
  console.log("Trial type counts:", types);
  
  // Verify divergent trials are actually divergent
  const divergentTrials = trials.filter(t => t.trialType === 'divergent');
  const actuallyDivergent = divergentTrials.filter(t => 
    t.categoricalWinner !== t.relationalWinner
  );
  console.log(`Divergent trials verified: ${actuallyDivergent.length}/${divergentTrials.length}`);
  
  // Check value ranges
  const allValues = trials.flatMap(t => [t.leftSum, t.rightSum]);
  console.log(`Value sum range: ${Math.min(...allValues)} - ${Math.max(...allValues)}`);
  
  // Check discriminability ranges
  const allDiscrim = trials.flatMap(t => Object.values(t.discriminability || {}));
  console.log(`Discriminability range: ${Math.min(...allDiscrim)} - ${Math.max(...allDiscrim)}`);
  
  // Sample a divergent trial
  if (divergentTrials.length > 0) {
    const sample = divergentTrials[0];
    console.log("\nSample divergent trial:");
    console.log(`  Categorical winner: ${sample.categoricalWinner} (sum: L=${sample.leftSum}, R=${sample.rightSum})`);
    console.log(`  Relational winner: ${sample.relationalWinner} (wins: L=${sample.leftWins}, R=${sample.rightWins})`);
    console.log(`  Critical dimension: ${sample.criticalDim}`);
  }
  
  return types;
};

// Call after generating trials:
verifyTrialGeneration(choiceArr);
