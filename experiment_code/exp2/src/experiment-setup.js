import { EXPERIMENT_PARAMS } from "./config.js";
import {
  setShapeImages,
  fullImgPaths,
  categoryToNumString,
  arrayRange,
  drawChoiceTrial,
  drawTrainingTrial,
  // drawCategorizationTrial,
  drawPreExpStim,
  preloadImages,
} from "./stimulus-helpers.js";
import {
  introInstructions,
  preCategoryInstructions,
  preChoiceInstructions1,
  preChoiceInstructions2,
  postCat1Instructions,
} from "./instructions.js";
import { generateChoiceTrials, addDiscriminabilityMetadata } from "./trial-generation.js";

import htmlKeyboardResponse from "@jspsych/plugin-html-keyboard-response";
import htmlButtonResponse from "@jspsych/plugin-html-button-response";
import canvasButtonResponse from "@jspsych/plugin-canvas-button-response";
import canvasKeyboardResponse from "@jspsych/plugin-canvas-keyboard-response";
import jsPsychExtensionMouseTracking from "@jspsych/extension-mouse-tracking";
import freeSort from '@jspsych/plugin-free-sort';
import fullscreen from '@jspsych/plugin-fullscreen';
import surveyText from '@jspsych/plugin-survey-text';
import jsPsychSurvey from '@jspsych/plugin-survey';

export const createTimeline = (jsPsych, plugins) => {
  // value setup ------------

  // --- Spotlight Helper Functions (Final Version) ---

  // Define the handler in a scope accessible to both start and stop functions
  const spotlightMouseMoveHandler = (e) => {
    // Use a try-catch block as a safeguard in case the element is already gone
    try {
      document
        .getElementById("spotlight")
        .style.setProperty("--x", e.clientX + "px");
      document
        .getElementById("spotlight")
        .style.setProperty("--y", e.clientY + "px");
    } catch (e) {}
  };

  // Function to START the spotlight effect
  const startSpotlightEffect = () => {
    // Create the spotlight div
    const spotlight = document.createElement("div");
    spotlight.id = "spotlight";

    // Apply all necessary styles directly using JavaScript
    Object.assign(spotlight.style, {
      position: "fixed",
      top: "0",
      left: "0",
      width: "100vw",
      height: "100vh",
      backgroundColor: "rgba(0, 0, 0, 0.6)",
      backdropFilter: "blur(20px) grayscale(100%)",
      webkitBackdropFilter: "blur(10px) grayscale(100%)",
      mask: "radial-gradient(circle at var(--x) var(--y), transparent 100px, black 120px)",
      webkitMask:
        "radial-gradient(circle at var(--x) var(--y), transparent 100px, black 120px)",
      zIndex: "9999",
      pointerEvents: "none",
      cursor: "none",
    });

    // Append to the body
    document.body.appendChild(spotlight);

    // Add the event listener that updates the position
    document.addEventListener("mousemove", spotlightMouseMoveHandler);
  }

  // Function to STOP the spotlight effect
  const stopSpotlightEffect = () => {
    const spotlight = document.querySelector("#spotlight");
    if (spotlight) {
      spotlight.remove();
    }
    // Remove the event listener
    document.removeEventListener("mousemove", spotlightMouseMoveHandler);
  }
  const preExposureVals = {
    shapeTrial:
      EXPERIMENT_PARAMS.values.shape[0] === 0
        ? arrayRange(1, 231, 1)
        : arrayRange(231, 1, -1),
    colorTrial:
      EXPERIMENT_PARAMS.values.color[0] === 0
        ? arrayRange(1, 231, 1)
        : arrayRange(231, 1, -1),
    lineTrial:
      EXPERIMENT_PARAMS.values.line[0] === 0
        ? arrayRange(1, 231, 1)
        : arrayRange(231, 1, -1),
  }; // we check to see if the initial value is 20 and if so we go 1->181, if not we reverse it

  const angleDimToVal = (angle, dim) => {
    for (let i = 0; i < EXPERIMENT_PARAMS.categories.length; i++) {
      if (angle > EXPERIMENT_PARAMS.categories[i].max_val) {
        continue;
      } else {
        return EXPERIMENT_PARAMS.values[dim][i];
      }
    }
  };

  const dims = jsPsych.randomization.shuffle(['shape','color','line'])
  
  // preloading shapes -------

  function setBackgroundColor(color) {
    document.body.style.backgroundColor = color;
  }

  // Block setup --------------

  // learning trials, right now set up so all three are the same and instead I should turn this into a function and get 3 such arrays
  // I also need to add the prompt randomization. 

  
  let learning_pairings = [
    [ 0, 1 ],
    [ 0, 2 ],
    [ 0, 3 ],
    [ 0, 4 ],
    [ 1, 2 ],
    [ 1, 3 ],
    [ 1, 4 ],
    [ 2, 3 ],
    [ 2, 4 ],
    [ 3, 4 ],
  ]

  const generateLearningBlock = (bigFirst) => {
      let trainingPrompts
      if (bigFirst === 1) {
        trainingPrompts = [...Array(EXPERIMENT_PARAMS.nPerPrompt).fill('bigger'), ...Array(EXPERIMENT_PARAMS.nPerPrompt).fill('smaller'), ...Array(EXPERIMENT_PARAMS.nPerPrompt).fill('bigger'), ...Array(EXPERIMENT_PARAMS.nPerPrompt).fill('smaller')]
      } else {
        trainingPrompts = [...Array(EXPERIMENT_PARAMS.nPerPrompt).fill('smaller'), ...Array(EXPERIMENT_PARAMS.nPerPrompt).fill('bigger'), ...Array(EXPERIMENT_PARAMS.nPerPrompt).fill('smaller'), ...Array(EXPERIMENT_PARAMS.nPerPrompt).fill('bigger')]
      }
      var initial_blocks = [];
      for (let i = 0; i < learning_pairings.length; i++) {
        for (let j = 0; j < EXPERIMENT_PARAMS.nPairingsForLearning; j++){
          let current_pair = jsPsych.randomization.shuffle(learning_pairings[i])
          let [trial_left, trial_right] = current_pair; 
          let tempTrialObj = {
            left: categoryToNumString(EXPERIMENT_PARAMS.categories, trial_left),
            left_category: trial_left,
            right: categoryToNumString(EXPERIMENT_PARAMS.categories, trial_right),
            right_category: trial_right,
          };
          initial_blocks.push(tempTrialObj);
      }}
      initial_blocks = jsPsych.randomization.shuffle(initial_blocks)

      initial_blocks = initial_blocks.map((trial, index) => {
        trial.prompt_dir = trainingPrompts[index]
        return trial
      })
      return initial_blocks
  }

  let colorNumsForLearning = generateLearningBlock(jsPsych.randomization.sampleBernoulli(0.5))
  let shapeNumsForLearning = generateLearningBlock(jsPsych.randomization.sampleBernoulli(0.5))
  let lineNumsForLearning = generateLearningBlock(jsPsych.randomization.sampleBernoulli(0.5))

  const zip = (...arrays) => {
    const minLength = Math.min(...arrays.map((arr) => arr.length));
    return Array.from({ length: minLength }, (_, i) =>
      arrays.map((arr) => arr[i]),
    );
  };
  // choice trials
  // currently the choice dims are always going to be in a set order from trial to trial, but randomized across ppts
  const colorDimArr = Array(EXPERIMENT_PARAMS.nChoiceTrials).fill("color");
  const shapeDimArr = Array(EXPERIMENT_PARAMS.nChoiceTrials).fill("shape");
  const lineDimArr = Array(EXPERIMENT_PARAMS.nChoiceTrials).fill("line");

  let choiceDims = zip(colorDimArr, shapeDimArr, lineDimArr);

  choiceDims = choiceDims.map((el) => jsPsych.randomization.shuffle(el));

  // category trials

  const numbers = [...Array(20).keys()]
  let freeSortNumbersForShape = [1 , 11 , 21 , 31 , 51 , 61 , 71 , 81 , 101 , 111 , 121 , 131 , 151 , 161 , 171 , 181, 201, 211, 221, 231]
  freeSortNumbersForShape = freeSortNumbersForShape.map(num => String(num).padStart(3,'0'))
  const freeSortStim = {
    'shape': freeSortNumbersForShape.map(num => `VCS_${num}_transparent_cv2.png`),
    'line': numbers.map(num => `frame_line_${num}.png`),
    'color': numbers.map(num => `frame_color_${num}.png`),
  }
  const makeFreeSort = (trial_type, prompt_type) => {
    let size = 80
    return {
      type: freeSort,
      stimuli: freeSortStim[trial_type],
      stim_width: size,
      stim_height: size,
      sort_area_width: 1000,
      sort_area_height: 1000,
      sort_area_shape: "square",
      change_border_background_color: false,
      // column_spread_factor: 0.8,
      stim_starts_inside: true,
      trial_duration: EXPERIMENT_PARAMS.freeSortMaxDuration,
      prompt: () => {
        if (prompt_type === 'perceptual') {
         return "<p>Click and drag the images below to sort them so that items that <b>look</b> similar are close together.</p>"
        } else {
         return "<p>Click and drag the images below to sort them so that items that are similar in <b>value</b> are close together.</p>" 
        }
    }
    };
  };

  // === CHOICE TRIALS (CONTROLLED DIVERGENT/NON-DIVERGENT) ===

  // Target proportion of divergent trials (50% as per pre-registration)
  const TARGET_DIVERGENT_PROPORTION = 0.5;

  // Generate controlled choice trials
  let choiceArr = generateChoiceTrials(
    EXPERIMENT_PARAMS.nChoiceTrials,      // 120 trials
    TARGET_DIVERGENT_PROPORTION,           // ~60 divergent, ~60 non-divergent
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

  let trainingChoiceArr = []
  let sought_obj = {
      dim:'color',
      angle: 290,
      value: 'NA'
  };
  let filler_obj = {
    dim: 'color',
    angle: 340,
    value: 'NA'
  }

  const createTrainingChoiceTrial = (side,position) => {
  let choiceTrainingTrialTemplate = {
    left: {
      top: filler_obj,
      mid: filler_obj,
      bot: filler_obj,
    },
    right: {
      top: filler_obj,
      mid: filler_obj,
      bot: filler_obj,      
    },
    prompt: "<p>Which side has the purple?</p>"
  }
  choiceTrainingTrialTemplate[side][position] = sought_obj
  return choiceTrainingTrialTemplate
  }


  for (const side of ['left','right']) {
    for (const position of ['top','mid','bot']) {
      trainingChoiceArr.push(createTrainingChoiceTrial(side,position))
    }
  }



  // --------------------
  const assignFeedback = (last_trial_data) => { // I need to fix this so that it tracks whether the selection matched the prompt 
                                                // (so if the prompt is "bigger" then we need to see if the response is bigger)
    let response = last_trial_data.response;
    let prompt = last_trial_data.prompt_dir; 
    let { left_category, right_category, dim } = last_trial_data;
    let trial_values = EXPERIMENT_PARAMS.values[dim];
    let feedbackObj = {
      'choice': response,
      'left_val': trial_values[left_category],
      'right_val': trial_values[right_category],
    }
    const leftBigger = trial_values[left_category] > trial_values[right_category]
    const targetLeft = (prompt ==='bigger') ? leftBigger: !leftBigger
    const correctResponseKey = targetLeft ? 'f':'j'
    
    feedbackObj.correct = (response === correctResponseKey) 
    feedbackObj.timeout = (response != 'f' && response != 'j')
    feedbackObj.left_correct = (response === 'f' && feedbackObj.correct)
    feedbackObj.right_correct = (response === 'j' && feedbackObj.correct)
    return feedbackObj
  };

  // --- TIMELINE ---
  const enter_fullscreen = {
    type: fullscreen,
    fullscreen_mode: true
  }

  const fixation = {
    type: htmlKeyboardResponse,
    stimulus: '<div style="font-size: 100px">+</div>',
    response_ends_trial: false,
    choices: "NO_KEYS",
    trial_duration: EXPERIMENT_PARAMS.fixationDuration,
  };
  const choice_fixation = {
    type: htmlButtonResponse,
    stimulus: "",
    // response_ends_trial: false,
    choices: [" "],
    button_html: ['<div>Click Here</div>'],
    // trial_duration: EXPERIMENT_PARAMS.fixationDuration,
    // on_start: () => {
    //   document.body.style.backgroundColor = "black";
    // },
    // on_finish: () => {
    //   document.body.style.backgroundColor = "white";
    // },
  };

  // preExposure drawing functions

  // This factory function builds the animation trial.
  // It takes the specific drawing function we want to use.
  const createPreExposureTrial = (trial_type) => {
    return {
      type: canvasKeyboardResponse,
      stimulus: (c) => {
        const ctx = c.getContext("2d");
        let frameIndex = 0;
        let preExpArr = preExposureVals[trial_type];
        const frameDuration = 30; // e.g., 30ms per frame
        let lastFrameTime = performance.now();
        let dim = trial_type.replace("Trial", "");

        const animationLoop = (currentTime) => {
          // Stop the animation after 180 frames
          if (frameIndex >= preExpArr.length) {
            jsPsych.finishTrial();
            return;
          }

          // Use requestAnimationFrame to loop
          requestAnimationFrame(animationLoop);

          // Check if enough time has passed to show the next frame
          if (currentTime - lastFrameTime >= frameDuration) {
            lastFrameTime = currentTime;

            drawPreExpStim(dim, ctx, c, preExpArr[frameIndex]);

            frameIndex++;
          }
        };

        // Start the animation loop
        requestAnimationFrame(animationLoop);
      },
      choices: "NO_KEYS",
    };
  };

  const shapePreExposureTrial = createPreExposureTrial("shapeTrial");
  const colorPreExposureTrial = createPreExposureTrial("colorTrial");
  const linePreExposureTrial = createPreExposureTrial("lineTrial");

  shapePreExposureTrial.data = { pre_exposure_type: "shape" };
  colorPreExposureTrial.data = { pre_exposure_type: "color" };
  linePreExposureTrial.data = { pre_exposure_type: "line" };

  // // drawing functions for the training trials

  // factory function for the training trials
  const createTrainingTrial = (drawingFunction, isFeedback, trial_type) => {
    let dim = trial_type.replace("Trial", "");
    return {
      type: canvasKeyboardResponse,
      canvas_size: [360, 860],
      stimulus: (canvas) => {
        const ctx = canvas.getContext("2d");

        let left_val = parseInt(jsPsych.timelineVariable("left"));
        let right_val = parseInt(jsPsych.timelineVariable("right"));
        if (isFeedback) {
          let last_trial_data = jsPsych.data.get().last().values()[0];
          let feedback = assignFeedback(last_trial_data);

          drawingFunction(canvas, dim, left_val, right_val, feedback);
        } else {
          let feedback = {
            val: null,
            left_val: null,
            right_val: null,
          };
          drawingFunction(canvas, dim, left_val, right_val, feedback);
        }
      },
      choices: () => {
        if (isFeedback) {
          return ["NO_KEYS"];
        } else {
          return ["f", "j"];
        }
      },
      trial_duration: () => {
        if (isFeedback) {
          return EXPERIMENT_PARAMS.feedbackDuration;
        } else {
          return EXPERIMENT_PARAMS.trialDuration;
        }
      },
      prompt: () => {
        if (jsPsych.timelineVariable('prompt_dir') === 'bigger') {
          return "<p>Which is higher value?</p>"
        } else {
          return "<p>Which is lower value?</p>" 
        }
      },
      data: () => {
        let left_rew =
          EXPERIMENT_PARAMS.values[dim][
            jsPsych.timelineVariable("left_category")
          ];
        let right_rew =
          EXPERIMENT_PARAMS.values[dim][
            jsPsych.timelineVariable("right_category")
          ];
        return {
          task: "learning",
          dim: dim,
          left_choice: jsPsych.timelineVariable("left"),
          left_category: jsPsych.timelineVariable("left_category"),
          left_rew: left_rew,
          right_choice: jsPsych.timelineVariable("right"),
          right_category: jsPsych.timelineVariable("right_category"),
          right_rew: right_rew,
          prompt_dir: jsPsych.timelineVariable("prompt_dir")
        };
      },
    };
  };

  const shapeTrial = createTrainingTrial(
    drawTrainingTrial,
    false,
    "shapeTrial",
  );
  const shapeTrialFeedback = createTrainingTrial(
    drawTrainingTrial,
    true,
    "shapeTrial",
  );

  const colorTrial = createTrainingTrial(
    drawTrainingTrial,
    false,
    "colorTrial",
  );
  const colorTrialFeedback = createTrainingTrial(
    drawTrainingTrial,
    true,
    "colorTrial",
  );

  const lineTrial = createTrainingTrial(drawTrainingTrial, false, "lineTrial");
  const lineTrialFeedback = createTrainingTrial(
    drawTrainingTrial,
    true,
    "lineTrial",
  );

  const shapeTrials = {
    timeline: [shapeTrial, shapeTrialFeedback, fixation],
    timeline_variables: shapeNumsForLearning,
  };

  const colorTrials = {
    timeline: [colorTrial, colorTrialFeedback, fixation],
    timeline_variables: colorNumsForLearning,
  };

  const lineTrials = {
    timeline: [lineTrial, lineTrialFeedback, fixation],
    timeline_variables: lineNumsForLearning,
  };

  // Categorization trials

  const PreExpChoiceTrial = {
    type: canvasButtonResponse,
    canvas_size: [800, 860],
    stimulus: (canvas) => {
    const parent = canvas.parentNode;

    // Create our new container div
    const container = document.createElement('div');
    container.id = 'stimulus-container';
    container.style.position = 'relative';

    // Insert the container into the DOM right before the canvas
    parent.insertBefore(container, canvas);

    // Move the canvas inside our new container
    container.appendChild(canvas);

    // Now that the canvas is wrapped, call your drawing function
    drawChoiceTrial(canvas);
    },
    choices: ["left", "right"],
    prompt: jsPsych.timelineVariable("prompt"),
    trial_duration: EXPERIMENT_PARAMS.covertAttentionDuration,
    on_finish: () => {
      let trialAttributes = {};
      let trialObject = {};
      trialObject.left = jsPsych.timelineVariable("left");
      trialObject.right = jsPsych.timelineVariable("right");
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
      jsPsych.data.get().addToLast(trialAttributes);
    },
    extensions: [{ type: jsPsychExtensionMouseTracking , params: {targets: ['#left-top','#left-mid','#left-bot','#right-top','#right-mid','#right-bot']}}],
  };

  const choiceTrial = {
    type: canvasButtonResponse,
    canvas_size: [800, 860],
    stimulus: (canvas) => {
    const parent = canvas.parentNode;

    // Create our new container div
    const container = document.createElement('div');
    container.id = 'stimulus-container';
    container.style.position = 'relative';

    // Insert the container into the DOM right before the canvas
    parent.insertBefore(container, canvas);

    // Move the canvas inside our new container
    container.appendChild(canvas);

    // Now that the canvas is wrapped, call your drawing function
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

      // Add trial type metadata for analysis
      trialAttributes.trial_type = jsPsych.timelineVariable("trialType");
      trialAttributes.categorical_winner = jsPsych.timelineVariable("categoricalWinner");
      trialAttributes.relational_winner = jsPsych.timelineVariable("relationalWinner");
      trialAttributes.critical_dim = jsPsych.timelineVariable("criticalDim");
      trialAttributes.left_sum = jsPsych.timelineVariable("leftSum");
      trialAttributes.right_sum = jsPsych.timelineVariable("rightSum");
      trialAttributes.left_wins = jsPsych.timelineVariable("leftWins");
      trialAttributes.right_wins = jsPsych.timelineVariable("rightWins");

      // Add discriminability data
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
    extensions: [{ type: jsPsychExtensionMouseTracking , params: {targets: ['#left-top','#left-mid','#left-bot','#right-top','#right-mid','#right-bot']}}],
  };

  const practiceChoiceTrials = {
    timeline: [choiceTrial, choice_fixation],
    timeline_variables: trainingChoiceArr,
    randomize_order: true
  };

  // Split choice trials into two halves for mid-experiment break
  const halfwayPoint = Math.floor(choiceArr.length / 2);
  const choiceArrFirstHalf = choiceArr.slice(0, halfwayPoint);
  const choiceArrSecondHalf = choiceArr.slice(halfwayPoint);

  const choiceTrialsFirstHalf = {
    timeline: [PreExpChoiceTrial, choiceTrial, choice_fixation],
    timeline_variables: choiceArrFirstHalf,
  };

  const choiceTrialsSecondHalf = {
    timeline: [PreExpChoiceTrial, choiceTrial, choice_fixation],
    timeline_variables: choiceArrSecondHalf,
  };

  // Break trial in the middle of choice trials
  const midExperimentBreak = {
    type: htmlButtonResponse,
    stimulus: `
      <div style="font-size: 20px; max-width: 600px; margin: auto;">
        <p><b>Halfway there!</b></p>
        <p>You've completed half of the choice trials. Feel free to take a short break (up to 1 minute).</p>
        <p>When you're ready to continue, click the button below.</p>
      </div>
    `,
    choices: ['Continue'],
    trial_duration: 60000, // Max 1 minute break
    data: {
      task: 'break',
      phase: 'mid_experiment_break'
    }
  };


  const survey_qs = {
    type: surveyText,
    questions: [
    {prompt: 'What input device did you use on the choice trials (trackpad, mouse, something else)?', required: true},
    {prompt: 'Did you have more difficulty with any of the three dimensions (shape, color, line angle)?',required: true},
    {prompt: 'What strategy did you use to select the options in the choice trials where you had to use the mouse to look around?',required: true,rows:5},
    {prompt: 'Do you have any other comments or concerns that you would like us to know?',required: true, rows: 5},
  ]
  }

  const endQuestionnaire = {
      type: jsPsychSurvey,
      survey_json: {
        title: "",
        showQuestionNumbers: "on",
        pages: [{
          name: "page1",
          elements: [
            {
              type: "text",
              name: "age",
              title: "How old are you? (in years)",
              isRequired: true
            },
            {
              type: "radiogroup",
              name: "gender",
              title: "How do you identify your gender?",
              choices: ["Male", "Female", "Trans male", "Trans female", "Gender nonbinary", "Genderqueer", "Different identity"],
              isRequired: true
            },
            {
              type: "checkbox",
              name: "race",
              title: "What is your race? (Select all that apply)",
              choices: ["American Indian or Alaskan Native", "Asian", "Black or African American", "Native Hawaiian or Other Pacific Islander", "White", "Other"],
              isRequired: true
            },
            {
              type: "radiogroup",
              name: "ethnicity",
              title: "What is your ethnic background? (Select one)",
              choices: ["Hispanic, Latinx, or Spanish origin", "Not Hispanic, Latinx, or Spanish origin"],
              isRequired: true
            },
            {
              type: "radiogroup",
              name: "how_well",
              title: "How well do you think you did on the tasks, overall?",
              choices: ["Very well", "Fairly well", "Fairly poorly", "Poorly"],
              isRequired: true
            },
            {
              type: "radiogroup",
              name: "medical_conditions",
              title: "Do you have any medical or neurological conditions that might impact your cognitive performance? (e.g. stroke, neurodegenerative disease, traumatic brain injury, untreated psychiatric/psychological conditions, etc.)?",
              choices: ["Yes", "No"],
              isRequired: true
            },
            {
              type: "comment",
              name: "medical_conditions_explanation",
              title: "If yes, please specify:"
            }
          ]
        }]
      },
      data: {
        phase: "experiment_end",
        task: "experiment_complete_questionnaire"
      }
    };
  const fin = {
    type: htmlKeyboardResponse,
    stimulus: "Thank you for participating, you earned a $3 bonus.<br> You should receive an email in a few days from the experimenter to send you your bonus.<br><br> Feel free to close this screen after pressing 'j'",
    trial_duration: 10000,
    response_ends_trial: true,
    choices: ['j']
  }

  // Instructions -> dimBlock1 ... choice
  // Where dimBlock is: PreExposure->learningTrials->PreCategoryInstructions->Fixation->Categorization->Fixation->postCatInstructions

  const trainingTrials = {
    "shape":shapeTrials,
    "color":colorTrials,
    "line":lineTrials
  }
  const introBlock = {
    timeline: [
      introInstructions(dims[0]),
      fixation,
      createPreExposureTrial(dims[0]+'Trial'),
      fixation,
      trainingTrials[dims[0]],
      preCategoryInstructions,
      makeFreeSort(dims[0],'perceptual'),
      fixation,
      makeFreeSort(dims[0], 'value'),
      fixation,
      postCat1Instructions(dims[0],dims[1]),
      fixation
    ]
  }
  const block2 = {
    timeline: [
      createPreExposureTrial(dims[1]+'Trial'),
      fixation,
      trainingTrials[dims[1]],
      preCategoryInstructions,
      makeFreeSort(dims[1], 'perceptual'),
      fixation,
      makeFreeSort(dims[1], 'value'),
      fixation,
      postCat1Instructions(dims[1],dims[2]),
      fixation
    ] 
  }
  const block3 = {
    timeline: [
      createPreExposureTrial(dims[2]+'Trial'),
      fixation,
      trainingTrials[dims[2]],
      preCategoryInstructions,
      makeFreeSort(dims[2], 'perceptual'),
      fixation,
      makeFreeSort(dims[2], 'value'),
      fixation,
    ] 
  }

  const timeline = [
    enter_fullscreen,
    introBlock,
    block2,
    block3,
    preChoiceInstructions1,
    fixation,
    practiceChoiceTrials,
    fixation,
    preChoiceInstructions2,
    fixation,
    choiceTrialsFirstHalf,
    midExperimentBreak,
    choiceTrialsSecondHalf,
    survey_qs,
    endQuestionnaire,
    fin
  ];
  return timeline;
};
