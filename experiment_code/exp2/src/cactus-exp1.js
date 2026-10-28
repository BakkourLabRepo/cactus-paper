const jsPsych = initJsPsych({
  on_start: () => {
    // let subid = jsPsych.data.getURLVariable("PROLIFIC_PID");
    let subid = jsPsych.data.getURLVariable("sona_id");
    jsPsych.data.addProperties({ subid: subid });
    preloadImages();
  },
  on_finish: () => {
    jsPsych.data.addProperties({ experiment: "CACTUS-exp1" });
    // window.location = "https://wupsych.sona-systems.com/webstudy_credit.aspx?experiment_id=643&credit_token=4df908af0ec14031a67903da04b48994&survey_code="  + sona_id
  },
});

// value setup ------------

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
}; // we check to see if the initial value is 0 and if so we go 1->231, if not we reverse it

const angleDimToVal = (angle, dim) => {
  for (let i = 0; i < EXPERIMENT_PARAMS.categories.length; i++) {
    if (angle > EXPERIMENT_PARAMS.categories[i].max_val) {
      continue;
    } else {
      return EXPERIMENT_PARAMS.values[dim][i];
    }
  }
};

// preloading shapes -------
const shapeImages = new Array(360);

const fullImgPaths = Array.from({ length: 360 }, (_, i) =>
  makeImagePath(i + 1),
);
const preloadImages = () => {
  for (let i = 1; i < 361; i++) {
    const image = new Image();

    image.src = makeImagePath(i);
    shapeImages[i - 1] = image;
  }
};
const preloadTrial = {
  type: preload,
  images: fullImgPaths,
  on_start: () => {
    preloadImages();
  },
};

// Block setup --------------

// learning trials
var initial_blocks = [];

for (let i = 0; i < EXPERIMENT_PARAMS.nLearningTrials; i++) {
  let [trial_left, trial_right] =
    jsPsych.randomization.sampleWithoutReplacement([0, 1, 2, 3], 2);
  let tempTrialObj = {
    left: categoryToNumString(EXPERIMENT_PARAMS.categories, trial_left),
    left_category: trial_left,
    right: categoryToNumString(EXPERIMENT_PARAMS.categories, trial_right),
    right_category: trial_right,
  };
  initial_blocks.push(tempTrialObj);
}
let numsForLearning = initial_blocks;

// choice trials
// currently the choice dims are always going to be in a set order from trial to trial, but randomized across ppts
const colorDimArr = Array(EXPERIMENT_PARAMS.nChoiceTrials).fill("color");
const shapeDimArr = Array(EXPERIMENT_PARAMS.nChoiceTrials).fill("shape");
const lineDimArr = Array(EXPERIMENT_PARAMS.nChoiceTrials).fill("line");

let choiceDims = [colorDimArr, shapeDimArr, lineDimArr];

choiceDims = jsPsych.randomization.shuffle(choiceDims);

// category trials
const assignCatFoils = (angle) => {
  if (angle < 20) {
    return [angle + 20, angle + 40];
  } else if (angle > 160) {
    return [angle - 20, angle - 40];
  } else {
    return [angle - 20, angle + 20];
  }
};

let catTrials = {
  shape: [],
  color: [],
  line: [],
};
for (const dim of Object.keys(catTrials)) {
  for (let j = 0; j < EXPERIMENT_PARAMS.nCatTrials; j++) {
    let currCat = jsPsych.randomization.sampleWithoutReplacement(
      [0, 1, 2, 3],
      1,
    );
    let currCatAng = parseInt(
      categoryToNumString(EXPERIMENT_PARAMS.categories, currCat),
    );
    let foils = assignCatFoils(currCatAng);
    foils = jsPsych.randomization.shuffle(foils);
    let catObj = {
      cue: currCatAng,
      cueCat: currCat,
      left: foils[0],
      right: foils[1],
    };
    catTrials[dim].push(catObj);
  }
}
catTrials.shape.forEach((trial) => (trial.dim = "shape"));
catTrials.color.forEach((trial) => (trial.dim = "color"));
catTrials.line.forEach((trial) => (trial.dim = "line"));

// fully random setup for the values for now but we can think more about pseudo-randomization later
// we need 6 values per choiceTrial

let choiceArr = [];
for (let i = 0; i < EXPERIMENT_PARAMS.nChoiceTrials; i++) {
  let angleArr = [];
  for (let j = 0; j < 6; j++) {
    let angleStr = categoryToNumString(
      EXPERIMENT_PARAMS.categories,
      jsPsych.randomization.sampleWithoutReplacement([0, 1, 2, 3], 1),
    );
    let angleNum = parseInt(angleStr);
    angleArr.push({
      angleStr: angleStr,
      angle: angleNum,
    });
  }
  let ChoiceTrialObj = {
    left: {
      top: {
        dim: choiceDims[0][i],
        angle: angleArr[0].angle,
        value: angleDimToVal(angleArr[0].angle, choiceDims[0][i]),
      },
      mid: {
        dim: choiceDims[1][i],
        angle: angleArr[1].angle,
        value: angleDimToVal(angleArr[1].angle, choiceDims[1][i]),
      },
      bot: {
        dim: choiceDims[2][i],
        angle: angleArr[2].angle,
        value: angleDimToVal(angleArr[2].angle, choiceDims[2][i]),
      },
    },
    right: {
      top: {
        dim: choiceDims[0][i],
        angle: angleArr[3].angle,
        value: angleDimToVal(angleArr[3].angle, choiceDims[0][i]),
      },
      mid: {
        dim: choiceDims[1][i],
        angle: angleArr[4].angle,
        value: angleDimToVal(angleArr[4].angle, choiceDims[1][i]),
      },
      bot: {
        dim: choiceDims[2][i],
        angle: angleArr[5].angle,
        value: angleDimToVal(angleArr[5].angle, choiceDims[2][i]),
      },
    },
  };
  choiceArr.push(ChoiceTrialObj);
}

// --------------------
const assignFeedback = (last_trial_data) => {
  let response = last_trial_data.response;
  let { left_category, right_category, dim } = last_trial_data;
  let trial_values = EXPERIMENT_PARAMS.values[dim];

  if (response === "f") {
    console.log([trial_values[left_category], null]);
    return [trial_values[left_category], null];
  } else if (response === "j") {
    return [null, trial_values[right_category]];
  } else {
    return [null, null];
  }
};

const fixation = {
  type: htmlKeyboardResponse,
  stimulus: '<div style="font-size: 100px">+</div>',
  response_ends_trial: false,
  choices: "NO_KEYS",
  trial_duration: EXPERIMENT_PARAMS.fixationDuration,
};

let frameIndex = 0;

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
        if (frameIndex >= 180) {
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
        console.log(last_trial_data)
        let feedback = assignFeedback(last_trial_data);

        drawingFunction(canvas, dim, left_val, right_val, feedback);
      } else {
        let feedback = [null, null];
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
    prompt: "<p>Which is higher value?</p>",
    data: () => {
      let left_rew = EXPERIMENT_PARAMS.values[dim][jsPsych.timelineVariable('left_category')]
      let right_rew = EXPERIMENT_PARAMS.values[dim][jsPsych.timelineVariable('right_category')]
      return {
      task: "learning",
      dim: dim,
      left_choice: jsPsych.timelineVariable("left"),
      left_category: jsPsych.timelineVariable("left_category"),
      left_rew: left_rew,
      right_choice: jsPsych.timelineVariable("right"),
      right_category: jsPsych.timelineVariable("right_category"),
      right_rew: right_rew,
    }
  }
}
};

const shapeTrial = createTrainingTrial(drawTrainingTrial, false, "shapeTrial");
const shapeTrialFeedback = createTrainingTrial(
  drawTrainingTrial,
  true,
  "shapeTrial",
);

const colorTrial = createTrainingTrial(drawTrainingTrial, false, "colorTrial");
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
  timeline_variables: numsForLearning
};

const colorTrials = {
  timeline: [colorTrial, colorTrialFeedback, fixation],
  timeline_variables: numsForLearning
};

const lineTrials = {
  timeline: [lineTrial, lineTrialFeedback, fixation],
  timeline_variables: numsForLearning
};

// Categorization trials


const categorizationTrial = {
  type: canvasKeyboardResponse,
  canvas_size: [600, 600], // A square canvas might work well here
  stimulus: (canvas) => {
    // Get the values for the current trial from timeline_variables
    const dim = jsPsych.timelineVariable("dim");
    const cue = jsPsych.timelineVariable("cue");
    const left = jsPsych.timelineVariable("left");
    const right = jsPsych.timelineVariable("right");

    drawCategorizationTrial(canvas, dim, cue, left, right);
  },
  choices: ["f", "j"],
  prompt: "<p>Which is closer to the top item?</p>",
  trialDuration: EXPERIMENT_PARAMS.trialDuration,
  data: {
    task: "categorization",
    dim: jsPsych.timelineVariable("dim"),
    cue: jsPsych.timelineVariable("cue"),
    cue_category: jsPsych.timelineVariable("cueCat"),
    left_choice: jsPsych.timelineVariable("left"),
    right_choice: jsPsych.timelineVariable("right"),
  },
};

// Generate the timeline nodes for each categorization block.
// We add the 'dim' property here so the trial knows what to draw.
catTrials.shape.forEach((trial) => (trial.dim = "shape"));
catTrials.color.forEach((trial) => (trial.dim = "color"));
catTrials.line.forEach((trial) => (trial.dim = "line"));

// Create a timeline for each dimension using the generated trials
const shapeCategorizationBlock = {
  timeline: [categorizationTrial, fixation],
  timeline_variables: catTrials.shape,
  randomize_order: true
};

const colorCategorizationBlock = {
  timeline: [categorizationTrial, fixation],
  timeline_variables: catTrials.color,
  randomize_order: true
};

const lineCategorizationBlock = {
  timeline: [categorizationTrial, fixation],
  timeline_variables: catTrials.line,
  randomize_order: true
};

const choiceTrial = {
  type: canvasKeyboardResponse,
  canvas_size: [500, 860],
  stimulus: (canvas) => {
    drawChoiceTrial(canvas);
  },
  choices: ["f", "j"],
  prompt: "<p>Which is higher value?</p>",
  on_finish: () => {
    let trialAttributes = {};
    let trialObject = {}
    trialObject.left = jsPsych.timelineVariable('left');
    trialObject.right = jsPsych.timelineVariable('right')
    for (let side in trialObject) {
      let total_rew = 0
      for (let position in trialObject[side]) {
        for (let attribute in trialObject[side][position]) {
          let key = `choice_trial_${side}_${position}_${attribute}`;
          trialAttributes[key] = trialObject[side][position][attribute];
          if (attribute == 'value') {
            total_rew = total_rew + trialObject[side][position][attribute]
          }
          
        }
      }
      let key = `choice_trial_${side}_total_rew`
      trialAttributes[key] = total_rew
    }
    jsPsych.data.get().addToLast(trialAttributes);
  },
};

const choiceTrials = {
  timeline: [choiceTrial, fixation],
  timeline_variables: choiceArr
};

const timeline = [
  preloadTrial,
  introInstructions,
  fixation,
  shapePreExposureTrial,
  fixation,
  shapeTrials,
  preCategoryInstructions,
  fixation,
  shapeCategorizationBlock,
  postCat1Instructions,
  fixation,
  colorPreExposureTrial,
  fixation,
  colorTrials,
  postLearn2Instructions,
  fixation,
  colorCategorizationBlock,
  postCat2Instructions,
  fixation,
  linePreExposureTrial,
  fixation,
  lineTrials,
  postLearn2Instructions,
  fixation,
  lineCategorizationBlock,
  preChoiceInstructions,
  fixation,
  choiceTrials,
]

jsPsych.run(timeline)