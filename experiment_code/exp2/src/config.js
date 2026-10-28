// =======================================================================
// EXPERIMENT CONFIGURATION
// =======================================================================

export const EXPERIMENT_PARAMS = {
  nLearningTrials: 40,
  nPerPrompt: 10,
  nPairingsForLearning: 4,
  nChoiceTrials: 120,
  nDivergentTrials: 60,
  fixationDuration: 1000,
  feedbackDuration: 1000,
  trialDuration: 2500,
  stimulusRadius: 100,
  choiceMaxDuration: 120000,
  covertAttentionDuration: 250,
  freeSortMaxDuration: 60000,
  // Define categories and values here
  categories: [
    { min_val: 1, max_val: 31 },
    { min_val: 51, max_val: 81 },
    { min_val: 101, max_val: 131 },
    { min_val: 151, max_val: 181 },
    { min_val: 201, max_val: 231 },
  ],
  // The point values assigned to each category for each dimension
  values: {
    shape: Math.random() < 0.5 ? [0, 25, 50, 75, 100] : [100, 75, 50, 25, 0],
    color: Math.random() < 0.5 ? [0, 25, 50, 75, 100] : [100, 75, 50, 25, 0],
    line: Math.random() < 0.5 ? [0, 25, 50, 75, 100] : [100, 75, 50, 25, 0],
  },
};

export var ParameterType;
(function (ParameterType) {
  ParameterType[(ParameterType["BOOL"] = 0)] = "BOOL";
  ParameterType[(ParameterType["STRING"] = 1)] = "STRING";
  ParameterType[(ParameterType["INT"] = 2)] = "INT";
  ParameterType[(ParameterType["FLOAT"] = 3)] = "FLOAT";
  ParameterType[(ParameterType["FUNCTION"] = 4)] = "FUNCTION";
  ParameterType[(ParameterType["KEY"] = 5)] = "KEY";
  ParameterType[(ParameterType["KEYS"] = 6)] = "KEYS";
  ParameterType[(ParameterType["SELECT"] = 7)] = "SELECT";
  ParameterType[(ParameterType["HTML_STRING"] = 8)] = "HTML_STRING";
  ParameterType[(ParameterType["IMAGE"] = 9)] = "IMAGE";
  ParameterType[(ParameterType["AUDIO"] = 10)] = "AUDIO";
  ParameterType[(ParameterType["VIDEO"] = 11)] = "VIDEO";
  ParameterType[(ParameterType["OBJECT"] = 12)] = "OBJECT";
  ParameterType[(ParameterType["COMPLEX"] = 13)] = "COMPLEX";
  ParameterType[(ParameterType["TIMELINE"] = 14)] = "TIMELINE";
})(ParameterType || (ParameterType = {}));
