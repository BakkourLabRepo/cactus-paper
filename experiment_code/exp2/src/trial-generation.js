// =======================================================================
// DIVERGENT/NON-DIVERGENT TRIAL GENERATION FOR CACTUS
// =======================================================================
// 
// This module generates choice trials with controlled proportions of
// divergent trials (where categorical and relational policies disagree)
// and non-divergent trials (where they agree).
//
// Definitions:
// - Categorical winner: option with higher summed expected value
// - Relational winner: option that "wins" on more dimensions (2+ of 3)
// - Divergent trial: categorical winner ≠ relational winner
// =======================================================================

/**
 * Compute the summed value for an option (array of 3 category indices)
 * @param {number[]} categories - Array of 3 category indices (0-4)
 * @param {number[]} valueMap - Array mapping category index to point value
 * @returns {number} - Total value
 */
const computeSummedValue = (categories, valueMap) => {
  return categories.reduce((sum, cat) => sum + valueMap[cat], 0);
};

/**
 * Count how many dimensions an option "wins" on
 * @param {number[]} leftCats - Left option's category indices [dim0, dim1, dim2]
 * @param {number[]} rightCats - Right option's category indices [dim0, dim1, dim2]
 * @param {Object} valueMaps - Object with value arrays for each dimension
 * @param {string[]} dims - Dimension names in order ['shape', 'color', 'line']
 * @returns {Object} - {leftWins: number, rightWins: number, ties: number}
 */
const countDimensionWins = (leftCats, rightCats, valueMaps, dims) => {
  let leftWins = 0;
  let rightWins = 0;
  let ties = 0;
  
  for (let i = 0; i < 3; i++) {
    const leftVal = valueMaps[dims[i]][leftCats[i]];
    const rightVal = valueMaps[dims[i]][rightCats[i]];
    
    if (leftVal > rightVal) leftWins++;
    else if (rightVal > leftVal) rightWins++;
    else ties++;
  }
  
  return { leftWins, rightWins, ties };
};

/**
 * Classify a trial as divergent, non-divergent, or tie
 * @param {number[]} leftCats - Left option's category indices
 * @param {number[]} rightCats - Right option's category indices
 * @param {Object} valueMaps - Value mappings per dimension
 * @param {string[]} dims - Dimension names
 * @returns {Object} - Trial classification with metadata
 */
const classifyTrial = (leftCats, rightCats, valueMaps, dims) => {
  // Compute summed values (categorical policy)
  const leftSum = leftCats.reduce((sum, cat, i) => sum + valueMaps[dims[i]][cat], 0);
  const rightSum = rightCats.reduce((sum, cat, i) => sum + valueMaps[dims[i]][cat], 0);
  
  // Determine categorical winner
  let categoricalWinner;
  if (leftSum > rightSum) categoricalWinner = 'left';
  else if (rightSum > leftSum) categoricalWinner = 'right';
  else categoricalWinner = 'tie';
  
  // Count dimension wins (relational policy)
  const { leftWins, rightWins, ties } = countDimensionWins(leftCats, rightCats, valueMaps, dims);
  
  // Determine relational winner
  let relationalWinner;
  if (leftWins > rightWins) relationalWinner = 'left';
  else if (rightWins > leftWins) relationalWinner = 'right';
  else relationalWinner = 'tie';
  
  // Classify trial type
  let trialType;
  if (categoricalWinner === 'tie' || relationalWinner === 'tie') {
    trialType = 'tie';
  } else if (categoricalWinner === relationalWinner) {
    trialType = 'non-divergent';
  } else {
    trialType = 'divergent';
  }
  
  // Find critical dimension (largest absolute value difference)
  let maxDiff = 0;
  let criticalDim = 0;
  for (let i = 0; i < 3; i++) {
    const diff = Math.abs(valueMaps[dims[i]][leftCats[i]] - valueMaps[dims[i]][rightCats[i]]);
    if (diff > maxDiff) {
      maxDiff = diff;
      criticalDim = i;
    }
  }
  
  return {
    trialType,
    categoricalWinner,
    relationalWinner,
    leftSum,
    rightSum,
    leftWins,
    rightWins,
    criticalDim,
    criticalDimName: dims[criticalDim],
    valueDifference: Math.abs(leftSum - rightSum)
  };
};

/**
 * Generate a divergent trial configuration
 *
 * Strategy: Randomly generate trials and keep ones that are divergent
 * This is simpler and more robust than trying to construct divergent trials
 *
 * @param {Object} valueMaps - Value mappings per dimension
 * @param {string[]} dims - Dimension names
 * @param {Function} randomSample - Function to sample random elements
 * @param {number} maxAttempts - Maximum retry attempts (default 1000)
 * @returns {Object} - {leftCats, rightCats, metadata}
 */
const generateDivergentTrial = (valueMaps, dims, randomSample, maxAttempts = 1000) => {
  let attempts = 0;

  while (attempts < maxAttempts) {
    attempts++;

    // Generate random categories for left and right options
    let leftCats = [0, 0, 0];
    let rightCats = [0, 0, 0];

    for (let i = 0; i < 3; i++) {
      leftCats[i] = randomSample([0, 1, 2, 3, 4], 1)[0];
      rightCats[i] = randomSample([0, 1, 2, 3, 4], 1)[0];
    }

    // Classify the trial
    const classification = classifyTrial(leftCats, rightCats, valueMaps, dims);

    // Keep it if it's divergent
    if (classification.trialType === 'divergent') {
      return {
        leftCats,
        rightCats,
        ...classification
      };
    }
  }

  // If we've exhausted attempts, throw an error
  throw new Error(`Failed to generate divergent trial after ${maxAttempts} attempts. Check value configuration.`);
};

/**
 * Generate a non-divergent trial configuration
 *
 * Strategy: Randomly generate trials and keep ones that are non-divergent
 * This is simpler and more robust than trying to construct non-divergent trials
 *
 * @param {Object} valueMaps - Value mappings per dimension
 * @param {string[]} dims - Dimension names
 * @param {Function} randomSample - Function to sample random elements
 * @param {number} maxAttempts - Maximum retry attempts (default 1000)
 * @returns {Object} - {leftCats, rightCats, metadata}
 */
const generateNonDivergentTrial = (valueMaps, dims, randomSample, maxAttempts = 1000) => {
  let attempts = 0;

  while (attempts < maxAttempts) {
    attempts++;

    // Generate random categories for left and right options
    let leftCats = [0, 0, 0];
    let rightCats = [0, 0, 0];

    for (let i = 0; i < 3; i++) {
      leftCats[i] = randomSample([0, 1, 2, 3, 4], 1)[0];
      rightCats[i] = randomSample([0, 1, 2, 3, 4], 1)[0];
    }

    // Classify the trial
    const classification = classifyTrial(leftCats, rightCats, valueMaps, dims);

    // Keep it if it's non-divergent
    if (classification.trialType === 'non-divergent') {
      return {
        leftCats,
        rightCats,
        ...classification
      };
    }
  }

  // If we've exhausted attempts, throw an error
  throw new Error(`Failed to generate non-divergent trial after ${maxAttempts} attempts. Check value configuration.`);
};

/**
 * Generate the full set of choice trials with controlled divergent proportion
 * 
 * @param {number} nTrials - Total number of trials
 * @param {number} targetDivergentProp - Target proportion of divergent trials (e.g., 0.35)
 * @param {Object} valueMaps - Value mappings per dimension {shape: [...], color: [...], line: [...]}
 * @param {string[]} dims - Dimension names in order for this participant
 * @param {Object} categories - Category boundaries from config
 * @param {Function} randomSample - jsPsych.randomization.sampleWithoutReplacement
 * @param {Function} shuffle - jsPsych.randomization.shuffle
 * @returns {Object[]} - Array of trial objects ready for timeline_variables
 */
export const generateChoiceTrials = (
  nTrials,
  targetDivergentProp,
  valueMaps,
  dims,
  categories,
  randomSample,
  shuffle
) => {
  console.log('=== Generating Choice Trials ===');
  console.log('Value maps:', valueMaps);
  console.log('Dimensions order:', dims);
  console.log('Categories:', categories);

  const nDivergent = Math.round(nTrials * targetDivergentProp);
  const nNonDivergent = nTrials - nDivergent;

  console.log(`Generating ${nDivergent} divergent and ${nNonDivergent} non-divergent trials...`);

  const trials = [];

  // Generate divergent trials
  for (let i = 0; i < nDivergent; i++) {
    try {
      const config = generateDivergentTrial(valueMaps, dims, randomSample);
      trials.push({
        ...config,
        trialIndex: i
      });
    } catch (error) {
      console.error(`Failed to generate divergent trial ${i}:`, error);
      throw error;
    }
  }

  // Generate non-divergent trials
  for (let i = 0; i < nNonDivergent; i++) {
    try {
      const config = generateNonDivergentTrial(valueMaps, dims, randomSample);
      trials.push({
        ...config,
        trialIndex: nDivergent + i
      });
    } catch (error) {
      console.error(`Failed to generate non-divergent trial ${i}:`, error);
      throw error;
    }
  }
  
  // Shuffle all trials
  const shuffledTrials = shuffle(trials);

  // Convert category indices to angles and build full trial objects
  return shuffledTrials.map((trial, idx) => {
    // Shuffle dimension order for THIS trial (like exp1)
    // Create pairs of [dim, catIndex] to shuffle together
    const dimCatPairs = dims.map((dim, i) => ({
      dim: dim,
      leftCat: trial.leftCats[i],
      rightCat: trial.rightCats[i]
    }));
    const shuffledPairs = shuffle(dimCatPairs);

    // Helper to convert category index to angle string
    const catToAngle = (catIdx) => {
      const cat = categories[catIdx];
      // Sample uniformly within category range
      const angle = cat.min_val + Math.floor(Math.random() * (cat.max_val - cat.min_val + 1));
      return {
        angleStr: String(angle).padStart(3, '0'),
        angle: angle
      };
    };

    // Generate angles for each position using the shuffled pairs
    const positions = ['top', 'mid', 'bot'];
    const leftObj = {};
    const rightObj = {};

    positions.forEach((pos, i) => {
      const pair = shuffledPairs[i];
      const leftAngle = catToAngle(pair.leftCat);
      const rightAngle = catToAngle(pair.rightCat);

      leftObj[pos] = {
        dim: pair.dim,
        angle: leftAngle.angle,
        value: valueMaps[pair.dim][pair.leftCat]
      };

      rightObj[pos] = {
        dim: pair.dim,
        angle: rightAngle.angle,
        value: valueMaps[pair.dim][pair.rightCat]
      };
    });

    // Build the trial object in your existing format
    return {
      left: leftObj,
      right: rightObj,
      prompt: "<p>Which is higher value?</p>",
      // Metadata for analysis
      trialType: trial.trialType,
      categoricalWinner: trial.categoricalWinner,
      relationalWinner: trial.relationalWinner,
      criticalDim: trial.criticalDimName,
      leftSum: trial.leftSum,
      rightSum: trial.rightSum,
      leftWins: trial.leftWins,
      rightWins: trial.rightWins
    };
  });
};

/**
 * Add discriminability metadata to trials
 * Call this after generating trials to compute per-dimension discriminability
 * 
 * @param {Object[]} trials - Array of trial objects
 * @param {string[]} dims - Dimension names
 * @returns {Object[]} - Trials with discriminability added
 */
export const addDiscriminabilityMetadata = (trials, dims) => {
  return trials.map(trial => {
    const discriminability = {};
    
    // Compute discriminability for each dimension (absolute angle difference)
    const positions = ['top', 'mid', 'bot'];
    positions.forEach((pos, idx) => {
      const dimName = trial.left[pos].dim;
      const leftAngle = trial.left[pos].angle;
      const rightAngle = trial.right[pos].angle;
      discriminability[dimName] = Math.abs(leftAngle - rightAngle);
    });
    
    // Also note discriminability on critical dimension (for divergent trials)
    const criticalDiscrim = trial.criticalDim ? discriminability[trial.criticalDim] : null;
    
    return {
      ...trial,
      discriminability,
      criticalDimDiscriminability: criticalDiscrim
    };
  });
};
