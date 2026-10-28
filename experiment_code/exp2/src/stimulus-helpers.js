// Define the center and radius
import { lab, rgb } from "d3-color";

let shapeImages = [];

// This is the "setter" function we will export.
// It allows other modules to set the value of shapeImages.
const setShapeImages = (images) => {
  shapeImages = images;
};

const preloadImages = () => {
  const localShapeImages = new Array(360);
  for (let i = 1; i < 361; i++) {
    const image = new Image();
    image.src = makeImagePath(i);
    localShapeImages[i - 1] = image;
  }
  return localShapeImages;
};
const labToRgbAtAngle = (angleDegrees, L, aCenter, bCenter, radius) => {
  // Convert angle from degrees to radians
  const angleRadians = angleDegrees * (Math.PI / 180);

  // Calculate a* and b* based on the angle
  const a = aCenter + radius * Math.cos(angleRadians);
  const b = bCenter + radius * Math.sin(angleRadians);

  // Convert LAB to RGB using d3-color
  const labColor = lab(L, a, b);
  const rgbColor = rgb(labColor);

  // Clamp and round RGB values
  const rgb2 = {
    r: Math.max(0, Math.min(255, Math.round(rgbColor.r))),
    g: Math.max(0, Math.min(255, Math.round(rgbColor.g))),
    b: Math.max(0, Math.min(255, Math.round(rgbColor.b))),
  };

  return [rgb2.r, rgb2.g, rgb2.b];
};

const drawCircle = (ctx, centerX, centerY, angle, radius, feedback) => {
  // Use a constant L value and an ab-plane with a radius of 60
  const L = 50;
  const aCenter = 20;
  const bCenter = 20;
  const abRadius = 60; // Represents the chroma (C*) or saturation

  // Get the RGB color from the LAB space using the color angle
  let r, g, b;
  if (Number.isInteger(angle)) {
    [r, g, b] = labToRgbAtAngle(angle, L, aCenter, bCenter, abRadius);
    // Set the fill style to the calculated RGB color
  } else {
    r = 255;
    g = 255;
    b = 255;
    // ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
  }
  ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
  // Draw the circle
  ctx.beginPath();
  ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);

  // Fill the circle with the calculated color
  ctx.fill();

  // Draw the circle's outline
  ctx.stroke();
  if (Number.isInteger(feedback.val)) {
    drawFeedback(ctx, feedback, centerX, centerY, radius);
  }

};

const drawLine = (ctx, centerX, centerY, angle, radius, feedback) => {
  let r = 255;
  let g = 255;
  let b = 255;
  ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
  // Draw the circle
  ctx.beginPath();
  ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);

  // Fill the circle with the calculated color
  ctx.fill();

  // Draw the circle's outline
  ctx.stroke();
  const angle_in_radians = ((angle + 155) * Math.PI) / 180;
  const line_length = radius * 0.8;
  const lineEndX = centerX + line_length * Math.cos(angle_in_radians);
  const lineEndY = centerY + line_length * Math.sin(angle_in_radians);

  ctx.beginPath();
  ctx.moveTo(centerX, centerY);
  ctx.lineTo(lineEndX, lineEndY);
  ctx.stroke();
  if (Number.isInteger(feedback.val)) {
    drawFeedback(ctx, feedback, centerX, centerY, radius);
  }
};

const drawShape = (ctx, centerX, centerY, angle, radius, feedback) => {
  let img = shapeImages[angle];
  ctx.drawImage(
    img,
    centerX - radius * 1.5,
    centerY - radius * 1.5,
    radius * 3,
    radius * 3,
  );
  if (Number.isInteger(feedback.val)) {
    drawFeedback(ctx, feedback, centerX, centerY, radius);
  }
};

const drawFeedback = (ctx, feedback, centerX, centerY, radius) => {
  ctx.save();
  // Set the line width to make the box stand out
  ctx.lineWidth = 4;

  // Calculate the bounding box dimensions.
  // This example uses the circle's bounding box.
  const rectX = centerX - radius / 2 + 24;
  const rectY = centerY - radius - 20;
  // const rectWidth = radius * 3;
  // const rectHeight = radius * 3;

  // Draw the green rectangle
  ctx.font = "48px serif";
  if (feedback.chosen && feedback.correct){
    ctx.fillStyle = `rgba(28, 132, 23, 1)`;
  } else if (feedback.chosen && !feedback.correct) {
    ctx.fillStyle = `rgba(168, 18, 18, 1)`;
  } else {
    ctx.fillStyle = `rgba(0, 0, 0, 1)`;
  }
  ctx.fillText(feedback.val, rectX, rectY);
  ctx.restore();
};

const drawTrialStim = (dim, ctx, centerX, centerY, radius, angle, feedback=null) => {
  if (feedback === null) {
    feedback = {
      val: null,
      left_val: null,
      right_val: null,
    };
  }
  if (dim == "color") {
    return drawCircle(ctx, centerX, centerY, angle, radius, feedback);
  } else if (dim == "shape") {
    return drawShape(ctx, centerX, centerY, angle, radius, feedback);
  } else if (dim == "line") {
    return drawLine(ctx, centerX, centerY, angle, radius, feedback);
  }
};

const drawPreExpStim = (dim, ctx, c, frameIndex) => {
  const centerX = c.width / 2;
  const centerY = c.height / 2;
  const radius = 100;
  ctx.clearRect(0, 0, c.width, c.height);
  drawTrialStim(dim, ctx, centerX, centerY, radius, frameIndex);
};

const drawTrainingTrial = (canvas, dim, leftValue, rightValue, feedback) => {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // --- Define the layout for the stimuli ---
  const centerY = canvas.height / 2;
  const radius = 100; // The effective "size" of the stimulus area
  const left_centerX = canvas.width / 4;
  const right_centerX = (canvas.width / 4) * 3;

  if (feedback.timeout) {
    const faster = "You must respond more quickly!";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.font = "24px serif";
    feedback.left_val = null;
    feedback.right_val = null;
    const centerX = canvas.width / 2;
    ctx.fillText(faster, centerX, 0);

  }
  // --- Draw the stimuli using the generic helper ---
  let left_feedback = {...feedback}
  left_feedback.correct_choice = left_feedback.left_correct
  left_feedback.val = left_feedback.left_val
  left_feedback.chosen = left_feedback.choice === 'f'
  let right_feedback = {...feedback}
  right_feedback.correct_choice = right_feedback.right_correct
  right_feedback.val = right_feedback.right_val
  right_feedback.chosen = right_feedback.choice === 'j'
  drawTrialStim(
    dim,
    ctx,
    left_centerX,
    centerY,
    radius,
    leftValue,
    left_feedback,
  );
  drawTrialStim(
    dim,
    ctx,
    right_centerX,
    centerY,
    radius,
    rightValue,
    right_feedback,
  );
};
const drawCategorizationTrial = (
  canvas,
  dim,
  cueAngle,
  leftAngle,
  rightAngle,
) => {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Define positions and sizes relative to canvas dimensions
  const radius = canvas.width / 8; // Stimulus size
  const cueX = canvas.width / 2;
  const cueY = canvas.height * 0.25;
  const choiceY = canvas.height * 0.7;
  const choiceXOffset = canvas.width / 4;

  // Draw the top (cue) stimulus
  drawTrialStim(dim, ctx, cueX, cueY, radius, cueAngle);

  // Draw the bottom-left choice
  drawTrialStim(dim, ctx, cueX - choiceXOffset, choiceY, radius, leftAngle);

  // Draw the bottom-right choice
  drawTrialStim(dim, ctx, cueX + choiceXOffset, choiceY, radius, rightAngle);
};


const createTargetDiv = (container, id, centerX, centerY, size) => {
  size = 2*size
  const div = document.createElement('div');
  div.id = id;
  div.className = 'mousetracking-target';
  div.style.position = 'absolute';
  div.style.width = `${size}px`;
  div.style.height = `${size}px`;
  div.style.left = `${centerX - size / 2}px`;
  div.style.top = `${centerY - size / 2}px`;
  div.style.borderRadius = '50%';
  div.style.backgroundColor = 'transparent';
  container.appendChild(div);
};
const drawChoiceCol = (container, canvas, ctx, startX, side, sideName) => {
  const itemSize = canvas.height * 0.12;
  const margin = canvas.height * 0.2;
  const startY = canvas.height * 0.15;
  let { top, mid, bot } = side;
  
  const topY = startY;
  drawTrialStim(top.dim, ctx, startX, topY, itemSize, top.angle);
  createTargetDiv(container, `${sideName}-top`, startX, topY, itemSize);

  const midY = startY + itemSize + margin;
  drawTrialStim(mid.dim, ctx, startX, midY, itemSize, mid.angle);
  createTargetDiv(container, `${sideName}-mid`, startX, midY, itemSize);

  const botY = startY + (itemSize + margin) * 2;
  drawTrialStim(bot.dim, ctx, startX, botY, itemSize, bot.angle);
  createTargetDiv(container, `${sideName}-bot`, startX, botY, itemSize);
};

const drawChoiceTrial = (canvas) => {
  // Find the container, which is now the canvas's direct parent
  const container = canvas.parentNode; 
  
  // IMPORTANT: Clear old divs from the previous trial
  container.querySelectorAll('.mousetracking-target').forEach(el => el.remove());

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const columnMargin = canvas.width * 0.25;
  const leftColumnX = columnMargin;
  const rightColumnX = canvas.width - columnMargin;
  let left = jsPsych.timelineVariable("left");
  let right = jsPsych.timelineVariable("right");

  // Pass the found container and the side name ('left') to the drawing function
  drawChoiceCol(container, canvas, ctx, leftColumnX, left, 'left');

  // Pass the found container and the side name ('right')
  drawChoiceCol(container, canvas, ctx, rightColumnX, right, 'right');
};
function preloadCanvases() {
  const preloadedCanvases = [];
  for (let i = 0; i < fullTargList.length; i++) {
    let currCanvases = [];
    for (let j = 0; j < fullTargList[i].length; j++) {
      let c = document.createElement("canvas");
      c.width = 600;
      c.height = 600;
      createStimulusCanvas(
        c,
        fullTargList[i][j].color,
        fullTargList[i][j].shape,
      );
      currCanvases.push(c);
    }
    preloadedCanvases.push(currCanvases);
  }
  return preloadedCanvases;
}

// General helper functions
const uniformRandom = (low, high) => {
  return Math.floor(Math.random() * (high - low) + low);
};

const arrayRange = (start, stop, step) =>
  Array.from(
    { length: (stop - start) / step + 1 },
    (value, index) => start + index * step,
  );

// name related
function makeImagePath(shapeNum) {
  // leftpad shapeNum to 3 digits
  shapeNum = shapeNum.toString().padStart(3, "0");
  let imagePath = `/assets/transparent_VCSshapes/VCS_${shapeNum}_transparent_cv2.png`;
  return imagePath;
}
const fullImgPaths = Array.from({ length: 360 }, (_, i) =>
  makeImagePath(i + 1),
);
fullImgPaths.push("/assets/example_choice_trial.png");
const categoryToNumString = (categories, cat_num) => {
  let { min_val, max_val } = categories[cat_num];
  let num = uniformRandom(min_val, max_val);
  return num.toString().padStart(3, "0");
};

export {
  setShapeImages,
  makeImagePath,
  categoryToNumString,
  arrayRange,
  drawChoiceTrial,
  drawTrainingTrial,
  drawCategorizationTrial,
  drawPreExpStim,
  preloadImages,
  fullImgPaths,
};
