import { createCanvas } from 'canvas';
import { lab, rgb } from 'd3-color';
import fs from 'fs';
import path from 'path';

// Replicate the LAB to RGB conversion from stimulus-helpers.js
const labToRgbAtAngle = (angleDegrees, L, aCenter, bCenter, radius) => {
  const angleRadians = angleDegrees * (Math.PI / 180);
  const a = aCenter + radius * Math.cos(angleRadians);
  const b = bCenter + radius * Math.sin(angleRadians);

  const labColor = lab(L, a, b);
  const rgbColor = rgb(labColor);

  return {
    r: Math.max(0, Math.min(255, Math.round(rgbColor.r))),
    g: Math.max(0, Math.min(255, Math.round(rgbColor.g))),
    b: Math.max(0, Math.min(255, Math.round(rgbColor.b))),
  };
};

// Draw a color circle at a specific angle
const drawColorStimulus = (ctx, angle, centerX, centerY, radius) => {
  const L = 50;
  const aCenter = 20;
  const bCenter = 20;
  const abRadius = 60;

  const { r, g, b } = labToRgbAtAngle(angle, L, aCenter, bCenter, abRadius);

  ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
  ctx.strokeStyle = 'black';
  ctx.lineWidth = 2;

  ctx.beginPath();
  ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
  ctx.fill();
  ctx.stroke();
};

// Draw a line stimulus at a specific angle
const drawLineStimulus = (ctx, angle, centerX, centerY, radius) => {
  // White circle background
  ctx.fillStyle = 'rgb(255, 255, 255)';
  ctx.strokeStyle = 'black';
  ctx.lineWidth = 2;

  ctx.beginPath();
  ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
  ctx.fill();
  ctx.stroke();

  // Draw the line (adjust angle by +65 to match stimulus-helpers.js line 157)
  const adjustedAngle = angle + 65;
  const angleInRadians = ((adjustedAngle + 90) * Math.PI) / 180;
  const lineLength = radius * 0.8;
  const lineEndX = centerX + lineLength * Math.cos(angleInRadians);
  const lineEndY = centerY + lineLength * Math.sin(angleInRadians);

  ctx.strokeStyle = 'black';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(centerX, centerY);
  ctx.lineTo(lineEndX, lineEndY);
  ctx.stroke();
};

// Generate stimuli for the specified angles
const generateSortingStimuli = (angles, outputDir) => {
  const canvasSize = 400;
  const centerX = canvasSize / 2;
  const centerY = canvasSize / 2;
  const radius = 100;

  // Ensure output directory exists
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  angles.forEach((angle, index) => {
    // Generate color stimulus
    const colorCanvas = createCanvas(canvasSize, canvasSize);
    const colorCtx = colorCanvas.getContext('2d');

    // Transparent background (no fill)

    drawColorStimulus(colorCtx, angle, centerX, centerY, radius);

    const colorPath = path.join(outputDir, `frame_color_${index}.png`);
    const colorBuffer = colorCanvas.toBuffer('image/png');
    fs.writeFileSync(colorPath, colorBuffer);
    console.log(`Generated color frame ${index} at angle ${angle}`);

    // Generate line stimulus
    const lineCanvas = createCanvas(canvasSize, canvasSize);
    const lineCtx = lineCanvas.getContext('2d');

    // Transparent background (no fill)

    drawLineStimulus(lineCtx, angle, centerX, centerY, radius);

    const linePath = path.join(outputDir, `frame_line_${index}.png`);
    const lineBuffer = lineCanvas.toBuffer('image/png');
    fs.writeFileSync(linePath, lineBuffer);
    console.log(`Generated line frame ${index} at angle ${angle}`);
  });

  console.log(`\nGenerated ${angles.length} color and ${angles.length} line stimuli`);
};

// Generate angles: 1, 11, 21, 31, ..., 221, 231
const angles = [];
for (let angle = 1; angle <= 231; angle += 10) {
  angles.push(angle);
}

const outputDir = path.join(process.cwd(), 'public', 'assets', 'sorting-stim');

console.log(`Generating stimuli for angles: ${angles.join(', ')}`);
console.log(`Output directory: ${outputDir}\n`);

generateSortingStimuli(angles, outputDir);
