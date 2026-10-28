import { JSDOM } from "jsdom";

const dom = new JSDOM("<!DOCTYPE html>");
global.window = dom.window;
global.document = dom.window.document;

// Mock Image loading
global.Image = class {
  constructor() {
    // The 'onload' callback is often used in preloading, so we
    // can simulate it being called immediately.
    setTimeout(() => {
      if (this.onload) {
        this.onload();
      }
    }, 5); // Use a small timeout to simulate async behavior
  }
};

// Mock requestAnimationFrame and cancelAnimationFrame
let rafId = 0;
global.requestAnimationFrame = (callback) => {
  rafId++;
  // Call immediately for simulation (don't wait for next frame)
  setTimeout(() => callback(performance.now()), 0);
  return rafId;
};
global.cancelAnimationFrame = (id) => {
  // No-op for simulation
};

// Mock canvas methods that might be needed
global.HTMLCanvasElement = dom.window.HTMLCanvasElement;
global.CanvasRenderingContext2D = dom.window.CanvasRenderingContext2D;

// Mock performance.now if not available
if (!global.performance) {
  global.performance = {
    now: () => Date.now()
  };
}
