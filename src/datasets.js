// datasets.js — Synthetic dataset generators for testing and demos
//
// Usage:
//   import { Datasets } from './datasets.js';
//   const { inputs, targets } = Datasets.spiral(200, 3);
//   const { inputs, targets } = Datasets.moons(200);
//   const { inputs, targets } = Datasets.circles(200);

import { Matrix } from './matrix.js';

export const Datasets = {
  /**
   * XOR dataset: 4 points, classic non-linear test
   */
  xor() {
    return {
      inputs: Matrix.fromArray([[0, 0], [0, 1], [1, 0], [1, 1]]),
      targets: Matrix.fromArray([[0], [1], [1], [0]]),
      description: 'XOR: 4 points, 2 classes',
    };
  },

  /**
   * Spiral dataset: N points per class, wound into spirals
   * Good for testing non-linear classifiers
   */
  spiral(pointsPerClass = 100, numClasses = 2) {
    const N = pointsPerClass * numClasses;
    const inputs = Matrix.zeros(N, 2);
    const targets = Matrix.zeros(N, numClasses);
    
    let idx = 0;
    for (let c = 0; c < numClasses; c++) {
      for (let i = 0; i < pointsPerClass; i++) {
        const r = i / pointsPerClass;
        const t = c * 4 + r * 4 + (Math.random() - 0.5) * 0.2;
        inputs.set(idx, 0, r * Math.sin(t * 2.5));
        inputs.set(idx, 1, r * Math.cos(t * 2.5));
        targets.set(idx, c, 1); // One-hot
        idx++;
      }
    }
    
    return { inputs, targets, numClasses, description: `Spiral: ${N} points, ${numClasses} classes` };
  },

  /**
   * Two moons: two interleaving half-circles
   */
  moons(n = 200, noise = 0.1) {
    const inputs = Matrix.zeros(n, 2);
    const targets = Matrix.zeros(n, 1);
    
    for (let i = 0; i < n; i++) {
      if (i < n / 2) {
        const theta = Math.PI * i / (n / 2);
        inputs.set(i, 0, Math.cos(theta) + (Math.random() - 0.5) * noise);
        inputs.set(i, 1, Math.sin(theta) + (Math.random() - 0.5) * noise);
        targets.set(i, 0, 0);
      } else {
        const theta = Math.PI * (i - n / 2) / (n / 2);
        inputs.set(i, 0, 1 - Math.cos(theta) + (Math.random() - 0.5) * noise);
        inputs.set(i, 1, 0.5 - Math.sin(theta) + (Math.random() - 0.5) * noise);
        targets.set(i, 0, 1);
      }
    }
    
    return { inputs, targets, description: `Moons: ${n} points, 2 classes` };
  },

  /**
   * Concentric circles: inner class 0, outer class 1
   */
  circles(n = 200, noise = 0.05) {
    const inputs = Matrix.zeros(n, 2);
    const targets = Matrix.zeros(n, 1);
    
    for (let i = 0; i < n; i++) {
      const theta = Math.random() * 2 * Math.PI;
      if (i < n / 2) {
        const r = 0.5 + (Math.random() - 0.5) * noise;
        inputs.set(i, 0, r * Math.cos(theta));
        inputs.set(i, 1, r * Math.sin(theta));
        targets.set(i, 0, 0);
      } else {
        const r = 1.0 + (Math.random() - 0.5) * noise;
        inputs.set(i, 0, r * Math.cos(theta));
        inputs.set(i, 1, r * Math.sin(theta));
        targets.set(i, 0, 1);
      }
    }
    
    return { inputs, targets, description: `Circles: ${n} points, 2 classes` };
  },

  /**
   * Gaussian blobs: N clusters with Gaussian noise
   */
  blobs(n = 200, numBlobs = 3, spread = 0.5) {
    const pointsPerBlob = Math.floor(n / numBlobs);
    const total = pointsPerBlob * numBlobs;
    const inputs = Matrix.zeros(total, 2);
    const targets = Matrix.zeros(total, numBlobs);
    
    // Place blob centers evenly around a circle
    const centers = [];
    for (let c = 0; c < numBlobs; c++) {
      const angle = (2 * Math.PI * c) / numBlobs;
      centers.push([2 * Math.cos(angle), 2 * Math.sin(angle)]);
    }
    
    let idx = 0;
    for (let c = 0; c < numBlobs; c++) {
      for (let i = 0; i < pointsPerBlob; i++) {
        inputs.set(idx, 0, centers[c][0] + gaussianRandom() * spread);
        inputs.set(idx, 1, centers[c][1] + gaussianRandom() * spread);
        targets.set(idx, c, 1); // One-hot
        idx++;
      }
    }
    
    return { inputs, targets, numBlobs, description: `Blobs: ${total} points, ${numBlobs} clusters` };
  },

  /**
   * Sine regression: y = sin(x) on [-π, π]
   */
  sine(n = 100) {
    const inputs = Matrix.zeros(n, 1);
    const targets = Matrix.zeros(n, 1);
    
    for (let i = 0; i < n; i++) {
      const x = -Math.PI + (2 * Math.PI * i) / (n - 1);
      inputs.set(i, 0, x / Math.PI); // Normalize to [-1, 1]
      targets.set(i, 0, Math.sin(x));
    }
    
    return { inputs, targets, description: `Sine: ${n} points, regression` };
  },

  /**
   * Linear regression: y = 2x + 1 + noise
   */
  linear(n = 100, noise = 0.1) {
    const inputs = Matrix.zeros(n, 1);
    const targets = Matrix.zeros(n, 1);
    
    for (let i = 0; i < n; i++) {
      const x = -1 + 2 * i / (n - 1);
      inputs.set(i, 0, x);
      targets.set(i, 0, 2 * x + 1 + gaussianRandom() * noise);
    }
    
    return { inputs, targets, description: `Linear: ${n} points, y = 2x + 1` };
  },
};

function gaussianRandom() {
  // Box-Muller transform
  let u, v;
  do { u = Math.random(); } while (u === 0);
  do { v = Math.random(); } while (v === 0);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
