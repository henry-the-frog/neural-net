// digits.js — Simple 5×5 pixel digit dataset for training
// Each digit is a 5×5 binary grid (25 inputs)

import { Matrix } from './matrix.js';

// 5×5 pixel representations of digits 0-9
const DIGIT_PATTERNS = [
  // 0
  [0,1,1,1,0,
   1,0,0,0,1,
   1,0,0,0,1,
   1,0,0,0,1,
   0,1,1,1,0],
  // 1
  [0,0,1,0,0,
   0,1,1,0,0,
   0,0,1,0,0,
   0,0,1,0,0,
   0,1,1,1,0],
  // 2
  [0,1,1,1,0,
   0,0,0,0,1,
   0,1,1,1,0,
   1,0,0,0,0,
   1,1,1,1,1],
  // 3
  [1,1,1,1,0,
   0,0,0,0,1,
   0,1,1,1,0,
   0,0,0,0,1,
   1,1,1,1,0],
  // 4
  [1,0,0,1,0,
   1,0,0,1,0,
   1,1,1,1,1,
   0,0,0,1,0,
   0,0,0,1,0],
  // 5
  [1,1,1,1,1,
   1,0,0,0,0,
   1,1,1,1,0,
   0,0,0,0,1,
   1,1,1,1,0],
  // 6
  [0,1,1,1,0,
   1,0,0,0,0,
   1,1,1,1,0,
   1,0,0,0,1,
   0,1,1,1,0],
  // 7
  [1,1,1,1,1,
   0,0,0,1,0,
   0,0,1,0,0,
   0,1,0,0,0,
   0,1,0,0,0],
  // 8
  [0,1,1,1,0,
   1,0,0,0,1,
   0,1,1,1,0,
   1,0,0,0,1,
   0,1,1,1,0],
  // 9
  [0,1,1,1,0,
   1,0,0,0,1,
   0,1,1,1,1,
   0,0,0,0,1,
   0,1,1,1,0],
];

// Generate training data with noise augmentation
export function generateDigitDataset(samplesPerDigit = 50) {
  const inputs = [];
  const labels = [];

  for (let digit = 0; digit < 10; digit++) {
    const pattern = DIGIT_PATTERNS[digit];

    for (let s = 0; s < samplesPerDigit; s++) {
      // Add noise: flip random pixels with probability
      const noisy = pattern.map(p => {
        if (Math.random() < 0.1) return p ? 0 : 1; // 10% noise
        return p + (Math.random() - 0.5) * 0.3; // Continuous noise
      });
      inputs.push(noisy);
      labels.push(digit);
    }
  }

  // Shuffle
  const indices = Array.from({ length: inputs.length }, (_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  const shuffledInputs = indices.map(i => inputs[i]);
  const shuffledLabels = indices.map(i => labels[i]);

  return {
    inputs: Matrix.fromArray(shuffledInputs),
    targets: Matrix.oneHot(shuffledLabels, 10),
    labels: shuffledLabels,
    patterns: DIGIT_PATTERNS
  };
}

export { DIGIT_PATTERNS };
