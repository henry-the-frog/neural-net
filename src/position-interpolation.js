// position-interpolation.js — Position Interpolation for extending context
export function linearInterpolation(position, scaleFactor) {
  return position / scaleFactor;
}

export function dynamicNTK(position, maxTrainLen, currentLen) {
  if (currentLen <= maxTrainLen) return position;
  const scaleFactor = currentLen / maxTrainLen;
  return position / scaleFactor;
}

// Code Llama style: gradual scaling based on position
export function gradualScaling(position, maxTrainLen, threshold = 0.8) {
  if (position < maxTrainLen * threshold) return position;
  const excess = position - maxTrainLen * threshold;
  const range = maxTrainLen * (1 - threshold);
  const scale = 1 + excess / range;
  return position / scale;
}
