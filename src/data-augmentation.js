// data-augmentation.js — Basic data augmentation for training
export function addGaussianNoise(data, std = 0.1) {
  return data.map(x => x + std * (Math.random() * 2 - 1));
}

export function mixup(x1, y1, x2, y2, alpha = 0.2) {
  const lambda = alpha > 0 ? betaSample(alpha, alpha) : 1;
  const x = x1.map((v, i) => lambda * v + (1 - lambda) * x2[i]);
  const y = lambda * y1 + (1 - lambda) * y2;
  return { x, y };
}

export function cutmix(x1, x2, alpha = 1.0) {
  const lambda = Math.random();
  const cutLen = Math.floor(x1.length * (1 - lambda));
  const start = Math.floor(Math.random() * (x1.length - cutLen));
  const mixed = [...x1];
  for (let i = start; i < start + cutLen; i++) mixed[i] = x2[i];
  return { mixed, lambda };
}

function betaSample(a, b) {
  // Simplified: use uniform approximation for small alpha
  return Math.random();
}
