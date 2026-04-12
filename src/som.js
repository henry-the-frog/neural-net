// som.js — Self-Organizing Map (Kohonen Network)
// Unsupervised competitive learning, topology-preserving dimensionality reduction

export class SOM {
  constructor(gridWidth, gridHeight, inputDim, {
    learningRate = 0.5,
    sigma = null, // Initial neighborhood radius
    decayRate = 0.99,
  } = {}) {
    this.gridWidth = gridWidth;
    this.gridHeight = gridHeight;
    this.inputDim = inputDim;
    this.learningRate0 = learningRate;
    this.learningRate = learningRate;
    this.sigma0 = sigma || Math.max(gridWidth, gridHeight) / 2;
    this.sigma = this.sigma0;
    this.decayRate = decayRate;
    this.iteration = 0;

    // Initialize weight vectors for each node
    this.weights = [];
    for (let y = 0; y < gridHeight; y++) {
      this.weights[y] = [];
      for (let x = 0; x < gridWidth; x++) {
        this.weights[y][x] = Array.from({ length: inputDim }, () => Math.random());
      }
    }
  }

  // Euclidean distance between input and node weight
  distance(input, nodeWeights) {
    let sum = 0;
    for (let i = 0; i < this.inputDim; i++) {
      sum += (input[i] - nodeWeights[i]) ** 2;
    }
    return Math.sqrt(sum);
  }

  // Find Best Matching Unit (BMU)
  findBMU(input) {
    let minDist = Infinity;
    let bmu = { x: 0, y: 0 };
    for (let y = 0; y < this.gridHeight; y++) {
      for (let x = 0; x < this.gridWidth; x++) {
        const dist = this.distance(input, this.weights[y][x]);
        if (dist < minDist) {
          minDist = dist;
          bmu = { x, y, dist };
        }
      }
    }
    return bmu;
  }

  // Grid distance between two nodes
  gridDistance(x1, y1, x2, y2) {
    return Math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2);
  }

  // Neighborhood function (Gaussian)
  neighborhood(bmuX, bmuY, nodeX, nodeY) {
    const d = this.gridDistance(bmuX, bmuY, nodeX, nodeY);
    return Math.exp(-(d * d) / (2 * this.sigma * this.sigma));
  }

  // Train on single input
  trainStep(input) {
    const bmu = this.findBMU(input);

    // Update all nodes based on distance from BMU
    for (let y = 0; y < this.gridHeight; y++) {
      for (let x = 0; x < this.gridWidth; x++) {
        const h = this.neighborhood(bmu.x, bmu.y, x, y);
        if (h < 0.001) continue; // Skip negligible updates

        for (let i = 0; i < this.inputDim; i++) {
          this.weights[y][x][i] += this.learningRate * h * (input[i] - this.weights[y][x][i]);
        }
      }
    }

    this.iteration++;
  }

  // Decay learning rate and neighborhood radius
  decay() {
    this.learningRate = this.learningRate0 * this.decayRate ** this.iteration;
    this.sigma = Math.max(0.5, this.sigma0 * this.decayRate ** this.iteration);
  }

  // Train on dataset
  train(data, epochs = 100) {
    for (let epoch = 0; epoch < epochs; epoch++) {
      // Shuffle data each epoch
      const shuffled = [...data].sort(() => Math.random() - 0.5);
      for (const input of shuffled) {
        this.trainStep(input);
        this.decay();
      }
    }
  }

  // Map input to grid coordinates
  map(input) {
    return this.findBMU(input);
  }

  // Quantization error (average distance from inputs to their BMU)
  quantizationError(data) {
    let total = 0;
    for (const input of data) {
      const bmu = this.findBMU(input);
      total += bmu.dist;
    }
    return total / data.length;
  }

  // Topographic error (fraction of inputs where 2nd-BMU is not adjacent to BMU)
  topographicError(data) {
    let errors = 0;
    for (const input of data) {
      let best = Infinity, bestPos = null;
      let second = Infinity, secondPos = null;

      for (let y = 0; y < this.gridHeight; y++) {
        for (let x = 0; x < this.gridWidth; x++) {
          const dist = this.distance(input, this.weights[y][x]);
          if (dist < best) {
            second = best; secondPos = bestPos;
            best = dist; bestPos = { x, y };
          } else if (dist < second) {
            second = dist; secondPos = { x, y };
          }
        }
      }

      if (bestPos && secondPos) {
        const gridDist = this.gridDistance(bestPos.x, bestPos.y, secondPos.x, secondPos.y);
        if (gridDist > 1.5) errors++; // Not adjacent
      }
    }
    return errors / data.length;
  }

  // U-Matrix (unified distance matrix) — shows cluster boundaries
  uMatrix() {
    const umat = [];
    for (let y = 0; y < this.gridHeight; y++) {
      umat[y] = [];
      for (let x = 0; x < this.gridWidth; x++) {
        let sum = 0, count = 0;
        // Average distance to neighbors
        for (const [dy, dx] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
          const ny = y + dy, nx = x + dx;
          if (ny >= 0 && ny < this.gridHeight && nx >= 0 && nx < this.gridWidth) {
            sum += this.distance(this.weights[y][x], this.weights[ny][nx]);
            count++;
          }
        }
        umat[y][x] = count > 0 ? sum / count : 0;
      }
    }
    return umat;
  }

  // Component planes — show how each input dimension distributes across the map
  componentPlane(dim) {
    const plane = [];
    for (let y = 0; y < this.gridHeight; y++) {
      plane[y] = [];
      for (let x = 0; x < this.gridWidth; x++) {
        plane[y][x] = this.weights[y][x][dim];
      }
    }
    return plane;
  }

  // ASCII visualization of U-Matrix
  visualize() {
    const umat = this.uMatrix();
    const maxU = Math.max(...umat.flat());
    const chars = ' ░▒▓█';
    const lines = [];
    for (let y = 0; y < this.gridHeight; y++) {
      let line = '';
      for (let x = 0; x < this.gridWidth; x++) {
        const norm = maxU > 0 ? umat[y][x] / maxU : 0;
        const idx = Math.min(Math.floor(norm * chars.length), chars.length - 1);
        line += chars[idx];
      }
      lines.push(line);
    }
    return lines.join('\n');
  }

  // Get node count
  get nodeCount() {
    return this.gridWidth * this.gridHeight;
  }
}

// ===== Growing SOM (GSOM) =====
// Dynamic topology — grows as needed
export class GrowingSOM {
  constructor(inputDim, { growthThreshold = 0.5, learningRate = 0.3 } = {}) {
    this.inputDim = inputDim;
    this.growthThreshold = growthThreshold;
    this.learningRate = learningRate;

    // Start with a 2x2 grid
    this.nodes = new Map(); // key: "x,y" → weights
    for (let y = 0; y < 2; y++) {
      for (let x = 0; x < 2; x++) {
        this.nodes.set(`${x},${y}`, {
          x, y,
          weights: Array.from({ length: inputDim }, () => Math.random()),
          error: 0,
        });
      }
    }
  }

  findBMU(input) {
    let minDist = Infinity;
    let bmu = null;
    for (const node of this.nodes.values()) {
      let dist = 0;
      for (let i = 0; i < this.inputDim; i++) {
        dist += (input[i] - node.weights[i]) ** 2;
      }
      dist = Math.sqrt(dist);
      if (dist < minDist) {
        minDist = dist;
        bmu = node;
      }
    }
    return { node: bmu, dist: minDist };
  }

  trainStep(input) {
    const { node: bmu, dist } = this.findBMU(input);

    // Update BMU and neighbors
    for (const node of this.nodes.values()) {
      const d = Math.sqrt((node.x - bmu.x) ** 2 + (node.y - bmu.y) ** 2);
      if (d <= 1.5) { // Direct neighbors
        const h = d === 0 ? 1 : 0.5;
        for (let i = 0; i < this.inputDim; i++) {
          node.weights[i] += this.learningRate * h * (input[i] - node.weights[i]);
        }
      }
    }

    // Accumulate error
    bmu.error += dist;
  }

  // Grow: add nodes at boundaries of high-error nodes
  grow() {
    const toAdd = [];
    for (const node of this.nodes.values()) {
      if (node.error > this.growthThreshold) {
        // Try to add neighbors in each direction
        for (const [dx, dy] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
          const key = `${node.x + dx},${node.y + dy}`;
          if (!this.nodes.has(key)) {
            toAdd.push({
              x: node.x + dx, y: node.y + dy,
              weights: [...node.weights],
            });
          }
        }
        node.error = 0;
      }
    }

    for (const newNode of toAdd) {
      this.nodes.set(`${newNode.x},${newNode.y}`, {
        ...newNode,
        error: 0,
      });
    }

    return toAdd.length;
  }

  train(data, epochs = 50, growInterval = 10) {
    for (let epoch = 0; epoch < epochs; epoch++) {
      for (const input of data) {
        this.trainStep(input);
      }
      if ((epoch + 1) % growInterval === 0) {
        this.grow();
      }
    }
  }

  get nodeCount() {
    return this.nodes.size;
  }
}
