/**
 * gnn.js — Graph Neural Network
 * 
 * Implements message-passing Graph Neural Networks for:
 * - Node classification
 * - Graph-level prediction
 * - Link prediction
 * 
 * Architecture: GCN (Graph Convolutional Network) style
 *   h_v^(k+1) = σ(W · AGGREGATE({h_u^(k) : u ∈ N(v) ∪ {v}}))
 * 
 * Based on: Kipf & Welling (2017), "Semi-Supervised Classification with GCNs"
 */

import { Matrix } from './matrix.js';

function relu(x) { return Math.max(0, x); }
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

/**
 * Simple Graph representation
 */
export class Graph {
  /**
   * @param {number} numNodes
   * @param {number[][]} edges - Array of [src, dst] pairs
   * @param {number[][]} [nodeFeatures] - Feature vectors per node
   */
  constructor(numNodes, edges = [], nodeFeatures = null) {
    this.numNodes = numNodes;
    this.edges = edges;

    // Build adjacency lists
    this.adjList = Array.from({ length: numNodes }, () => []);
    for (const [src, dst] of edges) {
      this.adjList[src].push(dst);
      this.adjList[dst].push(src); // Undirected
    }

    // Node features
    this.nodeFeatures = nodeFeatures || Array.from({ length: numNodes }, () => [1]);
  }

  /**
   * Get neighbors of a node (including self-loop).
   */
  neighbors(nodeId) {
    return [...new Set([nodeId, ...this.adjList[nodeId]])];
  }

  /**
   * Get degree of a node.
   */
  degree(nodeId) {
    return this.adjList[nodeId].length;
  }

  /**
   * Get feature dimension.
   */
  get featureDim() {
    return this.nodeFeatures[0].length;
  }
}

/**
 * GCN Layer — Graph Convolutional Network layer.
 * 
 * Aggregates neighbor features, applies linear transformation + activation.
 * h_v' = σ(W · (1/|N(v)| · Σ_{u∈N(v)} h_u) + b)
 */
export class GCNLayer {
  /**
   * @param {number} inputDim - Input feature dimension
   * @param {number} outputDim - Output feature dimension
   * @param {string} [activation='relu']
   */
  constructor(inputDim, outputDim, activation = 'relu') {
    this.inputDim = inputDim;
    this.outputDim = outputDim;
    this.activation = activation;

    // Xavier initialization
    const scale = Math.sqrt(2 / (inputDim + outputDim));
    this.W = new Matrix(outputDim, inputDim);
    for (let i = 0; i < this.W.data.length; i++) {
      this.W.data[i] = (Math.random() * 2 - 1) * scale;
    }
    this.b = new Matrix(outputDim, 1);
  }

  /**
   * Forward pass: aggregate neighbor features and transform.
   * @param {Graph} graph
   * @param {number[][]} nodeEmbeddings - Current embeddings [numNodes][inputDim]
   * @returns {number[][]} New embeddings [numNodes][outputDim]
   */
  forward(graph, nodeEmbeddings) {
    const newEmbeddings = [];

    for (let v = 0; v < graph.numNodes; v++) {
      const neighbors = graph.neighbors(v);
      const degree = neighbors.length;

      // Aggregate: mean of neighbor features
      const aggregated = new Float64Array(this.inputDim);
      for (const u of neighbors) {
        const features = nodeEmbeddings[u];
        for (let d = 0; d < this.inputDim; d++) {
          aggregated[d] += features[d] / degree;
        }
      }

      // Transform: W · aggregated + b
      const input = new Matrix(this.inputDim, 1, aggregated);
      const output = this.W.dot(input).add(this.b);

      // Activation
      const activated = new Array(this.outputDim);
      for (let d = 0; d < this.outputDim; d++) {
        const val = output.data[d];
        if (this.activation === 'relu') activated[d] = relu(val);
        else if (this.activation === 'sigmoid') activated[d] = sigmoid(val);
        else activated[d] = val;
      }

      newEmbeddings.push(activated);
    }

    return newEmbeddings;
  }
}

/**
 * Graph Neural Network — stacked GCN layers.
 */
export class GNN {
  /**
   * @param {number[]} layerDims - [inputDim, hidden1, hidden2, ..., outputDim]
   * @param {Object} opts
   * @param {number} [opts.learningRate=0.01]
   */
  constructor(layerDims, opts = {}) {
    this.learningRate = opts.learningRate || 0.01;
    this.layers = [];

    for (let i = 0; i < layerDims.length - 1; i++) {
      const isLast = i === layerDims.length - 2;
      this.layers.push(new GCNLayer(
        layerDims[i],
        layerDims[i + 1],
        isLast ? 'linear' : 'relu',
      ));
    }
  }

  /**
   * Forward pass: propagate through all GCN layers.
   * @param {Graph} graph
   * @returns {number[][]} Final node embeddings
   */
  forward(graph) {
    let embeddings = graph.nodeFeatures.map(f => [...f]);
    for (const layer of this.layers) {
      embeddings = layer.forward(graph, embeddings);
    }
    return embeddings;
  }

  /**
   * Predict node classes (argmax of output embeddings).
   * @param {Graph} graph
   * @returns {number[]}
   */
  predict(graph) {
    const embeddings = this.forward(graph);
    return embeddings.map(emb => {
      let maxIdx = 0;
      for (let i = 1; i < emb.length; i++) {
        if (emb[i] > emb[maxIdx]) maxIdx = i;
      }
      return maxIdx;
    });
  }

  /**
   * Train on node classification (simple gradient descent).
   * Uses cross-entropy loss on labeled nodes.
   * @param {Graph} graph
   * @param {Object} opts
   * @param {Map<number, number>} opts.labels - nodeId → classLabel
   * @param {number} [opts.epochs=100]
   * @param {Function} [opts.onEpoch]
   * @returns {{ history: number[] }}
   */
  train(graph, opts = {}) {
    const { labels, epochs = 100, onEpoch } = opts;
    const history = [];
    const labeledNodes = [...labels.entries()];
    const numClasses = Math.max(...labels.values()) + 1;

    for (let epoch = 0; epoch < epochs; epoch++) {
      const embeddings = this.forward(graph);

      // Compute softmax + cross-entropy loss for labeled nodes
      let totalLoss = 0;

      for (const [nodeId, trueLabel] of labeledNodes) {
        const logits = embeddings[nodeId];

        // Softmax
        const maxLogit = Math.max(...logits);
        const exps = logits.map(l => Math.exp(l - maxLogit));
        const sumExps = exps.reduce((a, b) => a + b, 0);
        const probs = exps.map(e => e / sumExps);

        // Cross-entropy loss
        totalLoss -= Math.log(Math.max(probs[trueLabel], 1e-10));

        // Gradient: d_loss/d_logit = prob - one_hot
        const grad = probs.map((p, i) => p - (i === trueLabel ? 1 : 0));

        // Simple weight update for the last layer (approximate)
        const lastLayer = this.layers[this.layers.length - 1];
        const prevEmb = this.layers.length > 1
          ? this.layers[this.layers.length - 2].forward(graph,
              graph.nodeFeatures.map(f => [...f]))[nodeId]
          : graph.nodeFeatures[nodeId];

        // dW = grad · prevEmb^T
        for (let i = 0; i < lastLayer.outputDim; i++) {
          for (let j = 0; j < lastLayer.inputDim; j++) {
            lastLayer.W.data[i * lastLayer.inputDim + j] -=
              this.learningRate * grad[i] * prevEmb[j];
          }
          lastLayer.b.data[i] -= this.learningRate * grad[i];
        }
      }

      const avgLoss = totalLoss / labeledNodes.length;
      history.push(avgLoss);
      if (onEpoch) onEpoch({ epoch, loss: avgLoss });
    }

    return { history };
  }

  /**
   * Compute graph-level embedding (mean pool over node embeddings).
   * @param {Graph} graph
   * @returns {number[]}
   */
  graphEmbed(graph) {
    const embeddings = this.forward(graph);
    const dim = embeddings[0].length;
    const pooled = new Array(dim).fill(0);
    for (const emb of embeddings) {
      for (let d = 0; d < dim; d++) pooled[d] += emb[d] / graph.numNodes;
    }
    return pooled;
  }
}

/**
 * Create common test graphs.
 */
export function createKarateClub() {
  // Zachary's Karate Club — classic social network benchmark
  // 34 nodes, 2 communities
  const edges = [
    [0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[0,10],[0,11],[0,12],[0,13],[0,17],[0,19],[0,21],[0,31],
    [1,2],[1,3],[1,7],[1,13],[1,17],[1,19],[1,21],[1,30],
    [2,3],[2,7],[2,8],[2,9],[2,13],[2,27],[2,28],[2,32],
    [3,7],[3,12],[3,13],
    [4,6],[4,10],
    [5,6],[5,10],[5,16],
    [6,16],
    [8,30],[8,32],[8,33],
    [9,33],
    [13,33],
    [14,32],[14,33],
    [15,32],[15,33],
    [18,32],[18,33],
    [19,33],
    [20,32],[20,33],
    [22,32],[22,33],
    [23,25],[23,27],[23,29],[23,32],[23,33],
    [24,25],[24,27],[24,31],
    [25,31],
    [26,29],[26,33],
    [27,33],
    [28,31],[28,33],
    [29,32],[29,33],
    [30,32],[30,33],
    [31,32],[31,33],
    [32,33],
  ];

  // Ground truth: two communities (0 = Mr. Hi's, 1 = Officer's)
  const labels = new Map();
  const community0 = [0,1,2,3,4,5,6,7,8,10,11,12,13,16,17,19,21];
  const community1 = [9,14,15,18,20,22,23,24,25,26,27,28,29,30,31,32,33];
  for (const n of community0) labels.set(n, 0);
  for (const n of community1) labels.set(n, 1);

  // One-hot degree features (binned)
  const graph = new Graph(34, edges);
  const features = Array.from({ length: 34 }, (_, i) => {
    const deg = graph.degree(i);
    // Bin degree into 4 categories
    if (deg <= 2) return [1, 0, 0, 0];
    if (deg <= 5) return [0, 1, 0, 0];
    if (deg <= 10) return [0, 0, 1, 0];
    return [0, 0, 0, 1];
  });
  
  return { graph: new Graph(34, edges, features), labels };
}
