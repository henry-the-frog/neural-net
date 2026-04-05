import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { Graph, GCNLayer, GNN, createKarateClub } from '../src/gnn.js';

describe('Graph Neural Network', () => {
  describe('Graph', () => {
    it('should create a graph with adjacency list', () => {
      const g = new Graph(4, [[0,1],[1,2],[2,3]]);
      assert.equal(g.numNodes, 4);
      assert.ok(g.adjList[1].includes(0));
      assert.ok(g.adjList[1].includes(2));
    });

    it('should compute neighbors with self-loop', () => {
      const g = new Graph(3, [[0,1],[1,2]]);
      const neighbors = g.neighbors(1);
      assert.ok(neighbors.includes(0));
      assert.ok(neighbors.includes(1)); // self-loop
      assert.ok(neighbors.includes(2));
    });

    it('should compute degree', () => {
      const g = new Graph(3, [[0,1],[0,2]]);
      assert.equal(g.degree(0), 2);
      assert.equal(g.degree(1), 1);
    });

    it('should handle default node features', () => {
      const g = new Graph(3, []);
      assert.equal(g.featureDim, 1);
    });

    it('should accept custom node features', () => {
      const g = new Graph(3, [], [[1,0,0],[0,1,0],[0,0,1]]);
      assert.equal(g.featureDim, 3);
      assert.deepStrictEqual(g.nodeFeatures[1], [0,1,0]);
    });
  });

  describe('GCNLayer', () => {
    it('should transform features', () => {
      const layer = new GCNLayer(3, 2);
      const g = new Graph(3, [[0,1],[1,2]], [[1,0,0],[0,1,0],[0,0,1]]);
      const output = layer.forward(g, g.nodeFeatures);
      assert.equal(output.length, 3);
      assert.equal(output[0].length, 2);
    });

    it('should aggregate neighbor information', () => {
      const layer = new GCNLayer(2, 2, 'linear');
      // Set identity weights
      layer.W.data = new Float64Array([1,0,0,1]);
      layer.b.data = new Float64Array([0,0]);

      const g = new Graph(3, [[0,1],[1,2]], [[1,0],[0,1],[1,1]]);
      const output = layer.forward(g, g.nodeFeatures);
      // Node 1 aggregates from {0, 1, 2}: mean of [1,0], [0,1], [1,1] = [2/3, 2/3]
      assert.ok(Math.abs(output[1][0] - 2/3) < 0.01);
      assert.ok(Math.abs(output[1][1] - 2/3) < 0.01);
    });
  });

  describe('GNN', () => {
    it('should create multi-layer GNN', () => {
      const gnn = new GNN([3, 4, 2]);
      assert.equal(gnn.layers.length, 2);
    });

    it('should forward through layers', () => {
      const gnn = new GNN([3, 4, 2]);
      const g = new Graph(4, [[0,1],[1,2],[2,3]], [[1,0,0],[0,1,0],[0,0,1],[1,1,0]]);
      const embeddings = gnn.forward(g);
      assert.equal(embeddings.length, 4);
      assert.equal(embeddings[0].length, 2);
    });

    it('should predict node classes', () => {
      const gnn = new GNN([3, 4, 2]);
      const g = new Graph(4, [[0,1],[1,2],[2,3]], [[1,0,0],[0,1,0],[0,0,1],[1,1,0]]);
      const predictions = gnn.predict(g);
      assert.equal(predictions.length, 4);
      for (const p of predictions) {
        assert.ok(p === 0 || p === 1);
      }
    });

    it('should compute graph-level embedding', () => {
      const gnn = new GNN([3, 4, 2]);
      const g = new Graph(3, [[0,1],[1,2]], [[1,0,0],[0,1,0],[0,0,1]]);
      const pooled = gnn.graphEmbed(g);
      assert.equal(pooled.length, 2);
      assert.ok(!pooled.some(isNaN));
    });

    it('should train on node classification', () => {
      const gnn = new GNN([4, 8, 2], { learningRate: 0.01 });
      const { graph, labels } = createKarateClub();

      // Use subset of labels for semi-supervised learning
      const trainLabels = new Map();
      trainLabels.set(0, 0);  // Known community 0
      trainLabels.set(33, 1); // Known community 1
      trainLabels.set(1, 0);
      trainLabels.set(32, 1);

      const { history } = gnn.train(graph, {
        labels: trainLabels,
        epochs: 50,
      });

      assert.equal(history.length, 50);
      // Loss should decrease
      assert.ok(history[history.length - 1] < history[0],
        `Loss should decrease: ${history[0].toFixed(4)} → ${history[history.length - 1].toFixed(4)}`);
    });

    it('should use onEpoch callback', () => {
      const gnn = new GNN([1, 2, 2]);
      const g = new Graph(3, [[0,1],[1,2]]);
      const labels = new Map([[0, 0], [2, 1]]);
      const calls = [];
      gnn.train(g, { labels, epochs: 3, onEpoch: d => calls.push(d) });
      assert.equal(calls.length, 3);
    });
  });

  describe('Karate Club', () => {
    it('should create karate club graph', () => {
      const { graph, labels } = createKarateClub();
      assert.equal(graph.numNodes, 34);
      assert.equal(labels.size, 34);
      assert.equal(graph.featureDim, 4);
    });

    it('should have correct community structure', () => {
      const { labels } = createKarateClub();
      assert.equal(labels.get(0), 0); // Mr. Hi
      assert.equal(labels.get(33), 1); // Officer
    });
  });
});

describe('GNN Edge Cases', () => {
  it('should handle disconnected graph', () => {
    const gnn = new GNN([2, 3, 2]);
    const g = new Graph(4, [[0,1]], [[1,0],[0,1],[1,1],[0,0]]);
    const embeddings = gnn.forward(g);
    assert.equal(embeddings.length, 4);
    assert.ok(!embeddings.flat().some(isNaN));
  });

  it('should handle single-node graph', () => {
    const gnn = new GNN([1, 2]);
    const g = new Graph(1, [], [[5]]);
    const embeddings = gnn.forward(g);
    assert.equal(embeddings.length, 1);
    assert.equal(embeddings[0].length, 2);
  });
});
