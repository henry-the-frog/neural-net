// gnn-stress.test.js — Stress tests for Graph Neural Networks
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Graph, GCNLayer, GNN, createKarateClub } from '../src/gnn.js';

function makeTriangle() {
  return new Graph(3, [[0, 1], [1, 2], [0, 2]], [[1, 0], [0, 1], [1, 1]]);
}

describe('GCN Layer', () => {
  it('output has correct shape', () => {
    const g = makeTriangle();
    const layer = new GCNLayer(2, 4);
    const output = layer.forward(g, g.nodeFeatures);
    assert.equal(output.length, 3);
    assert.equal(output[0].length, 4);
  });

  it('output is finite', () => {
    const g = makeTriangle();
    const layer = new GCNLayer(2, 4);
    const output = layer.forward(g, g.nodeFeatures);
    for (const vec of output) assert.ok(vec.every(Number.isFinite));
  });

  it('different graphs produce different outputs', () => {
    const g1 = makeTriangle();
    const g2 = new Graph(3, [[0, 1]], [[1, 0], [0, 1], [1, 1]]);
    const layer = new GCNLayer(2, 3);
    const out1 = layer.forward(g1, g1.nodeFeatures);
    const out2 = layer.forward(g2, g2.nodeFeatures);
    let diff = 0;
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) diff += Math.abs(out1[i][j] - out2[i][j]);
    assert.ok(diff > 0.01, 'Different graphs should differ');
  });
});

describe('GNN Multi-Layer', () => {
  it('forward produces finite output', () => {
    const g = makeTriangle();
    const gnn = new GNN([2, 4, 3]);
    const output = gnn.forward(g);
    assert.equal(output.length, 3);
    for (const vec of output) assert.ok(vec.every(Number.isFinite));
  });

  it('Karate Club graph works', () => {
    const { graph: g, labels } = createKarateClub();
    assert.ok(g.numNodes > 30);
    const gnn = new GNN([g.nodeFeatures[0].length, 16, 2]);
    const output = gnn.forward(g);
    assert.equal(output.length, g.numNodes);
    for (const vec of output) assert.ok(vec.every(Number.isFinite));
  });

  it('3 GCN layers', () => {
    const g = makeTriangle();
    const gnn = new GNN([2, 4, 4, 3]);
    const output = gnn.forward(g);
    assert.equal(output.length, 3);
    for (const vec of output) assert.ok(vec.every(Number.isFinite));
  });
});

describe('Edge Cases', () => {
  it('star graph', () => {
    const edges = [[0, 1], [0, 2], [0, 3], [0, 4]];
    const feats = [[1, 0], [0, 1], [1, 1], [0, 0], [1, 0]];
    const g = new Graph(5, edges, feats);
    const gnn = new GNN([2, 3]);
    const output = gnn.forward(g);
    assert.equal(output.length, 5);
    for (const vec of output) assert.ok(vec.every(Number.isFinite));
  });

  it('complete graph', () => {
    const edges = [];
    for (let i = 0; i < 5; i++)
      for (let j = i + 1; j < 5; j++) edges.push([i, j]);
    const feats = Array.from({ length: 5 }, (_, i) => [i / 4, 1 - i / 4]);
    const g = new Graph(5, edges, feats);
    const gnn = new GNN([2, 3]);
    const output = gnn.forward(g);
    for (const vec of output) assert.ok(vec.every(Number.isFinite));
  });
});
