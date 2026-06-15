/**
 * Reusable spatial acceleration for vector features.
 * SpatialGridIndex limits viewport and pointer queries to nearby features.
 */
window.Foundation = window.Foundation || {};

Foundation.SpatialGridIndex = class SpatialGridIndex {
  constructor(features, getBounds, cellSize = 10) {
    this.getBounds = getBounds;
    this.cellSize = cellSize;
    this.cells = new Map();
    for (const feature of features) this.insert(feature);
  }

  insert(feature) {
    const bounds = this.getBounds(feature);
    if (!bounds) return;
    const [minX, minY, maxX, maxY] = bounds;
    for (let x = Math.floor(minX / this.cellSize); x <= Math.floor(maxX / this.cellSize); x++) {
      for (let y = Math.floor(minY / this.cellSize); y <= Math.floor(maxY / this.cellSize); y++) {
        const key = `${x}:${y}`;
        if (!this.cells.has(key)) this.cells.set(key, []);
        this.cells.get(key).push(feature);
      }
    }
  }

  queryPoint(x, y) {
    return this.cells.get(`${Math.floor(x / this.cellSize)}:${Math.floor(y / this.cellSize)}`) || [];
  }

  queryBounds(minX, minY, maxX, maxY) {
    const results = new Set();
    for (let x = Math.floor(minX / this.cellSize); x <= Math.floor(maxX / this.cellSize); x++) {
      for (let y = Math.floor(minY / this.cellSize); y <= Math.floor(maxY / this.cellSize); y++) {
        for (const feature of this.cells.get(`${x}:${y}`) || []) results.add(feature);
      }
    }
    return [...results];
  }
};
