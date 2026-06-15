/**
 * Foundation Layer Manager
 * Manages layer stack, ordering, and rendering
 */
window.Foundation = window.Foundation || {};

Foundation.LayerManager = class LayerManager {
  constructor(eventBus) {
    this.eventBus = eventBus;
    this.layers = new Map();
    this.order = [];

    // Built-in layer types
    this.layerTypes = {
      basemap: 'basemap',
      vector: 'vector',
      raster: 'raster',
      tile: 'tile'
    };
  }

  /**
   * Add a layer
   */
  addLayer(config) {
    const layer = {
      id: config.id,
      name: config.name || config.id,
      type: config.type || 'vector',
      visible: config.visible !== false,
      opacity: config.opacity || 1,
      interactive: config.interactive || false,
      moduleId: config.moduleId || null,
      source: config.source,
      renderer: config.renderer,
      hitTest: config.hitTest,
      style: config.style || {},
      metadata: config.metadata || {}
    };

    if (this.layers.has(layer.id)) {
      console.warn(`Layer "${layer.id}" already exists, replacing`);
      this.removeLayer(layer.id);
    }

    this.layers.set(layer.id, layer);
    this.order.push(layer.id);

    this.eventBus.emit(Foundation.Events.LAYER_ADD, { layer });

    return layer;
  }

  /**
   * Remove a layer
   */
  removeLayer(layerId) {
    if (!this.layers.has(layerId)) return false;

    this.layers.delete(layerId);
    this.order = this.order.filter(id => id !== layerId);

    this.eventBus.emit(Foundation.Events.LAYER_REMOVE, { layerId });

    return true;
  }

  /**
   * Set layer visibility
   */
  setVisibility(layerId, visible) {
    const layer = this.layers.get(layerId);
    if (!layer) return false;

    layer.visible = visible;

    this.eventBus.emit(Foundation.Events.LAYER_TOGGLE, { layerId, visible });

    return true;
  }

  /**
   * Toggle layer visibility
   */
  toggle(layerId) {
    const layer = this.layers.get(layerId);
    if (!layer) return false;

    layer.visible = !layer.visible;
    this.eventBus.emit(Foundation.Events.LAYER_TOGGLE, {
      layerId,
      visible: layer.visible
    });

    return layer.visible;
  }

  /**
   * Set layer opacity
   */
  setOpacity(layerId, opacity) {
    const layer = this.layers.get(layerId);
    if (!layer) return false;

    layer.opacity = Math.max(0, Math.min(1, opacity));
    return true;
  }

  /**
   * Reorder layers
   */
  reorder(newOrder) {
    // Validate new order
    for (const id of newOrder) {
      if (!this.layers.has(id)) {
        console.error(`Layer "${id}" not found in reorder`);
        return false;
      }
    }

    this.order = newOrder;
    this.eventBus.emit(Foundation.Events.LAYER_REORDER, { order: newOrder });

    return true;
  }

  /**
   * Get layer by ID
   */
  getLayer(layerId) {
    return this.layers.get(layerId);
  }

  /**
   * Get all layers
   */
  getLayers() {
    return this.order.map(id => this.layers.get(id));
  }

  /**
   * Get interactive layers
   */
  getInteractiveLayers() {
    return this.order
      .map(id => this.layers.get(id))
      .filter(layer => layer.interactive && layer.visible);
  }

  /**
   * Render all visible layers
   */
  render(ctx, viewport) {
    for (const layerId of this.order) {
      const layer = this.layers.get(layerId);
      if (!layer.visible) continue;

      ctx.globalAlpha = layer.opacity;

      try {
        if (layer.renderer) {
          // Custom renderer function
          layer.renderer(ctx, layer, viewport);
        } else if (layer.type === 'vector' && layer.source) {
          // Default vector layer rendering
          this.renderVectorLayer(ctx, layer, viewport);
        }
      } catch (err) {
        console.error(`Layer "${layerId}" render error:`, err);
      }

      ctx.globalAlpha = 1;
    }
  }

  /**
   * Default vector layer rendering
   */
  renderVectorLayer(ctx, layer, viewport) {
    const { source, style } = layer;
    if (!source || !source.length) return;

    const baseScale = (viewport.height / 180) * viewport.scale;
    const { width, height, offsetX, offsetY } = viewport;

    // Calculate visible bounds
    const leftLon = (-width / 2 - offsetX) / baseScale;
    const rightLon = (width / 2 - offsetX) / baseScale;

    // Handle world wrapping (360-degree segments)
    const firstSegment = Math.floor(leftLon / 360);
    const lastSegment = Math.ceil(rightLon / 360);

    for (const feature of source) {
      if (!this.isFeatureVisible(feature, viewport)) continue;

      // Render feature in each visible segment
      for (let seg = firstSegment; seg <= lastSegment; seg++) {
        const lonOffset = seg * 360;
        this.renderFeature(ctx, feature, viewport, lonOffset, style);
      }
    }
  }

  /**
   * Check if feature is visible in viewport
   */
  isFeatureVisible(feature, viewport) {
    if (!feature.bbox) return true;

    const baseScale = (viewport.height / 180) * viewport.scale;
    const { width, height, offsetX, offsetY } = viewport;
    const [minLon, minLat, maxLon, maxLat] = feature.bbox;

    const leftLon = (-width / 2 - offsetX) / baseScale;
    const rightLon = (width / 2 - offsetX) / baseScale;
    const firstSegment = Math.floor((leftLon - maxLon) / 360);
    const lastSegment = Math.ceil((rightLon - minLon) / 360);

    for (let seg = firstSegment; seg <= lastSegment; seg++) {
      const offset = seg * 360;
      const x0 = width / 2 + (minLon + offset) * baseScale + offsetX;
      const x1 = width / 2 + (maxLon + offset) * baseScale + offsetX;
      const y0 = height / 2 - minLat * baseScale + offsetY;
      const y1 = height / 2 - maxLat * baseScale + offsetY;

      if (Math.max(x0, x1) >= -40 && Math.min(x0, x1) <= width + 40 &&
          Math.max(y0, y1) >= -40 && Math.min(y0, y1) <= height + 40) {
        return true;
      }
    }

    return false;
  }

  /**
   * Render a single feature
   */
  renderFeature(ctx, feature, viewport, lonOffset, style) {
    const rings = feature.rings || feature.geometry?.rings || [];
    if (!rings.length) return;

    const baseScale = (viewport.height / 180) * viewport.scale;
    const { width, height, offsetX, offsetY } = viewport;

    const fillColor = style.fillColor || style.fill || 'rgba(100, 100, 100, 0.3)';
    const strokeColor = style.strokeColor || style.stroke || 'rgba(50, 50, 50, 0.5)';
    const strokeWidth = style.strokeWidth || 0.5;

    ctx.fillStyle = fillColor;
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = strokeWidth;

    for (const ring of rings) {
      if (ring.length < 3) continue;

      const path = new Path2D();
      let first = true;

      for (const [lon, lat] of ring) {
        const x = width / 2 + (lon + lonOffset) * baseScale + offsetX;
        const y = height / 2 - lat * baseScale + offsetY;

        if (first) {
          path.moveTo(x, y);
          first = false;
        } else {
          path.lineTo(x, y);
        }
      }

      ctx.fill(path);
      ctx.stroke(path);
    }
  }

  /**
   * Hit test at geographic position
   */
  hitTest(lon, lat, viewport) {
    const interactiveLayers = this.getInteractiveLayers();

    // Test from top layer to bottom (reverse order)
    for (let i = interactiveLayers.length - 1; i >= 0; i--) {
      const layer = interactiveLayers[i];

      let hit = null;

      if (layer.hitTest) {
        // Custom hit test function
        hit = layer.hitTest(lon, lat, viewport, layer);
      } else if (layer.type === 'vector') {
        // Default vector hit test
        hit = this.hitTestVectorLayer(lon, lat, viewport, layer);
      }

      if (hit) {
        return { feature: hit, layer };
      }
    }

    return null;
  }

  /**
   * Default vector layer hit test
   */
  hitTestVectorLayer(lon, lat, viewport, layer) {
    const { source } = layer;
    if (!source) return null;

    // Normalize lon for world wrapping
    const normalizedLon = Foundation.CoordinateSystem.normalizeLon(lon);
    const segment = Math.round(lon / 360);

    let bestHit = null;
    let bestPriority = Infinity;

    for (const feature of source) {
      if (!feature.bbox) continue;

      // Check bbox with segment offset
      const shiftedMin = feature.bbox[0] + segment * 360;
      const shiftedMax = feature.bbox[2] + segment * 360;

      if (lon >= shiftedMin && lon <= shiftedMax &&
          lat >= feature.bbox[1] && lat <= feature.bbox[3]) {

        // Test actual geometry
        const testLon = lon - segment * 360;
        if (this.pointInFeature(testLon, lat, feature)) {
          // Prefer smaller features (higher priority)
          const priority = feature.areaKm2 || feature.priority || Infinity;
          if (priority < bestPriority) {
            bestHit = feature;
            bestPriority = priority;
          }
        }
      }
    }

    return bestHit;
  }

  /**
   * Point in feature test (ray casting)
   */
  pointInFeature(lon, lat, feature) {
    const rings = feature.rings || feature.geometry?.rings || [];
    let inside = false;

    for (const ring of rings) {
      if (this.pointInRing(lon, lat, ring)) {
        inside = !inside;
      }
    }

    return inside;
  }

  /**
   * Point in ring test (ray casting algorithm)
   */
  pointInRing(lon, lat, ring) {
    let inside = false;

    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [xi, yi] = ring[i];
      const [xj, yj] = ring[j];

      const intersects = ((yi > lat) !== (yj > lat)) &&
        lon < ((xj - xi) * (lat - yi)) / (yj - yi + Number.EPSILON) + xi;

      if (intersects) inside = !inside;
    }

    return inside;
  }
};