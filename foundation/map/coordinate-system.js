/**
 * Foundation Coordinate System Utilities
 * Generic coordinate transformations and spatial utilities
 */
window.Foundation = window.Foundation || {};

Foundation.CoordinateSystem = class CoordinateSystem {
  constructor(config = {}) {
    this.projection = config.projection || 'equirectangular';
  }

  /**
   * Normalize longitude to [-180, 180]
   */
  static normalizeLon(lon) {
    while (lon > 180) lon -= 360;
    while (lon < -180) lon += 360;
    return lon;
  }

  /**
   * Calculate bounding box center
   */
  static bboxCenter(bbox) {
    return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2];
  }

  /**
   * Check if point is in bounding box
   */
  static pointInBbox(lon, lat, bbox) {
    return lon >= bbox[0] && lon <= bbox[2] &&
           lat >= bbox[1] && lat <= bbox[3];
  }

  /**
   * Convert geographic coordinates to screen coordinates
   */
  geoToScreen(lon, lat, viewport) {
    const { width, height, scale, offsetX, offsetY } = viewport;
    const baseScale = (height / 180) * scale;
    const x = width / 2 + lon * baseScale + offsetX;
    const y = height / 2 - lat * baseScale + offsetY;
    return { x, y };
  }

  /**
   * Convert screen coordinates to geographic coordinates
   */
  screenToGeo(x, y, viewport) {
    const { width, height, scale, offsetX, offsetY } = viewport;
    const baseScale = (height / 180) * scale;
    const lon = (x - width / 2 - offsetX) / baseScale;
    const lat = (height / 2 + offsetY - y) / baseScale;
    return { lon, lat };
  }

  /**
   * Calculate distance between two points (Haversine formula)
   */
  static haversineDistance(lon1, lat1, lon2, lat2) {
    const R = 6371; // Earth radius in km
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
              Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
              Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
  }

  /**
   * Get visible geographic bounds from viewport
   */
  getVisibleBounds(viewport) {
    const { width, height, scale, offsetX, offsetY } = viewport;
    const baseScale = (height / 180) * scale;

    const leftLon = (-width / 2 - offsetX) / baseScale;
    const rightLon = (width / 2 - offsetX) / baseScale;
    const topLat = (height / 2 + offsetY) / baseScale;
    const bottomLat = (-height / 2 + offsetY) / baseScale;

    return {
      west: leftLon,
      east: rightLon,
      north: topLat,
      south: bottomLat
    };
  }
};

// Utility functions
Foundation.geo = {
  normalizeLon: Foundation.CoordinateSystem.normalizeLon,
  bboxCenter: Foundation.CoordinateSystem.bboxCenter,
  pointInBbox: Foundation.CoordinateSystem.pointInBbox,
  haversineDistance: Foundation.CoordinateSystem.haversineDistance
};