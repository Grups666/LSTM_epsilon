/**
 * Foundation Event Bus
 * Generic event system for module communication
 */
window.Foundation = window.Foundation || {};

Foundation.EventBus = class EventBus {
  constructor() {
    this.listeners = new Map();
    this.onceHandlers = new Map();
  }

  /**
   * Register an event handler
   * @param {string} event - Event name
   * @param {function} handler - Handler function
   * @param {object} context - Optional context for handler
   */
  on(event, handler, context = null) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push({ handler, context });
    return this;
  }

  /**
   * Register a one-time event handler
   * @param {string} event - Event name
   * @param {function} handler - Handler function
   * @param {object} context - Optional context
   */
  once(event, handler, context = null) {
    if (!this.onceHandlers.has(event)) {
      this.onceHandlers.set(event, []);
    }
    this.onceHandlers.get(event).push({ handler, context });
    return this;
  }

  /**
   * Remove an event handler
   * @param {string} event - Event name
   * @param {function} handler - Handler to remove
   */
  off(event, handler) {
    if (this.listeners.has(event)) {
      const handlers = this.listeners.get(event);
      const idx = handlers.findIndex(h => h.handler === handler);
      if (idx >= 0) handlers.splice(idx, 1);
    }
    if (this.onceHandlers.has(event)) {
      const handlers = this.onceHandlers.get(event);
      const idx = handlers.findIndex(h => h.handler === handler);
      if (idx >= 0) handlers.splice(idx, 1);
    }
    return this;
  }

  /**
   * Emit an event
   * @param {string} event - Event name
   * @param {object} payload - Event payload
   */
  emit(event, payload = {}) {
    // Regular handlers
    if (this.listeners.has(event)) {
      for (const { handler, context } of this.listeners.get(event)) {
        try {
          handler.call(context, payload);
        } catch (err) {
          console.error(`EventBus handler error for "${event}":`, err);
        }
      }
    }

    // One-time handlers
    if (this.onceHandlers.has(event)) {
      const handlers = this.onceHandlers.get(event);
      this.onceHandlers.delete(event);
      for (const { handler, context } of handlers) {
        try {
          handler.call(context, payload);
        } catch (err) {
          console.error(`EventBus once handler error for "${event}":`, err);
        }
      }
    }

    return this;
  }

  /**
   * Remove all handlers for an event
   * @param {string} event - Event name
   */
  clear(event) {
    this.listeners.delete(event);
    this.onceHandlers.delete(event);
    return this;
  }

  /**
   * Get handler count for an event
   * @param {string} event - Event name
   */
  handlerCount(event) {
    const regular = this.listeners.has(event) ? this.listeners.get(event).length : 0;
    const once = this.onceHandlers.has(event) ? this.onceHandlers.get(event).length : 0;
    return regular + once;
  }
};

// Standard Foundation Events
Foundation.Events = {
  // Feature interactions
  FEATURE_CLICK: 'feature:click',
  FEATURE_HOVER: 'feature:hover',

  // Layer operations
  LAYER_ADD: 'layer:add',
  LAYER_REMOVE: 'layer:remove',
  LAYER_TOGGLE: 'layer:toggle',
  LAYER_REORDER: 'layer:reorder',

  // Dataset operations
  DATASET_LOAD: 'dataset:load',
  DATASET_ERROR: 'dataset:error',

  // Module lifecycle
  MODULE_LOAD: 'module:load',
  MODULE_ACTIVATE: 'module:activate',
  MODULE_DEACTIVATE: 'module:deactivate',
  MODULE_READY: 'module:ready',

  // Panel operations
  PANEL_SHOW: 'panel:show',
  PANEL_HIDE: 'panel:hide',
  INSPECTOR_SHOW: 'inspector:show',
  INSPECTOR_HIDE: 'inspector:hide',

  // Project operations
  PROJECT_SAVE: 'project:save',
  PROJECT_LOAD: 'project:load',

  // Filter/timeline
  FILTER_CHANGE: 'filter:change',
  TIMELINE_CHANGE: 'timeline:change',

  // Map viewport
  VIEWPORT_CHANGE: 'viewport:change'
};

// Create global event bus instance
Foundation.eventBus = new Foundation.EventBus();