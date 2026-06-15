/**
 * Foundation Module Loader
 * Handles module registration, loading, and lifecycle
 */
window.Foundation = window.Foundation || {};

Foundation.ModuleLoader = class ModuleLoader {
  constructor(eventBus) {
    this.eventBus = eventBus;
    this.modules = new Map();
    this.activeModule = null;
    this.loadingOrder = [];
  }

  /**
   * Register a module
   * @param {object} manifest - Module manifest (from module.json)
   * @param {class} moduleClass - Module class constructor
   */
  register(manifest, moduleClass) {
    if (this.modules.has(manifest.id)) {
      console.warn(`Module "${manifest.id}" already registered`);
      return this;
    }

    const module = {
      manifest,
      class: moduleClass,
      instance: null,
      status: 'registered' // registered, loading, loaded, active, error
    };

    this.modules.set(manifest.id, module);
    this.loadingOrder.push(manifest.id);

    this.eventBus.emit(Foundation.Events.MODULE_LOAD, {
      moduleId: manifest.id,
      status: 'registered'
    });

    return this;
  }

  /**
   * Load a module (instantiate and call onLoad)
   * @param {string} moduleId - Module ID
   * @param {object} foundation - Foundation reference for module access
   */
  async load(moduleId, foundation) {
    const module = this.modules.get(moduleId);
    if (!module) {
      console.error(`Module "${moduleId}" not registered`);
      return null;
    }

    if (module.status === 'loaded' || module.status === 'active') {
      return module.instance;
    }

    module.status = 'loading';
    this.eventBus.emit(Foundation.Events.MODULE_LOAD, {
      moduleId,
      status: 'loading'
    });

    try {
      // Instantiate module. The manifest is passed so a module can resolve its
      // own ontology, data, and entry-relative assets without Foundation knowing
      // domain-specific paths.
      module.instance = new module.class(foundation, module.manifest);

      // Call onLoad lifecycle hook
      if (module.instance.onLoad) {
        await module.instance.onLoad();
      }

      module.status = 'loaded';
      this.eventBus.emit(Foundation.Events.MODULE_LOAD, {
        moduleId,
        status: 'loaded'
      });

      return module.instance;
    } catch (err) {
      module.status = 'error';
      console.error(`Module "${moduleId}" load error:`, err);
      this.eventBus.emit(Foundation.Events.MODULE_LOAD, {
        moduleId,
        status: 'error',
        error: err
      });
      return null;
    }
  }

  /**
   * Load all registered modules
   * @param {object} foundation - Foundation reference
   */
  async loadAll(foundation) {
    const results = [];
    for (const moduleId of this.loadingOrder) {
      const instance = await this.load(moduleId, foundation);
      results.push({ moduleId, instance, success: !!instance });
    }
    return results;
  }

  /**
   * Activate a module
   * @param {string} moduleId - Module ID
   */
  activate(moduleId) {
    const module = this.modules.get(moduleId);
    if (!module || module.status !== 'loaded') {
      console.error(`Cannot activate module "${moduleId}" - not loaded`);
      return false;
    }

    // Deactivate current active module
    if (this.activeModule && this.activeModule !== moduleId) {
      this.deactivate(this.activeModule);
    }

    module.status = 'active';
    this.activeModule = moduleId;

    if (module.instance.onActivate) {
      module.instance.onActivate();
    }

    this.eventBus.emit(Foundation.Events.MODULE_ACTIVATE, { moduleId });
    return true;
  }

  /**
   * Deactivate a module
   * @param {string} moduleId - Module ID
   */
  deactivate(moduleId) {
    const module = this.modules.get(moduleId);
    if (!module || module.status !== 'active') {
      return false;
    }

    if (module.instance.onDeactivate) {
      module.instance.onDeactivate();
    }

    module.status = 'loaded';
    if (this.activeModule === moduleId) {
      this.activeModule = null;
    }

    this.eventBus.emit(Foundation.Events.MODULE_DEACTIVATE, { moduleId });
    return true;
  }

  /**
   * Get module instance
   * @param {string} moduleId - Module ID
   */
  get(moduleId) {
    const module = this.modules.get(moduleId);
    return module?.instance || null;
  }

  /**
   * Get module manifest
   * @param {string} moduleId - Module ID
   */
  getManifest(moduleId) {
    const module = this.modules.get(moduleId);
    return module?.manifest || null;
  }

  /**
   * Get all registered module IDs
   */
  getModuleIds() {
    return Array.from(this.modules.keys());
  }

  /**
   * Get active module ID
   */
  getActiveModuleId() {
    return this.activeModule;
  }

  /**
   * Check if module is loaded
   * @param {string} moduleId - Module ID
   */
  isLoaded(moduleId) {
    const module = this.modules.get(moduleId);
    return module && (module.status === 'loaded' || module.status === 'active');
  }
};

// Create global module loader instance
Foundation.moduleLoader = new Foundation.ModuleLoader(Foundation.eventBus);
