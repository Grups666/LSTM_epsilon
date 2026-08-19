/**
 * Epsilon Change Module
 *
 * Tereon module for catchment-scale epsilon distribution shifts.
 * Foundation supplies the map, module loader, layer manager, and inspector panel.
 */
window.EpsilonChangeModule = class EpsilonChangeModule {
  constructor(app, manifest = {}) {
    this.app = app;
    this.manifest = manifest;
    this.basePath = manifest.basePath || `/modules/${manifest.id || "epsilon-change"}/`;
    this.data = null;
    this.globalStory = null;
    this.rawBasins = [];
    this.basins = [];
    this.byId = new Map();
    this.selected = null;
    this.overviewNavScrollHandler = null;
    this.analysisView = ["change", "decomposition", "trend"].includes(manifest.analysisView)
      ? manifest.analysisView
      : "change";
    this.activeRegime = ["all", "low", "high"].includes(manifest.defaultRegime)
      ? manifest.defaultRegime
      : "all";
    this.dataFile = manifest.dataFile || manifest.datasets?.[0]?.file || "./data/epsilon-catchment-distributions.json";
    this.globalStoryFile = manifest.globalStoryFile || "./data/global-story-summary.json";
    this.layerId = `${manifest.id || "epsilon-change"}-catchments`;
    this.overviewLayerId = `${manifest.id || "epsilon-change"}-overview`;
    this.legendId = `${manifest.id || "epsilon-change"}-legend`;
    this.toolbar = null;
    this.overviewModal = null;
    this.distributionModal = null;
    this.activeDistribution = null;
    this.themeObserver = null;
    this.skillFilter = {
      metric: manifest.skillFilterMetric || "nse",
      threshold: Number.isFinite(Number(manifest.skillFilterThreshold)) ? Number(manifest.skillFilterThreshold) : 0.5
    };
    this.reliabilityEligibleCount = 0;
    this.insufficientExcludedCount = 0;
    this.displayRegimes = ["all", "low", "high"];
    this.handleModalPointer = (event) => this.onDistributionPointer(event);
    this.handleFeatureClick = (payload) => {
      if (payload.layer?.id !== this.layerId || payload.layer?.moduleId !== this.manifest.id) return;
      this.selected = payload.feature;
      this.showInspector(payload.feature);
      this.app.draw?.();
    };
    this.handleLayerToggle = (payload) => {
      if (payload.layerId !== this.overviewLayerId) return;
      if (payload.visible) this.showOverview();
      else this.closeOverview();
    };
    this.handleThemeChange = () => {
      if (this.selected) this.showInspector(this.selected);
      if (this.overviewModal?.classList.contains("visible")) this.showOverview();
      if (this.distributionModal?.classList.contains("visible")) this.drawDistributionModal();
    };
  }

  async onLoad() {
    [this.data, this.globalStory] = await Promise.all([
      this.fetchJson(this.resolve(this.dataFile)),
      this.fetchJson(this.resolve(this.globalStoryFile)).catch(() => null)
    ]);
    this.rawBasins = (this.data.basins || [])
      .filter((basin) => Number.isFinite(Number(basin.lon)) && Number.isFinite(Number(basin.lat)))
      .map((basin) => ({
        ...basin,
        id: String(basin.GCIN),
        lon: Number(basin.lon),
        lat: Number(basin.lat),
        area_km2: Number(basin.area_km2 || 0)
      }));
    this.applySkillFilter();
    this.colorScaleExtent = this.computeContinuousExtent();
    this.byId = new Map(this.basins.map((basin) => [basin.id, basin]));
    this.addLayer();
    this.ensurePreviewStyles();
    this.ensureToolbar();
    this.ensureLegend();
    this.showOverview();
    Foundation.eventBus.on(Foundation.Events.FEATURE_CLICK, this.handleFeatureClick);
    Foundation.eventBus.on(Foundation.Events.LAYER_TOGGLE, this.handleLayerToggle);
    this.themeObserver = new MutationObserver(this.handleThemeChange);
    this.themeObserver.observe(document.body, { attributes: true, attributeFilter: ["class"] });
    this.app.draw?.();
  }

  onUnload() {
    this.app.layerManager.removeLayer(this.layerId);
    this.app.layerManager.removeLayer(this.overviewLayerId);
    this.app.unregisterLegend?.(this.legendId);
    Foundation.eventBus.off(Foundation.Events.FEATURE_CLICK, this.handleFeatureClick);
    Foundation.eventBus.off(Foundation.Events.LAYER_TOGGLE, this.handleLayerToggle);
    this.themeObserver?.disconnect();
    this.themeObserver = null;
    this.selected = null;
    this.toolbar?.remove();
    this.toolbar = null;
    this.destroyModals();
  }

  getLayerIds() {
    return [this.layerId, this.overviewLayerId];
  }

  getTutorialBasin() {
    if (!this.basins.length) return null;
    const preferred = this.byId.get("3859");
    if (preferred) return preferred;
    const reference = { lon: -98, lat: 32 };
    return this.basins.reduce((closest, basin) => {
      const distance = Math.hypot(this.lonDistance(basin.lon, reference.lon), basin.lat - reference.lat);
      return !closest || distance < closest.distance ? { basin, distance } : closest;
    }, null)?.basin || this.basins[0];
  }

  focusTutorialBasin() {
    const basin = this.getTutorialBasin();
    if (!basin) return null;
    this.closeOverview();
    this.closeDistributionModal();
    this.selected = null;
    this.app.selectedFeature = null;
    this.app.selectedLayer = null;
    this.app.viewport.scale = Math.max(this.app.getMinViewportScale(), 4);
    const base = this.app.getBaseScale();
    this.app.viewport.offsetX = -basin.lon * base;
    this.app.viewport.offsetY = basin.lat * base;
    this.app.clampOffset();
    this.app.draw?.();
    return basin;
  }

  showTutorialBasin() {
    const basin = this.getTutorialBasin();
    if (!basin) return null;
    const layer = this.app.layerManager.getLayer?.(this.layerId);
    this.selected = basin;
    this.app.selectedFeature = basin;
    this.app.selectedLayer = layer || null;
    this.showInspector(basin);
    this.app.draw?.();
    return basin;
  }

  tutorialBasinRect() {
    const basin = this.getTutorialBasin();
    const canvasRect = this.app.canvas?.getBoundingClientRect();
    if (!basin || !canvasRect) return null;
    const base = this.app.getBaseScale();
    const x = this.app.viewport.width / 2 + basin.lon * base + this.app.viewport.offsetX;
    const y = this.app.viewport.height / 2 - basin.lat * base + this.app.viewport.offsetY;
    return {
      left: canvasRect.left + x - 18,
      top: canvasRect.top + y - 18,
      width: 36,
      height: 36,
      right: canvasRect.left + x + 18,
      bottom: canvasRect.top + y + 18,
    };
  }

  resolve(path) {
    if (/^https?:\/\//i.test(path) || path.startsWith("/")) return path;
    return this.basePath + path.replace(/^\.\//, "");
  }

  async fetchJson(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to load ${url}: ${response.status}`);
    return response.json();
  }

  applySkillFilter() {
    const metric = this.skillFilter.metric === "kge" ? "kge" : "nse";
    const threshold = Number(this.skillFilter.threshold);
    this.skillFilter.metric = metric;
    this.skillFilter.threshold = Number.isFinite(threshold) ? threshold : 0.5;
    const reliabilityEligible = this.rawBasins.filter((basin) => {
      const pre = Number(basin[`pre_${metric}`]);
      const post = Number(basin[`post_${metric}`]);
      return Number.isFinite(pre) && Number.isFinite(post) && pre > this.skillFilter.threshold && post > this.skillFilter.threshold;
    });
    this.reliabilityEligibleCount = reliabilityEligible.length;
    const supported = reliabilityEligible.filter((basin) => this.hasAnalysisData(basin));
    this.insufficientExcludedCount = reliabilityEligible.length - supported.length;
    this.basins = supported;
    this.byId = new Map(this.basins.map((basin) => [basin.id, basin]));
    if (this.selected && !this.byId.has(String(this.selected.id))) {
      this.selected = null;
      this.app.selectedFeature = null;
      this.app.selectedLayer = null;
      document.getElementById("inspectorPanel")?.classList.remove("visible");
    }
  }

  hasSufficientShiftData(basin) {
    const regime = this.activeRegime;
    return Boolean(this.shiftState(basin[`${regime}_epsilon_shift_class`]))
      && Number.isFinite(this.shiftValue(basin, regime));
  }

  hasAnalysisData(basin) {
    const regime = this.activeRegime;
    if (this.analysisView === "decomposition") {
      return ["gq", "q", "combined", "offsetting"].includes(basin[`${regime}_driver`])
        && Number.isFinite(Number(basin[`${regime}_gq_component_log`]))
        && Number.isFinite(Number(basin[`${regime}_q_component_log`]));
    }
    if (this.analysisView === "trend") {
      return Boolean(this.trendState(basin[`${regime}_epsilon_trend_class`]))
        && Number.isFinite(Number(basin[`${regime}_epsilon_slope_pct_decade`]));
    }
    return this.hasSufficientShiftData(basin);
  }

  updateSkillFilter(metric, threshold, refreshOverview = true) {
    this.skillFilter.metric = metric;
    this.skillFilter.threshold = threshold;
    this.applySkillFilter();
    this.colorScaleExtent = this.computeContinuousExtent();
    this.ensureLegend();
    if (refreshOverview && this.overviewModal?.classList.contains("visible")) this.showOverview();
    this.app.draw?.();
  }

  setRegime(regime) {
    if (!["all", "low", "high"].includes(regime)) return;
    this.activeRegime = regime;
    this.applySkillFilter();
    this.colorScaleExtent = this.computeContinuousExtent();
    this.toolbar?.querySelectorAll("[data-regime]").forEach((button) => {
      button.classList.toggle("active", button.dataset.regime === regime);
    });
    this.ensureLegend();
    if (this.selected && this.byId.has(String(this.selected.id))) this.showInspector(this.selected);
    if (this.overviewModal?.classList.contains("visible")) this.showOverview();
    this.app.draw?.();
  }

  ensureToolbar() {
    if (this.toolbar) return;
    this.toolbar = document.createElement("div");
    this.toolbar.className = "epsilon-toolbar";
    this.toolbar.innerHTML = `
      <div class="epsilon-toolbar-label">${this.escape(this.analysisTitle())}</div>
      <div class="epsilon-toolbar-segments" role="group" aria-label="Flow condition">
        ${[
          ["all", "All recession"],
          ["low", "Low flow"],
          ["high", "High flow"],
        ].map(([regime, label]) => `<button type="button" data-regime="${regime}" class="${this.activeRegime === regime ? "active" : ""}">${label}</button>`).join("")}
      </div>
      <button type="button" class="epsilon-toolbar-overview">Research overview</button>
    `;
    this.toolbar.querySelectorAll("[data-regime]").forEach((button) => {
      button.addEventListener("click", () => this.setRegime(button.dataset.regime));
    });
    this.toolbar.querySelector(".epsilon-toolbar-overview").addEventListener("click", () => {
      this.app.layerManager.setVisibility(this.overviewLayerId, true);
      this.app.updateLayerList?.();
      this.showOverview();
    });
    document.body.appendChild(this.toolbar);
  }

  skillFilterLabel() {
    return `${this.skillFilter.metric.toUpperCase()} > ${this.formatNumber(this.skillFilter.threshold, 2)}`;
  }

  addLayer() {
    this.app.layerManager.addLayer({
      id: this.layerId,
      name: this.layerName(),
      type: "vector",
      visible: true,
      interactive: true,
      moduleId: this.manifest.id,
      groupPath: ["epsilon"],
      metadata: {
        removable: false,
        periods: this.data?.meta?.periods,
        regimes: this.data?.meta?.regimes
      },
      renderer: (ctx, _layer, viewport) => this.render(ctx, viewport),
      hitTest: (lon, lat, viewport) => this.hitTest(lon, lat, viewport)
    });
    this.app.layerManager.addLayer({
      id: this.overviewLayerId,
      name: "Overview",
      type: "overlay",
      visible: false,
      interactive: false,
      moduleId: this.manifest.id,
      metadata: { removable: false },
      renderer: () => {}
    });
    this.app.updateLayerList?.();
  }

  render(ctx, viewport) {
    const base = (viewport.height / 180) * viewport.scale;
    const { width, height, offsetX, offsetY } = viewport;
    const leftLon = (-width / 2 - offsetX) / base;
    const rightLon = (width / 2 - offsetX) / base;
    const firstSeg = Math.floor(leftLon / 360);
    const lastSeg = Math.ceil(rightLon / 360);
    for (let seg = firstSeg; seg <= lastSeg; seg++) {
      const lonOffset = seg * 360;
      for (const basin of this.basins) {
        const x = width / 2 + (basin.lon + lonOffset) * base + offsetX;
        const y = height / 2 - basin.lat * base + offsetY;
        if (x < -20 || x > width + 20 || y < -20 || y > height + 20) continue;

        const selected = this.selected?.id === basin.id;
        const hovered = this.app.hoveredLayer?.id === this.layerId && this.app.hoveredFeatureId === basin.id;
        const radius = selected ? 6.5 : hovered ? this.pointRadius(basin, viewport) + 2.2 : this.pointRadius(basin, viewport);
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = this.basinColor(basin);
        ctx.globalAlpha = selected ? 0.98 : 0.72;
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.lineWidth = selected ? 2.2 : hovered ? 2.0 : 0.7;
        ctx.strokeStyle = selected ? "#0f172a" : hovered ? "#1d4ed8" : "rgba(15,23,42,0.24)";
        ctx.stroke();
      }
    }
  }

  pointRadius(basin, viewport) {
    const area = Math.max(Number(basin.area_km2 || 1), 1);
    const baseRadius = 1.8 + Math.log10(area) * 0.58;
    return Math.max(2, Math.min(5.6, baseRadius * Math.sqrt(Math.max(viewport.scale, 0.6))));
  }

  hitTest(lon, lat, viewport) {
    const normalizedLon = ((lon + 180) % 360 + 360) % 360 - 180;
    const threshold = Math.max(0.12, 7 / ((viewport.height / 180) * viewport.scale));
    let best = null;
    let bestDistance = Infinity;

    for (const basin of this.basins) {
      const dx = this.lonDistance(normalizedLon, basin.lon);
      const dy = lat - basin.lat;
      const distance = Math.hypot(dx, dy);
      if (distance < threshold && distance < bestDistance) {
        best = basin;
        bestDistance = distance;
      }
    }
    return best;
  }

  lonDistance(a, b) {
    let diff = a - b;
    while (diff > 180) diff -= 360;
    while (diff < -180) diff += 360;
    return diff;
  }

  showOverview() {
    const snapshotMetrics = this.renderSnapshotMetrics();
    const layer = this.app.layerManager.getLayer?.(this.overviewLayerId);
    if (layer && !layer.visible) return;
    this.ensureOverviewModal();
    this.overviewModal.querySelector(".epsilon-overview-body").innerHTML = `
      <div class="epsilon-overview-layout">
        ${this.renderOverviewNavigation()}
        <div class="epsilon-overview-content">
          <section id="epsilon-overview-snapshot">
            <h3>Current view</h3>
            <p class="epsilon-overview-lead">${this.escape(this.overviewText())}</p>
            ${this.renderOverviewFilter()}
            <div class="epsilon-overview-metrics">
              ${snapshotMetrics}
            </div>
          </section>
          <section id="epsilon-overview-map-key">
            <h3>Read the map</h3>
            <div class="epsilon-overview-classification">
               ${this.renderOverviewLegend()}
            </div>
            ${this.renderLegendDefinitions()}
            ${this.analysisView === "decomposition" ? `
              <div class="epsilon-overview-attribution">
                ${this.renderAttributionDefinitions()}
              </div>
            ` : ""}
          </section>
          ${this.analysisView === "change" ? this.renderGlobalEvidence() : ""}
          <section id="epsilon-overview-data-model">
            <h3>Data, model and equations</h3>
            ${this.renderMethodEssentials()}
          </section>
          ${this.analysisView === "change" ? this.renderMethodStory() : this.renderFocusedMethod()}
        </div>
      </div>
    `;
    this.bindOverviewFilter();
    this.bindOverviewNavigation();
    this.overviewModal.classList.add("visible");
  }

  renderSnapshotMetrics() {
    const regime = this.activeRegime;
    if (this.analysisView === "decomposition") {
      const count = (driver) => this.basins.filter((basin) => basin[`${regime}_driver`] === driver).length;
      return `
        ${this.metricCard("Catchments", this.basins.length.toLocaleString())}
        ${this.metricCard("GQ-dominant", count("gq").toLocaleString())}
        ${this.metricCard("Q-dominant", count("q").toLocaleString())}
        ${this.metricCard("Combined / offsetting", `${count("combined").toLocaleString()} / ${count("offsetting").toLocaleString()}`)}
      `;
    }
    if (this.analysisView === "trend") {
      const values = this.basins.map((basin) => Number(basin[`${regime}_epsilon_slope_pct_decade`])).filter(Number.isFinite);
      const stateCount = (state) => this.basins.filter((basin) => this.trendState(basin[`${regime}_epsilon_trend_class`]) === state).length;
      return `
        ${this.metricCard("Catchments", this.basins.length.toLocaleString())}
        ${this.metricCard("Median slope", `${this.formatPct(this.median(values))} / decade`, this.median(values))}
        ${this.metricCard("FDR increase", stateCount("increase").toLocaleString())}
        ${this.metricCard("FDR decrease", stateCount("decrease").toLocaleString())}
      `;
    }
    const values = this.basins.map((basin) => this.shiftValue(basin, regime)).filter(Number.isFinite);
    const stateCount = (state) => this.basins.filter((basin) => this.shiftState(basin[`${regime}_epsilon_shift_class`]) === state).length;
    return `
      ${this.metricCard("Catchments", this.basins.length.toLocaleString())}
      ${this.metricCard("Median effect", this.formatPct(this.median(values)), this.median(values))}
      ${this.metricCard("FDR increase", stateCount("increase").toLocaleString())}
      ${this.metricCard("FDR decrease", stateCount("decrease").toLocaleString())}
    `;
  }

  renderOverviewNavigation() {
    const items = [
      ["epsilon-overview-snapshot", "Snapshot"],
      ["epsilon-overview-map-key", "Map key"],
      ...(this.analysisView === "change" ? [
        ["epsilon-overview-global-evidence", "Global evidence"],
        ["epsilon-overview-data-model", "Data & model"],
        ["epsilon-overview-workflow", "Workflow"],
        ["epsilon-overview-crossfit", "Cross-fit"],
        ["epsilon-overview-era-shift", "Era shift"],
        ["epsilon-overview-attribution-step", "Decomposition"],
        ["epsilon-overview-trends", "Sensitivity"]
      ] : [
        ["epsilon-overview-data-model", "Data & model"],
        ["epsilon-overview-focused-method", "Interpretation"]
      ])
    ];
    return `
      <nav class="epsilon-overview-nav" aria-label="Overview sections">
        <div class="epsilon-overview-nav-title">On this page</div>
        ${items.map(([id, label], index) => `<a href="#${id}"${index === 0 ? ' class="active"' : ""}>${label}</a>`).join("")}
      </nav>
    `;
  }

  bindOverviewNavigation() {
    const body = this.overviewModal?.querySelector(".epsilon-overview-body");
    const nav = body?.querySelector(".epsilon-overview-nav");
    if (!body || !nav) return;
    if (this.overviewNavScrollHandler) body.removeEventListener("scroll", this.overviewNavScrollHandler);
    const links = [...nav.querySelectorAll("a")];
    const targets = links.map((link) => body.querySelector(link.getAttribute("href"))).filter(Boolean);
    const activate = (id) => links.forEach((link) => link.classList.toggle("active", link.getAttribute("href") === `#${id}`));
    links.forEach((link) => {
      link.onclick = (event) => {
        event.preventDefault();
        const target = body.querySelector(link.getAttribute("href"));
        if (!target) return;
        activate(target.id);
        const targetTop = target.getBoundingClientRect().top - body.getBoundingClientRect().top + body.scrollTop;
        const compact = window.innerWidth <= 760;
        const offset = compact ? 68 : 12;
        body.scrollTo({ top: Math.max(0, targetTop - offset), behavior: compact ? "auto" : "smooth" });
      };
    });
    this.overviewNavScrollHandler = () => {
      const top = body.getBoundingClientRect().top + (window.innerWidth <= 760 ? 96 : 34);
      let current = targets[0];
      for (const target of targets) {
        if (target.getBoundingClientRect().top <= top) current = target;
      }
      if (current) activate(current.id);
    };
    body.addEventListener("scroll", this.overviewNavScrollHandler, { passive: true });
  }

  renderMethodEssentials() {
    return `
      <div class="epsilon-method-facts">
        <div><strong>Daily data</strong><span>GCIN observed Q joined by catchment and date to ERA5-Land precipitation, temperature, PET, soil moisture and AET-related inputs.</span></div>
        <div><strong>Periods & regimes</strong><span>Pre 1950-1990; post 1991-2019. Low flow uses Qobs &le; catchment Q10; high flow uses Qobs &ge; catchment Q90.</span></div>
        <div><strong>Physics core</strong><span>Daily epsilon is inferred directly by the reference Ara LSTM-epsilon core. The governing equation, state reset and four-part loss are unchanged.</span></div>
        <div><strong>Evaluation</strong><span>Five paired temporal folds provide out-of-time estimates. NSE and KGE assess reconstructed Q and are indirect reliability checks for latent epsilon.</span></div>
      </div>
      <div class="epsilon-equation-grid">
        <div class="epsilon-equation-card"><span>Recession equation</span><code>dQ/dt = -epsilon * Q^2 - epsilon * alpha * AET * Q</code></div>
        <div class="epsilon-equation-card"><span>Component identity</span><code>GQ(t) = epsilon(t) * Qsim(t)</code><code>delta log epsilon = delta log GQ - delta log Qsim</code></div>
        <div class="epsilon-equation-card"><span>Reliability & inference rules</span><code>skill_pre &gt; threshold AND skill_post &gt; threshold</code><code>Increase / Decrease only when era-shift FDR q &lt; 0.05</code></div>
      </div>
      <p class="epsilon-method-caution"><strong>Interpretation boundary.</strong> The primary result is an association between model-inferred epsilon and the two climate eras. The GQ / Q decomposition is a descriptive identity, and neither result alone identifies an external climate cause.</p>
    `;
  }

  renderGlobalEvidence() {
    const story = this.globalStory;
    if (!story?.fieldEvidence) return "";
    const spread = story.fieldEvidence.distributionSpread;
    const median = story.fieldEvidence.distributionMedian;
    const soil = story.hydroclimateAssociation?.soilMoistureAllRecession;
    const spreadConfirmation = spread.confirmation;
    const medianConfirmation = median.confirmation;
    const soilJoint = soil?.jointPrecipitationSoilMoistureBlockFixed;
    const spreadPositive = 100 * Number(spreadConfirmation.positiveCatchmentFraction);
    const coverage = story.coverage;
    const coveragePct = 100 * Number(coverage?.fieldCoverageFraction);
    return `
      <section id="epsilon-overview-global-evidence">
        <h3>Global field evidence</h3>
        <p class="epsilon-overview-lead">A catchment can remain Unresolved after local FDR control while a spatially replicated field-level pattern is still supported. These tests pool catchment effects without relabeling any individual map point.${coverage ? ` At the fixed NSE &gt; 0.5 protocol, ${Number(coverage.fieldEligibleCatchments).toLocaleString()} of ${Number(coverage.reliabilityQualifiedCatchments).toLocaleString()} reliability-qualified catchments contribute to the all-recession field test (${this.formatNumber(coveragePct, 1)}%).` : ""}</p>
        <div class="epsilon-evidence-grid">
          <article class="epsilon-evidence-item epsilon-evidence-item--primary">
            <span class="epsilon-evidence-kicker">Secondary distribution-shape result</span>
            <div class="epsilon-evidence-value">${this.formatSignedPct(spreadConfirmation.estimatePct)}</div>
            <strong>Wider annual epsilon distribution after 1990</strong>
            <p>Confirmation 95% spatial-block CI ${this.formatSignedPct(spreadConfirmation.ciLowPct)} to ${this.formatSignedPct(spreadConfirmation.ciHighPct)}; Holm p ${this.formatPValue(spreadConfirmation.holmPValue)}. The direction was positive in ${this.formatNumber(spreadPositive, 1)}% of confirmation catchments.</p>
            <small>Independent discovery: ${this.formatSignedPct(spread.discovery.estimatePct)}. Full sample, descriptive: ${this.formatSignedPct(spread.fullDescriptive.estimatePct)}.</small>
          </article>
          <article class="epsilon-evidence-item">
            <span class="epsilon-evidence-kicker">Secondary location shift</span>
            <div class="epsilon-evidence-value">${this.formatSignedPct(medianConfirmation.estimatePct)}</div>
            <strong>Higher annual epsilon median in confirmation blocks</strong>
            <p>Confirmation 95% spatial-block CI ${this.formatSignedPct(medianConfirmation.ciLowPct)} to ${this.formatSignedPct(medianConfirmation.ciHighPct)}; Holm p ${this.formatPValue(medianConfirmation.holmPValue)}.</p>
            <small>Discovery was weaker (${this.formatSignedPct(median.discovery.estimatePct)}; interval crossed zero). This field result complements, but does not replace, the catchment-level era-effect map.</small>
          </article>
        </div>
        ${soilJoint ? `
          <div class="epsilon-evidence-association">
            <div>
              <span class="epsilon-evidence-kicker">Hydroclimate association</span>
              <strong>Wetter soil-moisture change aligns with a smaller annual-median epsilon shift</strong>
            </div>
            <div class="epsilon-evidence-association-value">${this.formatSignedPct(soilJoint.estimatePctPerDiscoverySd)}</div>
            <p>This model uses the all-recession annual-median effect, not distribution spread. Estimate is per discovery-sample SD after joint precipitation adjustment and 10-degree spatial-block fixed effects; 95% block-bootstrap CI ${this.formatSignedPct(soilJoint.ciLowPctPerDiscoverySd)} to ${this.formatSignedPct(soilJoint.ciHighPctPerDiscoverySd)}. This is an association, not a causal climate attribution.</p>
          </div>
        ` : ""}
        <div class="epsilon-evidence-guardrail"><strong>What did not replicate.</strong> Low-versus-high flow direction contrasts were not stable across the spatial split, and precipitation did not retain independent interval evidence after soil-moisture adjustment.</div>
        <p class="epsilon-evidence-method">Design: deterministic 10-degree spatial blocks, 40% discovery / 60% untouched confirmation, random-effects aggregation, spatial block bootstrap, and Holm family-wise correction. Sensitivity checks cover 1985/1990/1995 breaks, 3/5/10 annual days, NSE/KGE cohorts, and 5/10/20-degree blocks.</p>
      </section>
    `;
  }

  renderOverviewFilter() {
    return `
      <div class="epsilon-overview-filter">
        <div class="epsilon-filter-title">Reliability filter</div>
        <div class="epsilon-filter-grid">
          <label class="epsilon-filter-field">
            <span>Metric</span>
            <select class="epsilon-filter-metric" aria-label="Reliability metric">
              <option value="nse"${this.skillFilter.metric === "nse" ? " selected" : ""}>NSE</option>
              <option value="kge"${this.skillFilter.metric === "kge" ? " selected" : ""}>KGE</option>
            </select>
          </label>
          <label class="epsilon-filter-field">
            <span>Minimum</span>
            <input class="epsilon-filter-number" type="number" step="0.05" value="${this.formatNumber(this.skillFilter.threshold, 2)}" aria-label="Minimum reliability threshold">
          </label>
          <label class="epsilon-filter-field epsilon-filter-slider">
            <span>Threshold</span>
            <input class="epsilon-filter-range" type="range" min="-1" max="1" step="0.05" value="${this.formatNumber(this.skillFilter.threshold, 2)}" aria-label="Minimum reliability threshold slider">
          </label>
        </div>
        <div class="epsilon-field-mode-note"><strong>Scientific view is fixed.</strong> The top selector changes the flow condition; this control only filters catchments by reconstructed-streamflow skill. It never changes the estimand or significance rule.</div>
        <div class="epsilon-filter-count">${this.filterCountText()}</div>
      </div>
    `;
  }

  filterCountText() {
    const parts = [
      `${this.basins.length.toLocaleString()} catchments shown`,
      `${this.reliabilityEligibleCount.toLocaleString()} pass reliability`,
      `${this.insufficientExcludedCount.toLocaleString()} insufficient-support excluded`
    ];
    return parts.join(" | ");
  }

  bindOverviewFilter() {
    const root = this.overviewModal?.querySelector(".epsilon-overview-filter");
    if (!root) return;
    const metric = root.querySelector(".epsilon-filter-metric");
    const number = root.querySelector(".epsilon-filter-number");
    const range = root.querySelector(".epsilon-filter-range");
    const apply = (value, refreshOverview = true) => {
      const threshold = Number(value);
      const next = Number.isFinite(threshold) ? threshold : 0.5;
      number.value = next.toFixed(2);
      range.value = next.toFixed(2);
      this.updateSkillFilter(metric.value, next, refreshOverview);
      if (!refreshOverview) {
        const count = root.querySelector(".epsilon-filter-count");
        if (count) count.textContent = this.filterCountText();
      }
    };
    metric.onchange = () => this.updateSkillFilter(metric.value, Number(number.value));
    number.onchange = () => apply(number.value);
    range.oninput = () => apply(range.value, false);
    range.onchange = () => apply(range.value, true);
  }

  renderFocusedMethod() {
    if (this.analysisView === "decomposition") {
      return `
        <section id="epsilon-overview-focused-method">
          <h3>Interpretation boundary</h3>
          ${this.renderAttributionDefinitions()}
          <p class="epsilon-method-caution"><strong>Descriptive decomposition.</strong> GQ is computed from the same out-of-fold daily estimates as epsilon and Q. The identity closes algebraically, but dominance is not a causal climate attribution and has no separate significance claim.</p>
        </section>
      `;
    }
    return `
      <section id="epsilon-overview-focused-method">
        <h3>Interpretation boundary</h3>
        <div class="epsilon-overview-definitions">
          <div class="epsilon-overview-definition"><span class="epsilon-overview-definition-title">Slope</span><span>Fold-centered annual epsilon medians are summarized by a Theil-Sen slope in percent per decade.</span></div>
          <div class="epsilon-overview-definition"><span class="epsilon-overview-definition-title">Evidence</span><span>Kendall tests are corrected across catchments with Benjamini-Hochberg FDR. A non-significant slope is unresolved evidence, not proof of no change.</span></div>
          <div class="epsilon-overview-definition"><span class="epsilon-overview-definition-title">Role</span><span>This is a robustness check for monotonic evolution. The pre/post 1990 era coefficient remains the primary estimate.</span></div>
        </div>
      </section>
    `;
  }

  closeOverview() {
    this.overviewModal?.classList.remove("visible");
  }

  destroyModals() {
    this.activeDistribution = null;
    const overviewBody = this.overviewModal?.querySelector(".epsilon-overview-body");
    if (overviewBody && this.overviewNavScrollHandler) {
      overviewBody.removeEventListener("scroll", this.overviewNavScrollHandler);
    }
    this.overviewNavScrollHandler = null;
    this.overviewModal?.remove();
    this.distributionModal?.remove();
    this.overviewModal = null;
    this.distributionModal = null;
  }

  ensureOverviewModal() {
    if (this.overviewModal) return;
    this.overviewModal = document.createElement("div");
    this.overviewModal.className = "epsilon-overview-modal";
    this.overviewModal.innerHTML = `
      <div class="epsilon-overview-dialog" role="dialog" aria-label="Overview">
        <div class="epsilon-overview-header">
          <div>
            <div class="epsilon-overview-title">Overview</div>
          </div>
          <button class="epsilon-overview-close" type="button" aria-label="Close"></button>
        </div>
        <div class="epsilon-overview-body"></div>
      </div>
    `;
    this.overviewModal.querySelector(".epsilon-overview-close").onclick = () => {
      this.app.layerManager.setVisibility(this.overviewLayerId, false);
      this.closeOverview();
      this.app.updateLayerList?.();
    };
    document.body.appendChild(this.overviewModal);
  }

  renderMethodStory() {
    return `
      <section class="epsilon-story" id="epsilon-overview-workflow">
        <h3>Method workflow</h3>
        <p class="epsilon-story-lead">
          The analysis starts from daily catchment records, keeps only recession periods where epsilon is physically interpretable,
          infers epsilon inside a physics-informed LSTM, and then compares the inferred epsilon distributions before and after 1990.
        </p>
        ${this.renderStoryPanel({
          index: "01",
          title: "Build a catchment-day table rather than a raster archive",
          body: "Observed streamflow comes from GCIN-indexed streamflow records. ERA5-Land is reduced over each GCIN basin boundary to daily catchment means or sums. After joining by GCIN and date, every row represents one catchment on one day, with Qobs plus the meteorological and land-state drivers used by the model.",
          figure: this.renderDataAssemblyFigure()
        })}
        ${this.renderStoryPanel({
          index: "02",
          title: "Keep only hydrologically interpretable recession days",
          body: "Epsilon is evaluated on recession sequences because those days expose catchment drainage behavior. A sequence must decline for at least four days; the first decline day is dropped, a decreasing-rate filter removes irregular segments, and days with mean air temperature at or below 0 C are excluded as a snowmelt proxy.",
          figure: this.renderRecessionFigure()
        })}
        ${this.renderStoryPanel({
          index: "03",
          title: "Encode recent history and catchment context",
          body: "For each retained target day, the network sees the preceding 365 days of dynamic forcing and state variables, plus static attributes. The dynamic window carries precipitation, temperature, PET and root-zone soil moisture history; the static vector carries basin climate, soil, storage, area and location information.",
          figure: this.renderInputTensorFigure()
        })}
        ${this.renderStoryPanel({
          index: "04",
          title: "Infer epsilon inside the physics-informed LSTM",
          body: "The recurrent physics core matches the Ara reference implementation: one LSTM feeds dynamic epsilon and reset-flow heads plus bounded static alpha, LP and gamma heads. Ten component recession paths are averaged, and epsilon remains a learned daily coefficient inside the governing equation.",
          figure: this.renderModelFigure()
        })}
        ${this.renderStoryPanel({
          index: "05",
          title: "Constrain streamflow with the recession equation",
          body: "The inferred epsilon is used in the same state-reset, piecewise closed-form recession update as the reference code. Its four-term objective is also retained exactly: 25 L_path + 10 L_rhs + 0.1 L_smooth + 5 L_q0. These terms constrain the simulated Q path, local equation tendency, epsilon smoothness and reset-flow estimate.",
          figure: this.renderEquationFigure()
        })}
        ${this.renderStoryPanel({
          index: "06",
          id: "epsilon-overview-crossfit",
          title: "Generate out-of-fold daily epsilon estimates",
          body: "Five-fold temporal cross-fitting gives each eligible day an out-of-time epsilon estimate. In every rotation, one contiguous pre-1990 block and one contiguous post-1990 block are excluded from fitting for all catchments. Observed Q identifies recession days, defines each basin's Q10 low-flow and Q90 high-flow regimes, and evaluates reconstructed streamflow skill.",
          figure: this.renderOutputFigure()
        })}
        ${this.renderStoryPanel({
          index: "07",
          id: "epsilon-overview-era-shift",
          title: "Estimate one fold-adjusted post-1990 era shift",
          body: "Daily epsilon is reduced to a catchment-year-regime median when at least three recession days are available. The primary model regresses log annual epsilon on a post-1990 indicator with OOF-fold fixed effects. A series needs at least 10 valid years in each era and at least five pre and five post years inside paired folds. The effect, 95% HAC interval and FDR q-value all describe this same era-shift coefficient.",
          figure: this.renderEraShiftFigure()
        })}
        ${this.renderStoryPanel({
          index: "08",
          id: "epsilon-overview-global-field",
          title: "Test a spatially replicated global field story",
          body: "Catchments are assigned by 10-degree spatial blocks to a 40% discovery set and an untouched 60% confirmation set. Candidate field patterns are locked after discovery, then tested with random-effects aggregation, spatial block bootstrap intervals and Holm family-wise correction. This separates local unresolved labels from evidence about the global distribution.",
          figure: this.renderGlobalEvidenceFigure()
        })}
        ${this.renderStoryPanel({
          index: "09",
          id: "epsilon-overview-attribution-step",
          title: "Decompose epsilon change into GQ and Q components",
          body: "For every out-of-fold recession day, effective GQ is epsilon_effective multiplied by simulated Q. Period changes use geometric means so delta log epsilon = delta log GQ - delta log Q closes exactly. All-recession, low-flow and high-flow decompositions are selected separately. Combined means GQ and Q reinforce the same epsilon direction without one dominating; Offsetting means they push epsilon in opposite directions. The decomposition is descriptive, not causal or significance evidence.",
          figure: this.renderAttributionFigure()
        })}
        ${this.renderStoryPanel({
          index: "10",
          id: "epsilon-overview-trends",
          title: "Check whether the era result is robust to continuous time",
          body: "Continuous Theil-Sen and Kendall trends remain a prespecified sensitivity analysis, not the map classifier. Annual series with at least 20 years are fold-centered and trend-free prewhitened before FDR correction. Alternative 1985 and 1995 breakpoints and one-, three-, and five-day annual-support rules are also summarized without selecting whichever result is most significant.",
          figure: this.renderTrendFigure()
        })}
      </section>
    `;
  }

  renderStoryPanel({ index, id = "", title, body, figure }) {
    return `
      <article class="epsilon-story-panel"${id ? ` id="${this.escape(id)}"` : ""}>
        <div class="epsilon-story-copy">
          <div class="epsilon-story-index">${this.escape(index)}</div>
          <h4>${this.escape(title)}</h4>
          <p>${this.escape(body)}</p>
        </div>
        <div class="epsilon-story-figure">${figure}</div>
      </article>
    `;
  }

  renderDataAssemblyFigure() {
    return `
      <svg viewBox="0 0 520 170" role="img" aria-label="Data assembly workflow">
        ${this.svgBox(22, 28, 130, 48, "GCIN Qobs", "streamflow")}
        ${this.svgBox(22, 94, 130, 48, "ERA5-Land", "forcing + states")}
        ${this.svgArrow(162, 52, 230, 80)}
        ${this.svgArrow(162, 118, 230, 92)}
        ${this.svgBox(238, 54, 140, 58, "Catchment join", "GCIN / date")}
        ${this.svgArrow(388, 83, 448, 83)}
        ${this.svgBox(452, 54, 48, 58, "daily", "table")}
      </svg>
    `;
  }

  renderRecessionFigure() {
    return `
      <svg viewBox="0 0 520 170" role="img" aria-label="Recession filtering schematic">
        <polyline points="34,48 86,52 138,66 190,86 242,105 294,118 346,122 398,126 470,132" fill="none" stroke="#2563eb" stroke-width="4" stroke-linecap="round"/>
        <circle cx="86" cy="52" r="5" fill="#94a3b8"/><circle cx="138" cy="66" r="5" fill="#2563eb"/><circle cx="190" cy="86" r="5" fill="#2563eb"/><circle cx="242" cy="105" r="5" fill="#2563eb"/><circle cx="294" cy="118" r="5" fill="#2563eb"/>
        <text x="32" y="28" class="epsilon-svg-title">Qobs recession sequence</text>
        <text x="122" y="146" class="epsilon-svg-muted">drop first day -> keep declining days -> remove cold days</text>
        <line x1="86" y1="62" x2="86" y2="132" stroke="#ef4444" stroke-dasharray="4 4"/>
      </svg>
    `;
  }

  renderInputTensorFigure() {
    return `
      <svg viewBox="0 0 520 170" role="img" aria-label="Input tensor schematic">
        <rect x="28" y="44" width="210" height="78" rx="8" class="epsilon-svg-box"/>
        ${[0, 1, 2, 3, 4].map((i) => `<line x1="${62 + i * 32}" y1="50" x2="${62 + i * 32}" y2="116" class="epsilon-svg-grid"/>`).join("")}
        <text x="48" y="35" class="epsilon-svg-title">365-day dynamic window</text>
        <text x="52" y="144" class="epsilon-svg-muted">P, T, PET, soil moisture</text>
        ${this.svgArrow(250, 83, 310, 83)}
        ${this.svgBox(318, 50, 150, 66, "Static attributes", "climate, soil, area")}
      </svg>
    `;
  }

  renderModelFigure() {
    return `
      <svg viewBox="0 0 520 190" role="img" aria-label="LSTM epsilon model structure">
        ${this.svgBox(24, 68, 110, 54, "Inputs", "dynamic + static")}
        ${this.svgArrow(144, 95, 206, 95)}
        ${this.svgBox(214, 54, 120, 82, "LSTM", "hidden state")}
        ${this.svgArrow(344, 95, 400, 95)}
        <g>
          ${this.svgSmallPill(408, 28, "epsilon_t")}
          ${this.svgSmallPill(408, 62, "q_base_t")}
          ${this.svgSmallPill(408, 96, "alpha")}
          ${this.svgSmallPill(408, 130, "LP, gamma")}
        </g>
        <text x="210" y="164" class="epsilon-svg-muted">epsilon is inferred directly inside the recession equation</text>
      </svg>
    `;
  }

  renderEquationFigure() {
    return `
      <svg viewBox="0 0 520 190" role="img" aria-label="Physics equation and loss">
        <rect x="32" y="26" width="456" height="54" rx="8" class="epsilon-svg-formula"/>
        <text x="54" y="60" class="epsilon-svg-equation">dQ/dt = -epsilon * Q^2 - epsilon * alpha * AET * Q</text>
        ${this.svgArrow(260, 86, 260, 118)}
        ${this.svgBox(64, 122, 112, 44, "L_path", "Q path")}
        ${this.svgBox(204, 122, 112, 44, "L_rhs", "tendency")}
        ${this.svgBox(344, 122, 112, 44, "L_smooth + L_q0", "regularize")}
      </svg>
    `;
  }

  renderOutputFigure() {
    return `
      <svg viewBox="0 0 520 190" role="img" aria-label="Pre post epsilon contrast">
        ${this.svgBox(28, 42, 110, 50, "1950-1990", "pre")}
        ${this.svgBox(28, 108, 110, 50, "1991-2019", "post")}
        ${this.svgArrow(148, 68, 224, 90)}
        ${this.svgArrow(148, 132, 224, 108)}
        <rect x="232" y="50" width="124" height="92" rx="8" class="epsilon-svg-box"/>
        <path d="M250 126 C278 72, 310 72, 338 60" fill="none" stroke="#2563eb" stroke-width="3"/>
        <path d="M250 132 C280 100, 315 90, 338 78" fill="none" stroke="#b84235" stroke-width="3"/>
        <text x="255" y="42" class="epsilon-svg-title">epsilon CDF</text>
        ${this.svgArrow(366, 96, 428, 96)}
        ${this.svgBox(436, 56, 62, 78, "map", "classes")}
      </svg>
    `;
  }

  renderEraShiftFigure() {
    return `
      <svg viewBox="0 0 520 190" role="img" aria-label="Fold-adjusted era-shift inference">
        ${this.svgBox(22, 58, 118, 60, "Annual epsilon", "median >= 3 days")}
        ${this.svgArrow(150, 88, 210, 88)}
        ${this.svgBox(218, 48, 126, 80, "Fold fixed effect", "log epsilon ~ post")}
        ${this.svgArrow(354, 88, 408, 88)}
        ${this.svgBox(416, 30, 82, 42, "Effect", "% shift")}
        ${this.svgBox(416, 78, 82, 42, "95% CI", "HAC")}
        ${this.svgBox(416, 126, 82, 42, "FDR q", "evidence")}
        <text x="74" y="150" class="epsilon-svg-muted">10 years / era</text>
        <text x="205" y="150" class="epsilon-svg-muted">5 paired years / era</text>
      </svg>
    `;
  }

  renderAttributionFigure() {
    return `
      <svg viewBox="0 0 520 230" role="img" aria-label="GQ and Q component attribution">
        ${this.svgBox(24, 68, 112, 54, "epsilon(t)", "OOF daily")}
        <text x="154" y="100" class="epsilon-svg-equation">x</text>
        ${this.svgBox(180, 68, 112, 54, "Qsim(t)", "OOF daily")}
        ${this.svgArrow(302, 95, 358, 95)}
        ${this.svgBox(366, 68, 126, 54, "GQ(t)", "epsilon x Q")}
        <text x="72" y="154" class="epsilon-svg-muted">delta log epsilon</text>
        <text x="214" y="154" class="epsilon-svg-equation">=</text>
        <text x="252" y="154" class="epsilon-svg-muted">delta log GQ</text>
        <text x="374" y="154" class="epsilon-svg-equation">-</text>
        <text x="402" y="154" class="epsilon-svg-muted">delta log Q</text>
        ${this.svgSmallPill(44, 184, "All")}
        ${this.svgSmallPill(136, 184, "Low flow")}
        ${this.svgSmallPill(228, 184, "High flow")}
        <text x="334" y="200" class="epsilon-svg-muted">select condition -> classify contribution</text>
      </svg>
    `;
  }

  renderTrendFigure() {
    return `
      <svg viewBox="0 0 520 190" role="img" aria-label="Continuous trend and significance workflow">
        ${this.svgBox(22, 60, 112, 58, "Daily OOF", "epsilon / GQ / Q")}
        ${this.svgArrow(144, 89, 198, 89)}
        ${this.svgBox(206, 60, 112, 58, "Annual median", ">= 5 days")}
        ${this.svgArrow(328, 89, 382, 89)}
        ${this.svgBox(390, 38, 108, 48, "Theil-Sen", "% / decade")}
        ${this.svgBox(390, 102, 108, 48, "Kendall + FDR", "q < 0.05")}
        <text x="108" y="154" class="epsilon-svg-muted">at least 20 years</text>
        <text x="216" y="154" class="epsilon-svg-muted">fold-centered log series</text>
      </svg>
    `;
  }

  svgBox(x, y, w, h, title, subtitle) {
    return `
      <g>
        <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" class="epsilon-svg-box"/>
        <text x="${x + w / 2}" y="${y + h / 2 - 4}" text-anchor="middle" class="epsilon-svg-title">${this.escape(title)}</text>
        <text x="${x + w / 2}" y="${y + h / 2 + 14}" text-anchor="middle" class="epsilon-svg-muted">${this.escape(subtitle)}</text>
      </g>
    `;
  }

  renderGlobalEvidenceFigure() {
    return `
      <svg viewBox="0 0 520 210" role="img" aria-label="Spatial discovery and confirmation workflow">
        ${this.svgBox(20, 44, 106, 54, "Discovery", "40% of blocks")}
        ${this.svgArrow(136, 71, 192, 71)}
        ${this.svgBox(200, 44, 116, 54, "Lock candidates", "before testing")}
        ${this.svgArrow(326, 71, 382, 71)}
        ${this.svgBox(390, 44, 110, 54, "Confirmation", "60% of blocks")}
        ${this.svgBox(92, 130, 106, 46, "Random effects", "catchments")}
        ${this.svgBox(208, 130, 106, 46, "Block bootstrap", "spatial CI")}
        ${this.svgBox(324, 130, 106, 46, "Holm", "family-wise p")}
        <text x="260" y="198" text-anchor="middle" class="epsilon-svg-muted">field evidence never overwrites a catchment's local FDR class</text>
      </svg>
    `;
  }

  svgSmallPill(x, y, text) {
    return `
      <g>
        <rect x="${x}" y="${y}" width="82" height="24" rx="12" class="epsilon-svg-pill"/>
        <text x="${x + 41}" y="${y + 16}" text-anchor="middle" class="epsilon-svg-title">${this.escape(text)}</text>
      </g>
    `;
  }

  svgArrow(x1, y1, x2, y2) {
    return `
      <line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="epsilon-svg-arrow"/>
      <path d="M${x2},${y2} l-8,-4 l2,4 l-2,4 z" class="epsilon-svg-arrow-head"/>
    `;
  }

  showInspector(basin) {
    const title = `GCIN ${basin.GCIN}`;
    const curves = this.data.curves?.[String(basin.GCIN)] || {};
    const showDiagnostics = this.analysisView === "change";
    const streamflowRecord = showDiagnostics ? this.streamflowRecordHtml(basin) : "";
    const cards = `
      <div class="epsilon-inspector-metrics">
        ${this.metricCard("Area", `${this.formatNumber(basin.area_km2, 1)} km2`)}
        ${this.metricCard("Aridity", this.formatNumber(basin.Aridity, 3))}
        ${this.metricCard("Precip.", `${this.formatNumber(basin.Prec_mm, 1)} mm`)}
        ${this.metricCard("Temp.", `${this.formatNumber(basin.Temp_C, 1)} °C`)}
        ${this.skillMetricCard("NSE pre", this.formatNumber(basin.pre_nse, 3), basin.pre_nse)}
        ${this.skillMetricCard("NSE post", this.formatNumber(basin.post_nse, 3), basin.post_nse)}
        ${this.skillMetricCard("KGE pre", this.formatNumber(basin.pre_kge, 3), basin.pre_kge)}
        ${this.skillMetricCard("KGE post", this.formatNumber(basin.post_kge, 3), basin.post_kge)}
      </div>
      <div class="epsilon-signal-panel">
        <div class="epsilon-inspector-classification">${this.categoryBanner(basin)}</div>
        ${this.analysisView === "change" ? this.shiftPanel(basin) : ""}
        ${this.analysisView === "decomposition" ? this.attributionPanel(basin) : ""}
        ${this.analysisView === "trend" ? this.trendPanel(basin) : ""}
      </div>
    `;

    const cdfPreview = `
      <div
        class="epsilon-curve-preview"
        data-gcin="${this.escape(basin.GCIN)}"
        data-regime="${this.activeRegime}"
        aria-label="Open CDF panels"
        style="display:block;margin:2px 0 16px;cursor:pointer"
      >
        ${this.renderCombinedCdfSvg(curves)}
      </div>
    `;

    const sections = [this.activeRegime].map((regime) => `
      <section class="epsilon-stats-section" style="margin-top:10px;padding-top:12px;border-top:1px solid #e2e8f0">
        <h3 style="margin:0 0 8px;font-size:13px;letter-spacing:.04em;text-transform:uppercase;color:#64748b">${this.regimeLabel(regime)}</h3>
        ${this.renderStatsTable(basin, regime)}
      </section>
    `).join("");

    this.app.showInspector?.(title, `
      <p class="epsilon-inspector-context">
        Daily epsilon inferred within the physics-constrained streamflow equation. Comparison: 1950-1990 vs 1991-2019.
      </p>
      ${cards}
      ${showDiagnostics ? streamflowRecord : ""}
      ${showDiagnostics ? cdfPreview : ""}
      ${showDiagnostics ? sections : ""}
    `);
    this.bindCurvePreviews();
  }

  streamflowRecordHtml(basin) {
    const start = this.formatDate(basin.qobs_start);
    const end = this.formatDate(basin.qobs_end);
    if (!start || !end) return "";
    const validDays = Number(basin.qobs_valid_days);
    const calendarDays = Number(basin.qobs_calendar_days);
    const missingDays = Number(basin.qobs_missing_days);
    const missingPct = Number(basin.qobs_missing_pct);
    const coveragePct = Number(basin.qobs_coverage_pct);
    const gapRuns = Number(basin.qobs_gap_runs);
    const longGapCount = Number(basin.qobs_long_gap_count);
    const longGaps = Array.isArray(basin.qobs_long_gaps) ? basin.qobs_long_gaps : [];
    const availability = Number.isFinite(validDays) && Number.isFinite(calendarDays)
      ? `${this.formatInteger(validDays)} / ${this.formatInteger(calendarDays)} days (${this.formatNumber(coveragePct, 1)}%)`
      : "Unavailable";
    const missing = Number.isFinite(missingDays)
      ? `${this.formatInteger(missingDays)} days (${this.formatNumber(missingPct, 1)}%)${Number.isFinite(gapRuns) ? ` across ${this.formatInteger(gapRuns)} gaps` : ""}`
      : "Unavailable";
    const longGapHtml = longGaps.length
      ? `
        <div class="epsilon-completeness-subtitle">Long continuous gaps (≥30 days)</div>
        <ul class="epsilon-gap-list">
          ${longGaps.map((gap) => `
            <li>
              <span>${this.escape(this.formatDate(gap.start))} to ${this.escape(this.formatDate(gap.end))}</span>
              <span>${this.formatInteger(gap.days)} days</span>
            </li>
          `).join("")}
        </ul>
        ${Number.isFinite(longGapCount) && longGapCount > longGaps.length
          ? `<div class="epsilon-completeness-note">Showing the ${longGaps.length} longest of ${this.formatInteger(longGapCount)} long gaps.</div>`
          : ""}
      `
      : `<div class="epsilon-completeness-note">No continuous gaps of 30 days or longer.</div>`;
    return `
      <div class="epsilon-streamflow-completeness">
        <div class="epsilon-completeness-title">Time-series completeness</div>
        <div class="epsilon-completeness-row">
          <span>Missing variable</span>
          <span>Observed streamflow</span>
        </div>
        <div class="epsilon-completeness-row">
          <span>Record range</span>
          <span>${this.escape(start)} to ${this.escape(end)}</span>
        </div>
        <div class="epsilon-completeness-row">
          <span>Available</span>
          <span>${this.escape(availability)}</span>
        </div>
        <div class="epsilon-completeness-row">
          <span>Missing</span>
          <span>${this.escape(missing)}</span>
        </div>
        ${longGapHtml}
        <div class="epsilon-completeness-note">Gap statistics refer to observed streamflow. Meteorological forcing is complete on retained model dates.</div>
      </div>
    `;
  }

  formatDate(value) {
    if (!value || typeof value !== "string") return "";
    const match = value.match(/^(\d{4})-(\d{2})-(\d{2})/);
    return match ? `${match[1]}-${match[2]}-${match[3]}` : "";
  }

  formatYearMonth(value) {
    if (!value || typeof value !== "string") return "";
    const match = value.match(/^(\d{4})-(\d{2})/);
    return match ? `${match[1]}-${match[2]}` : "";
  }

  metricCard(label, value, signedValue = undefined) {
    const hasSignedValue = signedValue !== null
      && signedValue !== undefined
      && Number.isFinite(Number(signedValue));
    const color = hasSignedValue
      ? ` style="color:${Number(signedValue) < 0 ? "#2563eb" : "#b84235"}"`
      : "";
    const valueClass = hasSignedValue
      ? "epsilon-metric-value epsilon-metric-value--emphasis"
      : "epsilon-metric-value";
    return `
      <div class="epsilon-metric-card epsilon-metric-card--skill">
        <div class="${valueClass}"${color}>${this.escape(value)}</div>
        <div class="epsilon-metric-label">${this.escape(label)}</div>
      </div>
    `;
  }

  skillMetricCard(label, value, score) {
    const numeric = Number(score);
    const finite = Number.isFinite(numeric);
    const clamped = finite ? Math.max(0, Math.min(1, numeric)) : 0;
    const lightColor = finite ? this.skillScoreColor(clamped, "light") : "#64748b";
    const darkColor = finite ? this.skillScoreColor(clamped, "dark") : "#94a3b8";
    const title = finite
      ? `${label}: ${value}; color scale ${clamped.toFixed(3)} of 1.000`
      : `${label}: unavailable`;
    return `
      <div class="epsilon-metric-card">
        <div
          class="epsilon-metric-value epsilon-metric-value--skill"
          data-skill-score="${clamped.toFixed(6)}"
          title="${this.escape(title)}"
          style="--epsilon-skill-light:${lightColor};--epsilon-skill-dark:${darkColor}"
        >${this.escape(value)}</div>
        <div class="epsilon-metric-label">${this.escape(label)}</div>
      </div>
    `;
  }

  skillScoreColor(score, theme = "light") {
    const palettes = {
      light: [
        { at: 0, rgb: [190, 61, 82] },
        { at: 0.5, rgb: [158, 119, 0] },
        { at: 1, rgb: [0, 148, 73] },
      ],
      dark: [
        { at: 0, rgb: [251, 113, 133] },
        { at: 0.5, rgb: [250, 204, 21] },
        { at: 1, rgb: [74, 222, 128] },
      ],
    };
    const stops = palettes[theme] || palettes.light;
    const value = Math.max(0, Math.min(1, Number(score) || 0));
    const upperIndex = value <= stops[1].at ? 1 : 2;
    const lower = stops[upperIndex - 1];
    const upper = stops[upperIndex];
    const ratio = (value - lower.at) / (upper.at - lower.at);
    const rgb = lower.rgb.map((channel, index) => (
      Math.round(channel + (upper.rgb[index] - channel) * ratio)
    ));
    return `rgb(${rgb.join(",")})`;
  }

  categoryBanner(basin) {
    const color = this.basinColor(basin);
    const label = this.escape(this.basinLabel(basin));
    return `
      <div class="epsilon-classification-banner">
        <div class="epsilon-classification-kicker">${this.escape(this.analysisTitle())}</div>
        <div class="epsilon-classification-main">
          <span class="epsilon-classification-dot" style="background:${color}"></span>
          <span aria-label="${this.escape(this.basinLabel(basin))}">${label}</span>
        </div>
        <div class="epsilon-classification-subtitle">${this.escape(this.basinLabelSubtitle())}</div>
      </div>
    `;
  }

  shiftPanel(basin) {
    const regimes = [this.activeRegime];
    return `
      <div class="epsilon-shift-panel">
          <div class="epsilon-attribution-title">Post-1990 annual-median shift</div>
        ${regimes.map((regime) => {
          const state = this.shiftState(basin[`${regime}_epsilon_shift_class`]);
          const effect = basin[`${regime}_epsilon_shift_pct`];
          const ciLow = basin[`${regime}_epsilon_shift_ci_low_pct`];
          const ciHigh = basin[`${regime}_epsilon_shift_ci_high_pct`];
          const qValue = basin[`${regime}_epsilon_shift_q_value`];
          const preYears = Number(basin[`${regime}_epsilon_shift_pre_years`]);
          const postYears = Number(basin[`${regime}_epsilon_shift_post_years`]);
          const identifyingPre = Number(basin[`${regime}_epsilon_shift_identifying_pre_years`]);
          const identifyingPost = Number(basin[`${regime}_epsilon_shift_identifying_post_years`]);
          return `
            <div class="epsilon-shift-row">
               <div class="epsilon-trend-regime">${this.regimeShortLabel(regime)}</div>
              <div class="epsilon-shift-main">
                <span class="epsilon-trend-class">${state ? this.stateLabel(state) : "Insufficient"}</span>
                <span class="epsilon-shift-effect">${this.formatPct(effect)}</span>
              </div>
              <div class="epsilon-shift-ci">95% CI ${this.formatPct(ciLow)} to ${this.formatPct(ciHigh)}</div>
              <div class="epsilon-trend-meta">
                <span>FDR q ${this.formatSmall(qValue)}</span>
                <span>${Number.isFinite(preYears) && Number.isFinite(postYears) ? `${preYears} / ${postYears} yr` : "NA"}</span>
                <span>${Number.isFinite(identifyingPre) && Number.isFinite(identifyingPost) ? `${identifyingPre} / ${identifyingPost} paired yr` : "NA"}</span>
              </div>
            </div>
          `;
        }).join("")}
        <div class="epsilon-attribution-note">Effect, interval and FDR q refer to one fold-adjusted log-era coefficient. Unresolved means the direction is not established after multiple-testing correction, not that epsilon is stable.</div>
      </div>
    `;
  }

  attributionPanel(basin) {
    const regimes = [this.activeRegime];
    return `
      <div class="epsilon-attribution-panel">
        <div class="epsilon-attribution-title">GQ / Q component attribution</div>
        <div class="epsilon-attribution-columns" aria-hidden="true">
          <span></span><span></span><span>Driver</span><span>&Delta;ln GQ</span><span>&minus;&Delta;ln Q</span>
        </div>
        ${regimes.map((regime) => {
          const driver = basin[`${regime}_driver`] || "insufficient";
          return `
            <div class="epsilon-attribution-row">
              <span class="epsilon-attribution-swatch" style="--driver-color:${this.driverColor(driver) || "#cbd5e1"}"></span>
               <span class="epsilon-attribution-regime">${this.regimeShortLabel(regime)}</span>
              <span class="epsilon-attribution-driver">${this.driverLabel(driver)}</span>
              <span class="epsilon-attribution-value">${this.formatSigned(basin[`${regime}_gq_component_log`], 3)}</span>
              <span class="epsilon-attribution-value">${this.formatSigned(basin[`${regime}_q_component_log`], 3)}</span>
            </div>
          `;
        }).join("")}
        <div class="epsilon-attribution-note">Signed terms in &Delta;ln epsilon = &Delta;ln GQ + (&minus;&Delta;ln Q). Opposite signs mean Offsetting. Descriptive only; not a significance or causal test.</div>
      </div>
    `;
  }

  trendPanel(basin) {
    const regimes = [this.activeRegime];
    return `
      <div class="epsilon-trend-panel">
        <div class="epsilon-attribution-title">Sensitivity: continuous epsilon trend</div>
        ${regimes.map((regime) => {
          const state = this.trendState(basin[`${regime}_epsilon_trend_class`]);
          const years = Number(basin[`${regime}_epsilon_n_years`]);
          const slope = basin[`${regime}_epsilon_slope_pct_decade`];
          const qValue = basin[`${regime}_epsilon_q_value`];
          return `
            <div class="epsilon-trend-row">
               <div class="epsilon-trend-regime">${this.regimeShortLabel(regime)}</div>
              <div class="epsilon-trend-main">
                <span class="epsilon-trend-class">${state ? this.stateLabel(state) : "Insufficient"}</span>
                <span class="epsilon-trend-slope">${this.formatPct(slope)} / decade</span>
              </div>
              <div class="epsilon-trend-meta">
                <span>q ${this.formatSmall(qValue)}</span>
                <span>${Number.isFinite(years) ? `${years} yr` : "NA"}</span>
              </div>
            </div>
          `;
        }).join("")}
        <div class="epsilon-attribution-note">This prespecified sensitivity check asks whether change is monotonic through time. It does not replace the pre/post era estimate. Annual medians require &ge;5 recession days/year and &ge;20 years.</div>
      </div>
    `;
  }

  driverColor(driver) {
    return {
      gq: "#00897b",
      q: "#d97706",
      combined: "#111827",
      offsetting: "#64748b"
    }[driver] || null;
  }

  driverLabel(driver) {
    return {
      gq: "GQ-dominant",
      q: "Q-dominant",
      combined: "Combined",
      offsetting: "Offsetting",
      nonsignificant: "No significant driver",
      unresolved: "Unresolved",
      insufficient: "Insufficient"
    }[driver] || "Insufficient";
  }

  renderDriverLegend(compact = true) {
    const items = ["gq", "q", "combined", "offsetting"];
    return `
      <div class="epsilon-driver-legend${compact ? " epsilon-driver-legend--compact" : ""}">
        <div class="epsilon-driver-legend-title">${compact ? "Component driver" : "GQ / Q component class"}</div>
        <div class="epsilon-driver-legend-items">
          ${items.map((driver) => `
            <span class="epsilon-driver-legend-item">
              <span class="epsilon-driver-swatch" style="--driver-color:${this.driverColor(driver)}"></span>
              ${this.driverLabel(driver)}
            </span>
          `).join("")}
        </div>
        <div class="epsilon-driver-legend-note">${this.focusTitle()} · GQ = epsilon x simulated Q. Colors describe the pre/post component balance; they are not causal evidence.</div>
      </div>
    `;
  }

  renderOverviewLegend() {
    if (this.analysisView === "decomposition") return this.renderDriverLegend(false);
    return this.renderContinuousOverviewLegend();
  }

  renderContinuousOverviewLegend() {
    const regime = this.activeRegime;
    const values = this.basins
      .map((basin) => this.analysisView === "trend"
        ? Number(basin[`${regime}_epsilon_slope_pct_decade`])
        : this.shiftValue(basin, regime))
      .filter(Number.isFinite);
    const median = this.median(values);
    const stateFor = (basin) => this.analysisView === "trend"
      ? this.trendState(basin[`${regime}_epsilon_trend_class`])
      : this.shiftState(basin[`${regime}_epsilon_shift_class`]);
    const increaseCount = this.basins.filter((basin) => stateFor(basin) === "increase").length;
    const decreaseCount = this.basins.filter((basin) => stateFor(basin) === "decrease").length;
    const unit = this.analysisView === "trend" ? " / decade" : "";
    return `
      <div style="margin:0 0 14px">
        <div style="font-size:13px;font-weight:700;color:#0f172a;margin-bottom:8px">${this.escape(this.focusTitle())} ${this.analysisView === "trend" ? "continuous epsilon slope" : "post-1990 epsilon shift"}</div>
        ${this.renderContinuousLegendBar()}
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:10px">
          ${this.metricCard("Median estimate", `${this.formatPct(median)}${unit}`, median)}
          ${this.metricCard("FDR increase", increaseCount.toLocaleString())}
          ${this.metricCard("FDR decrease", decreaseCount.toLocaleString())}
        </div>
      </div>
    `;
  }

  renderAttributionDefinitions() {
    return `
      <div class="epsilon-driver-guide">
        <div class="epsilon-driver-guide-formula"><code>GQ contribution = delta log GQ</code><code>Q contribution = -delta log Qsim</code></div>
        <div class="epsilon-driver-guide-grid">
          <div><strong>GQ-dominant</strong><span>Both contributions reinforce the epsilon change; GQ supplies more than two thirds of their absolute total.</span></div>
          <div><strong>Q-dominant</strong><span>Both contributions reinforce the epsilon change; Q supplies more than two thirds of their absolute total.</span></div>
          <div><strong>Combined</strong><span>GQ and Q reinforce the same epsilon direction, and each supplies between one third and two thirds.</span></div>
          <div><strong>Offsetting</strong><span>The contributions oppose: one pushes epsilon upward while the other pushes it downward. The neutral outlined symbol marks this class.</span></div>
        </div>
      </div>
    `;
  }

  trendState(value) {
    if (value === "increase" || value === "decrease") return value;
    if (value === "no_significant_trend") return "unresolved";
    return null;
  }

  shiftState(value) {
    if (value === "increase" || value === "decrease" || value === "unresolved") return value;
    return null;
  }

  layerName() {
    if (this.analysisView === "decomposition") return "GQ / Q decomposition";
    if (this.analysisView === "trend") return "Temporal robustness";
    return "Epsilon era change";
  }

  moduleTitle() {
    return this.analysisTitle();
  }

  focusTitle() {
    if (this.activeRegime === "low") return "Low-flow";
    if (this.activeRegime === "high") return "High-flow";
    return "All-recession";
  }

  analysisTitle() {
    if (this.analysisView === "decomposition") return "GQ / Q decomposition";
    if (this.analysisView === "trend") return "Temporal robustness";
    return "Epsilon change";
  }

  overviewText() {
    if (this.analysisView === "decomposition") {
      return `${this.focusTitle()} view of the exact identity delta log epsilon = delta log GQ - delta log Qsim. Color shows which algebraic component dominates the estimated era change; it is descriptive, not causal attribution.`;
    }
    if (this.analysisView === "trend") {
      return `${this.focusTitle()} fold-centered Theil-Sen epsilon slopes. This module checks whether the pre/post result is also compatible with a monotonic temporal trend; it never replaces the era comparison.`;
    }
    return `${this.focusTitle()} fold-adjusted annual-median epsilon change after 1990. Color shows the estimated effect for every supported catchment; FDR q-values quantify local evidence without turning Unresolved into Stable.`;
  }

  renderLegendDefinitions() {
    const regimeDefinition = this.activeRegime === "low"
      ? "Low flow uses recession days with observed Q at or below each catchment's Q10."
      : this.activeRegime === "high"
        ? "High flow uses recession days with observed Q at or above each catchment's Q90."
        : "All recession includes every retained recession day, without a Q10 or Q90 subset.";
    if (this.analysisView === "decomposition") {
      return `
        <div class="epsilon-overview-definitions">
          <div class="epsilon-overview-definition"><span class="epsilon-overview-definition-title">Flow condition</span><span>${regimeDefinition}</span></div>
          <div class="epsilon-overview-definition"><span class="epsilon-overview-definition-title">Map color</span><span>Color identifies the larger algebraic contribution to the estimated epsilon era shift. Combined means reinforcement without a dominant term; Offsetting means opposing signed terms.</span></div>
          <div class="epsilon-overview-definition"><span class="epsilon-overview-definition-title">Evidence boundary</span><span>The decomposition is exact and descriptive. It is not a significance test and does not establish a causal climate driver.</span></div>
        </div>
      `;
    }
    if (this.analysisView === "trend") {
      return `
        <div class="epsilon-overview-definitions">
          <div class="epsilon-overview-definition"><span class="epsilon-overview-definition-title">Flow condition</span><span>${regimeDefinition}</span></div>
          <div class="epsilon-overview-definition"><span class="epsilon-overview-definition-title">Color scale</span><span>Violet indicates a negative Theil-Sen slope, gray is near zero, and amber indicates a positive slope. Values are percent per decade.</span></div>
          <div class="epsilon-overview-definition"><span class="epsilon-overview-definition-title">Significance</span><span>Increase or Decrease requires Kendall FDR q &lt; 0.05. Otherwise the direction is unresolved; that is not evidence that epsilon is stable.</span></div>
        </div>
      `;
    }
    return `
      <div class="epsilon-overview-definitions">
        <div class="epsilon-overview-definition"><span class="epsilon-overview-definition-title">Flow condition</span><span>${regimeDefinition}</span></div>
        <div class="epsilon-overview-definition"><span class="epsilon-overview-definition-title">Era effect</span><span>Color is the fold-adjusted percent change in annual-median epsilon after 1990. It remains continuous so unresolved catchments are not erased or painted as zero.</span></div>
        <div class="epsilon-overview-definition"><span class="epsilon-overview-definition-title">Local evidence</span><span>Annual medians require at least three recession days, 10 years in each era, and five paired-fold years per era. Increase or Decrease requires Benjamini-Hochberg FDR q &lt; 0.05; otherwise the direction is Unresolved, not Stable.</span></div>
      </div>
    `;
  }

  basinLabel(basin) {
    const regime = this.activeRegime;
    if (this.analysisView === "decomposition") return `${this.focusTitle()} ${this.driverLabel(basin[`${regime}_driver`])}`;
    if (this.analysisView === "trend") {
      const state = this.trendState(basin[`${regime}_epsilon_trend_class`]);
      return `${this.stateLabel(state)}: ${this.formatPct(basin[`${regime}_epsilon_slope_pct_decade`])} / decade`;
    }
    const state = this.shiftState(basin[`${regime}_epsilon_shift_class`]);
    return `${this.stateLabel(state)}: ${this.formatPct(this.shiftValue(basin, regime))}`;
  }

  basinLabelSubtitle() {
    if (this.analysisView === "decomposition") return "Exact GQ / Q identity for the selected flow condition; descriptive rather than causal.";
    if (this.analysisView === "trend") return "Fold-centered Theil-Sen sensitivity estimate; evidence is classified after FDR correction.";
    return "Fold-adjusted post-1990 annual-median effect; evidence is classified after FDR correction.";
  }

  stateLabel(state) {
    return {
      decrease: "Decrease",
      unresolved: "Unresolved",
      increase: "Increase"
    }[state] || "insufficient";
  }

  basinColor(basin) {
    const regime = this.activeRegime;
    if (this.analysisView === "decomposition") {
      return this.driverColor(basin[`${regime}_driver`]) || "#d8dee8";
    }
    if (this.analysisView === "trend") {
      return this.continuousColor(Number(basin[`${regime}_epsilon_slope_pct_decade`]));
    }
    return this.continuousColor(this.shiftValue(basin, regime));
  }

  computeContinuousExtent() {
    const regime = this.activeRegime;
    if (this.analysisView === "decomposition") return 1;
    const values = this.basins
      .map((basin) => this.analysisView === "trend"
        ? Number(basin[`${regime}_epsilon_slope_pct_decade`])
        : this.shiftValue(basin, regime))
      .filter(Number.isFinite)
      .map(Math.abs)
      .sort((a, b) => a - b);
    if (!values.length) return 50;
    const p95 = values[Math.min(values.length - 1, Math.floor(values.length * 0.95))];
    return this.analysisView === "trend"
      ? Math.max(2, Math.min(60, p95 || 10))
      : Math.max(10, Math.min(80, p95 || 50));
  }

  continuousColor(value) {
    if (!Number.isFinite(value)) return "#d8dee8";
    const extent = this.colorScaleExtent || 50;
    const t = Math.max(-1, Math.min(1, value / extent));
    const palette = this.continuousPalette();
    if (Math.abs(t) < 0.02) return palette.neutral;
    if (t < 0) return this.mix(palette.negative, palette.neutral, t + 1);
    return this.mix(palette.neutral, palette.positive, t);
  }

  continuousPalette() {
    if (this.analysisView === "trend") {
      return { negative: "#6750c9", neutral: "#d7dee8", positive: "#e58a17" };
    }
    return { negative: "#00d7ff", neutral: "#cbd5e1", positive: "#ff3bbd" };
  }

  shiftValue(basin, regime) {
    return Number(basin?.[`${regime}_epsilon_shift_pct`]);
  }

  renderContinuousLegendBar() {
    const extent = this.colorScaleExtent || 50;
    const palette = this.continuousPalette();
    return `
      <div class="epsilon-continuous-key">
        <div style="height:12px;border-radius:999px;background:linear-gradient(90deg,${palette.negative} 0%,${palette.neutral} 50%,${palette.positive} 100%);border:1px solid rgba(15,23,42,.12)"></div>
        <div class="epsilon-continuous-labels">
          <span>${this.formatPct(-extent)}</span>
          <span>0%</span>
          <span>${this.formatPct(extent)}</span>
        </div>
      </div>
    `;
  }

  renderStatsTable(basin, regime) {
    const rows = [
      ["Mean", basin[`${regime}_pre_mean`], basin[`${regime}_post_mean`]],
      ["IQR", `${this.formatSmall(basin[`${regime}_pre_q25`])}-${this.formatSmall(basin[`${regime}_pre_q75`])}`, `${this.formatSmall(basin[`${regime}_post_q25`])}-${this.formatSmall(basin[`${regime}_post_q75`])}`],
      ["N", basin[`${regime}_pre_n`], basin[`${regime}_post_n`]]
    ];
    return `
      <table style="width:100%;border-collapse:collapse;font-size:12px">
        <thead>
          <tr style="color:#64748b">
            <th style="text-align:left;padding:5px 4px;border-bottom:1px solid #e2e8f0">Metric</th>
            <th style="text-align:right;padding:5px 4px;border-bottom:1px solid #e2e8f0">Pre</th>
            <th style="text-align:right;padding:5px 4px;border-bottom:1px solid #e2e8f0">Post</th>
          </tr>
        </thead>
        <tbody>
          ${rows.map(([label, pre, post]) => `
            <tr>
              <td style="padding:5px 4px;border-bottom:1px solid #edf2f7">${this.escape(label)}</td>
              <td style="padding:5px 4px;border-bottom:1px solid #edf2f7;text-align:right">${this.escape(this.formatSmall(pre))}</td>
              <td style="padding:5px 4px;border-bottom:1px solid #edf2f7;text-align:right">${this.escape(this.formatSmall(post))}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
  }

  renderCombinedCdfSvg(curves) {
    const theme = this.themeColors();
    const width = 300;
    const regimes = [this.activeRegime];
    const rowHeight = 118;
    const height = rowHeight + 18;
    const margin = { left: 34, right: 10, top: 20, bottom: 18 };
    const rows = regimes.map((regime, rowIndex) => {
      const curve = curves?.[regime];
      if (!curve?.x?.length) {
        const y = 18 + rowIndex * rowHeight;
        return `<text x="${margin.left}" y="${y + 28}" fill="#94a3b8" font-size="10">No ${this.regimeShortLabel(regime)} data</text>`;
      }
      const x = curve.x.map(Number);
      const pre = curve.preCdf.map(Number);
      const post = curve.postCdf.map(Number);
      const minX = Math.min(...x);
      const maxX = Math.max(...x, minX + 1e-12);
      const plotTop = margin.top + rowIndex * rowHeight;
      const plotBottom = plotTop + rowHeight - 26;
      const plotW = width - margin.left - margin.right;
      const plotH = plotBottom - plotTop;
      const sx = (value) => margin.left + ((value - minX) / Math.max(1e-12, maxX - minX)) * plotW;
      const sy = (value) => plotBottom - value * plotH;
      const path = (values) => x.map((value, index) => `${index ? "L" : "M"}${sx(value).toFixed(1)},${sy(values[index] || 0).toFixed(1)}`).join(" ");
      return `
        <line x1="${margin.left}" y1="${plotBottom}" x2="${width - margin.right}" y2="${plotBottom}" stroke="${theme.axis}"/>
        <line x1="${margin.left}" y1="${plotTop}" x2="${margin.left}" y2="${plotBottom}" stroke="${theme.axis}"/>
        <text x="${margin.left}" y="${plotTop - 5}" fill="${theme.text}" font-size="10" font-weight="400">${this.regimeShortLabel(regime)}</text>
        <text x="${width - margin.right}" y="${plotTop - 5}" fill="${theme.muted}" font-size="9" text-anchor="end">${this.formatSmall(minX)}-${this.formatSmall(maxX)}</text>
        <path d="${path(pre)}" fill="none" stroke="#2563eb" stroke-width="1.6"/>
        <path d="${path(post)}" fill="none" stroke="#b84235" stroke-width="1.6"/>
      `;
    }).join("");
    return `
      <svg viewBox="0 0 ${width} ${height}" style="display:block;width:100%;height:auto;background:${theme.card};pointer-events:none">
        ${rows}
      </svg>
    `;
  }

  bindCurvePreviews() {
    this.ensurePreviewStyles();
    setTimeout(() => {
      document.querySelectorAll(".epsilon-curve-preview").forEach((node) => {
        node.onclick = () => this.openDistributionModal(node.dataset.gcin, node.dataset.regime);
      });
    }, 0);
  }

  ensurePreviewStyles() {
    if (document.getElementById("epsilon-preview-styles")) return;
    const style = document.createElement("style");
    style.id = "epsilon-preview-styles";
    style.textContent = `
      .epsilon-toolbar{position:fixed;top:14px;left:calc(50% + 155px);transform:translateX(-50%);z-index:120;display:flex;align-items:center;gap:8px;max-width:calc(100vw - 380px);min-height:42px;padding:5px 6px 5px 12px;border:1px solid #dbe3ef;border-radius:8px;background:rgba(255,255,255,.96);box-shadow:0 8px 24px rgba(15,23,42,.12);backdrop-filter:blur(10px)}
      .epsilon-toolbar-label{color:#334155;font-size:12.5px;font-weight:700;white-space:nowrap}
      .epsilon-toolbar-segments{display:flex;align-items:center;gap:2px;padding:2px;border-radius:6px;background:#eef2f7}
      .epsilon-toolbar-segments button,.epsilon-toolbar-overview{min-height:32px;border:0;border-radius:5px;padding:0 12px;color:#64748b;background:transparent;font-size:12px;font-weight:700;white-space:nowrap;cursor:pointer}
      .epsilon-toolbar-segments button:hover{color:#0f172a}
      .epsilon-toolbar-segments button.active{background:#fff;color:#1d4ed8;box-shadow:0 1px 3px rgba(15,23,42,.14)}
      .epsilon-toolbar-overview{background:#183b56;color:#fff}
      .epsilon-toolbar-overview:hover{background:#0f2f48}
      .epsilon-curve-preview{box-sizing:border-box;border:1px solid #e2e8f0;border-radius:6px;overflow:hidden;background:#fbfdff;transition:background-color .16s ease,border-color .16s ease,box-shadow .16s ease}
      .epsilon-curve-preview:hover{background:#eef7ff;border-color:#60a5fa!important;box-shadow:0 0 0 1px rgba(96,165,250,.26),0 0 18px rgba(96,165,250,.18)}
      .epsilon-inspector-context{margin:0 0 16px;color:#64748b;font-size:12.5px;line-height:1.58}
      .epsilon-inspector-metrics{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-bottom:16px}
      .epsilon-signal-panel{margin-bottom:16px;padding:12px;border:1px solid #dbe3ef;border-radius:7px;background:#f8fafc}
      .epsilon-classification-kicker{margin-bottom:6px;font-size:10.5px;font-weight:700;color:#94a3b8;text-transform:uppercase}
      .epsilon-classification-main{display:flex;align-items:flex-start;gap:8px;color:#0f172a;font-size:12.5px;font-weight:700;line-height:1.4}
      .epsilon-classification-dot{width:13px;height:13px;margin-top:2px;border:1px solid rgba(15,23,42,.2);border-radius:50%;flex:0 0 auto}
      .epsilon-classification-subtitle{margin-top:6px;color:#64748b;font-size:11.5px;line-height:1.5}
      .epsilon-metric-card{background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px}
      .epsilon-metric-value{font-size:17px;font-weight:400;color:#64748b}
      .epsilon-metric-value--emphasis,
      .epsilon-metric-value--skill{font-weight:700}
      .epsilon-metric-value--skill{color:var(--epsilon-skill-light)}
      .epsilon-metric-label{font-size:12px;color:#64748b;margin-top:3px}
      .epsilon-streamflow-completeness{margin:0 0 16px;padding:14px 3px 0;border:0;border-top:1px solid #dbe3ef;border-radius:0;background:transparent;font-size:11.5px;line-height:1.5;color:#64748b}
      .epsilon-completeness-title{margin-bottom:8px;color:#334155;font-size:12px;font-weight:700;text-transform:none;letter-spacing:0}
      .epsilon-completeness-row{display:grid;grid-template-columns:92px minmax(0,1fr);gap:8px;padding:2px 0}
      .epsilon-completeness-row span:last-child{color:#475569}
      .epsilon-completeness-subtitle{margin-top:8px;padding-top:7px;border-top:1px solid #e2e8f0;color:#64748b}
      .epsilon-gap-list{list-style:none;margin:4px 0 0;padding:0}
      .epsilon-gap-list li{display:flex;justify-content:space-between;gap:8px;padding:2px 0;color:#475569}
      .epsilon-gap-list li span:last-child{white-space:nowrap;color:#64748b}
      .epsilon-completeness-note{margin-top:6px;color:#64748b}
      .epsilon-overview-modal{position:fixed;inset:0;display:none;align-items:center;justify-content:center;z-index:150;pointer-events:none}
      .epsilon-overview-modal.visible{display:flex}
      .epsilon-overview-dialog{width:min(1040px,calc(100vw - 64px));max-height:min(840px,calc(100vh - 64px));background:#fff;border:1px solid #dbe3ef;border-radius:8px;box-shadow:0 22px 58px rgba(15,23,42,.24);display:flex;flex-direction:column;overflow:hidden;pointer-events:auto}
      .epsilon-overview-header{height:58px;min-height:58px;flex:0 0 58px;padding:0 16px;border-bottom:1px solid #e2e8f0;display:flex;align-items:center;justify-content:space-between;gap:16px;background:#f8fafc}
      .epsilon-overview-title{font-size:15px;font-weight:700;color:#0f172a;letter-spacing:0}
      .epsilon-overview-close{width:32px;height:32px;border:0;background:transparent;color:#64748b;font-size:0;line-height:1;cursor:pointer;border-radius:6px;position:relative;padding:0}
      .epsilon-overview-close:hover{background:#eef2f7;color:#0f172a}
      .epsilon-overview-close::before,.epsilon-overview-close::after{content:"";position:absolute;left:50%;top:50%;width:12px;height:1.5px;border-radius:999px;background:currentColor;transform-origin:center}
      .epsilon-overview-close::before{transform:translate(-50%,-50%) rotate(45deg)}
      .epsilon-overview-close::after{transform:translate(-50%,-50%) rotate(-45deg)}
      .epsilon-overview-body{padding:20px;overflow:auto;color:#334155;font-size:13.5px;line-height:1.65}
      .epsilon-overview-layout{display:grid;grid-template-columns:134px minmax(0,1fr);align-items:start}
      .epsilon-overview-nav{position:sticky;top:0;display:grid;gap:2px;padding:2px 14px 6px 0;border-right:1px solid #e2e8f0}
      .epsilon-overview-nav-title{margin:0 0 7px;padding:0 8px;color:#94a3b8;font-size:10.5px;font-weight:700;text-transform:uppercase}
      .epsilon-overview-nav a{display:block;padding:7px 8px;border-left:2px solid transparent;color:#64748b;font-size:11.5px;font-weight:600;line-height:1.3;text-decoration:none}
      .epsilon-overview-nav a:hover{color:#0f172a;background:#f8fafc}
      .epsilon-overview-nav a.active{border-left-color:#2563eb;background:#eff6ff;color:#1d4ed8}
      .epsilon-overview-content{min-width:0;padding-left:18px}
      .epsilon-overview-content section,.epsilon-story-panel[id]{scroll-margin-top:16px}
      .epsilon-overview-body section + section{margin-top:18px;padding-top:16px;border-top:1px solid #e2e8f0}
      .epsilon-overview-body h3{margin:0 0 8px;font-size:14px;color:#0f172a;letter-spacing:.03em;text-transform:uppercase}
      .epsilon-overview-body p{margin:0 0 10px}
      .epsilon-overview-lead{color:#475569}
      .epsilon-overview-definitions{display:grid;gap:10px;margin-top:10px;color:#475569;font-size:13px;line-height:1.58}
      .epsilon-overview-definition{position:relative;padding-left:14px}
      .epsilon-overview-definition::before{content:"";position:absolute;left:1px;top:7px;width:5px;height:5px;border-radius:50%;background:#64748b}
      .epsilon-overview-definition-title{display:block;margin-bottom:1px;color:#0f172a;font-weight:700}
      .epsilon-overview-attribution{margin-top:14px;padding-top:12px;border-top:1px solid #e2e8f0}
      .epsilon-evidence-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-top:12px}
      .epsilon-evidence-item{min-width:0;padding:12px;border:1px solid #dbe3ef;border-radius:7px;background:#f8fafc}
      .epsilon-evidence-item--primary{border-top:3px solid #15803d;padding-top:10px}
      .epsilon-evidence-kicker{display:block;margin-bottom:5px;color:#64748b;font-size:10.5px;font-weight:700;text-transform:uppercase}
      .epsilon-evidence-value{color:#166534;font-size:23px;font-weight:750;line-height:1.1;font-variant-numeric:tabular-nums}
      .epsilon-evidence-item>strong,.epsilon-evidence-association strong{display:block;margin-top:5px;color:#0f172a;font-size:12px;line-height:1.45}
      .epsilon-evidence-item p{margin:6px 0!important;color:#475569;font-size:12px;line-height:1.55}
      .epsilon-evidence-item small{display:block;color:#64748b;font-size:11px;line-height:1.5}
      .epsilon-evidence-association{display:grid;grid-template-columns:minmax(190px,1fr) auto;gap:4px 16px;align-items:end;margin-top:10px;padding:11px 0;border-top:1px solid #dbe3ef;border-bottom:1px solid #dbe3ef}
      .epsilon-evidence-association .epsilon-evidence-kicker{margin:0}
      .epsilon-evidence-association strong{margin:2px 0 0}
      .epsilon-evidence-association-value{color:#166534;font-size:20px;font-weight:750;line-height:1;font-variant-numeric:tabular-nums}
      .epsilon-evidence-association p{grid-column:1/-1;margin:3px 0 0!important;color:#64748b;font-size:11.5px;line-height:1.52}
      .epsilon-evidence-guardrail{margin-top:10px;padding-left:10px;border-left:2px solid #94a3b8;color:#64748b;font-size:11.5px;line-height:1.52}
      .epsilon-evidence-guardrail strong{color:#334155}
      .epsilon-evidence-method{margin:8px 0 0!important;color:#64748b;font-size:11px;line-height:1.5}
      .epsilon-method-facts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px 16px;margin-top:10px}
      .epsilon-method-facts>div{display:grid;gap:2px;padding:8px 0;border-top:1px solid #edf1f6}
      .epsilon-method-facts strong{color:#0f172a;font-size:12px}
      .epsilon-method-facts span{color:#64748b;font-size:11.5px;line-height:1.52}
      .epsilon-equation-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:12px}
      .epsilon-equation-card{display:grid;gap:5px;min-width:0;padding:9px;border:1px solid #dbe3ef;border-radius:6px;background:#f8fafc}
      .epsilon-equation-card:last-child{grid-column:1/-1}
      .epsilon-equation-card>span{color:#334155;font-size:11.5px;font-weight:700}
      .epsilon-equation-card code,.epsilon-driver-guide code{display:block;overflow-wrap:anywhere;color:#0f172a;font:600 11px/1.5 Consolas,monospace}
      .epsilon-method-caution{margin:10px 0 0!important;padding-left:10px;border-left:2px solid #94a3b8;color:#64748b;font-size:11.5px;line-height:1.52}
      .epsilon-method-caution strong{color:#334155}
      .epsilon-shift-panel,.epsilon-attribution-panel,.epsilon-trend-panel{margin:12px 0 0;padding:12px 0 0;border:0;border-top:1px solid #dbe3ef;border-radius:0;background:transparent}
      .epsilon-attribution-title,.epsilon-driver-legend-title{font-size:12px;font-weight:700;color:#0f172a;margin-bottom:8px}
      .epsilon-attribution-columns,.epsilon-attribution-row{display:grid;grid-template-columns:14px 56px minmax(70px,1fr) 48px 48px;gap:6px;align-items:center}
      .epsilon-attribution-columns{padding:0 0 3px;font-size:10px;color:#94a3b8;text-align:right}
      .epsilon-attribution-columns span:nth-child(3){text-align:left}
      .epsilon-attribution-row{padding:5px 0;font-size:11.5px;color:#475569}
      .epsilon-attribution-swatch,.epsilon-driver-swatch{width:11px;height:11px;border-radius:50%;border:1px solid rgba(15,23,42,.22);background:var(--driver-color);box-sizing:border-box;display:inline-block;flex:0 0 auto}
      .epsilon-attribution-regime{color:#64748b}
      .epsilon-attribution-driver{font-weight:700;color:#334155}
      .epsilon-attribution-value{text-align:right;color:#64748b;white-space:nowrap;font-variant-numeric:tabular-nums}
      .epsilon-attribution-note,.epsilon-driver-legend-note{margin-top:8px;font-size:11px;line-height:1.5;color:#64748b}
      .epsilon-shift-row{min-width:0;padding:8px 0;font-size:11.5px;color:#64748b}
      .epsilon-shift-row+.epsilon-shift-row{border-top:1px solid #e2e8f0}
      .epsilon-shift-main{display:flex;align-items:baseline;justify-content:space-between;gap:10px;min-width:0;margin-top:2px}
      .epsilon-shift-effect{color:#0f172a;font-size:15px;font-weight:700;white-space:nowrap;font-variant-numeric:tabular-nums}
      .epsilon-shift-ci{margin-top:2px;color:#475569;font-variant-numeric:tabular-nums}
      .epsilon-trend-row{min-width:0;padding:8px 0;font-size:11.5px;color:#64748b}
      .epsilon-trend-row+.epsilon-trend-row{border-top:1px solid #e2e8f0}
      .epsilon-trend-main{display:flex;align-items:baseline;justify-content:space-between;gap:10px;min-width:0;margin-top:2px}
      .epsilon-trend-meta{display:flex;flex-wrap:wrap;gap:2px 12px;margin-top:4px;color:#64748b}
      .epsilon-trend-regime{font-size:10.5px;font-weight:600;color:#64748b;text-transform:uppercase}
      .epsilon-trend-class{min-width:0;font-weight:700;color:#334155;line-height:1.3}
      .epsilon-trend-slope{white-space:nowrap}
      .epsilon-driver-legend{margin-top:10px}
      .epsilon-continuous-key{padding:10px;border:1px solid #e2e8f0;border-radius:6px;background:#f8fafc}
      .epsilon-continuous-labels{display:flex;justify-content:space-between;margin-top:6px;color:#64748b;font-size:11.5px}
      .epsilon-driver-legend--compact{margin-top:0}
      .epsilon-driver-legend-items{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px 10px}
      .epsilon-driver-legend-item{display:inline-flex;align-items:center;gap:6px;color:#475569;font-size:11.5px;white-space:nowrap}
      .epsilon-driver-guide{margin-top:10px;padding:10px 0 0;border-top:1px solid #e2e8f0}
      .epsilon-driver-guide-formula{display:flex;flex-wrap:wrap;gap:5px 16px;margin-bottom:9px}
      .epsilon-driver-guide-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px 16px}
      .epsilon-driver-guide-grid>div{display:grid;gap:1px}
      .epsilon-driver-guide-grid strong{color:#0f172a;font-size:12px}
      .epsilon-driver-guide-grid span{color:#64748b;font-size:11.5px;line-height:1.5}
      .epsilon-overview-metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:16px 0}
      .epsilon-overview-filter{margin:14px 0 16px;padding:10px;border:1px solid #e2e8f0;border-radius:6px;background:#f8fafc}
      .epsilon-filter-title{font-size:12px;font-weight:800;margin-bottom:8px;color:#0f172a}
      .epsilon-filter-grid{display:grid;grid-template-columns:minmax(112px,.7fr) minmax(112px,.7fr) minmax(180px,1.4fr);gap:10px;align-items:end}
      .epsilon-filter-field{display:grid;gap:4px;margin:0;color:#64748b;font-size:12px;line-height:1.35}
      .epsilon-filter-field select,.epsilon-filter-field input{width:100%;box-sizing:border-box;border:1px solid #cbd5e1;border-radius:6px;background:#fff;color:#0f172a;font-size:13px;padding:6px 7px}
      .epsilon-filter-range{padding:0!important;accent-color:#2563eb}
      .epsilon-field-mode-note{grid-column:1/-1;margin:1px 0 0;padding:9px 10px;border-left:2px solid #22c55e;background:#fff;color:#64748b;font-size:11.5px;line-height:1.5}
      .epsilon-field-mode-note strong{color:#334155}
      .epsilon-filter-count{font-size:12px;color:#475569;margin-top:8px;line-height:1.4}
      .epsilon-story{position:relative}
      .epsilon-story::before{content:"";position:absolute;left:14px;top:54px;bottom:18px;width:2px;background:linear-gradient(#93b4ff,#b84235);border-radius:999px;opacity:.45}
      .epsilon-story-lead{max-width:820px;color:#475569;font-size:13px;line-height:1.7;margin:0 0 14px}
      .epsilon-story-panel{position:relative;display:grid;grid-template-columns:minmax(250px,.92fr) minmax(320px,1.18fr);gap:18px;align-items:center;padding:12px 0 14px 42px}
      .epsilon-story-copy{position:relative}
      .epsilon-story-index{position:absolute;left:-42px;top:0;width:30px;height:30px;border-radius:50%;background:#1d4ed8;color:#fff;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:800;box-shadow:0 0 0 5px rgba(255,255,255,.94)}
      .epsilon-story-copy h4{margin:0 0 7px;font-size:14px;color:#0f172a;letter-spacing:0}
      .epsilon-story-copy p{margin:0;font-size:13px;line-height:1.68;color:#475569}
      .epsilon-story-figure{min-height:150px;border:1px solid #e2e8f0;border-radius:7px;background:#fff;padding:8px;box-shadow:0 8px 24px rgba(15,23,42,.045)}
      .epsilon-story-figure svg{display:block;width:100%;height:auto}
      .epsilon-svg-box,.epsilon-svg-formula{fill:#f8fafc;stroke:#cbd5e1;stroke-width:1.2}
      .epsilon-svg-pill{fill:#e8f0ff;stroke:#93b4ff;stroke-width:1}
      .epsilon-svg-grid{stroke:#dbe3ef;stroke-width:1}
      .epsilon-svg-arrow{stroke:#94a3b8;stroke-width:1.4}
      .epsilon-svg-arrow-head{fill:#94a3b8}
      .epsilon-svg-title{fill:#0f172a;font-size:12px;font-weight:700}
      .epsilon-svg-muted{fill:#64748b;font-size:11.5px}
      .epsilon-map-legend-note{margin-top:9px;color:#64748b;font-size:11.5px;line-height:1.5}
      .epsilon-svg-equation{fill:#0f172a;font-size:15px;font-family:Consolas,monospace;font-weight:700}
      body.theme-dark .epsilon-curve-preview{background:#111827;border-color:#263449}
      body.theme-dark .epsilon-toolbar{background:rgba(15,23,42,.96);border-color:#334155;box-shadow:0 8px 24px rgba(0,0,0,.32)}
      body.theme-dark .epsilon-toolbar-label{color:#cbd5e1}
      body.theme-dark .epsilon-toolbar-segments{background:#0b1220}
      body.theme-dark .epsilon-toolbar-segments button{color:#94a3b8}
      body.theme-dark .epsilon-toolbar-segments button.active{background:#1e293b;color:#93c5fd;box-shadow:none}
      body.theme-dark .epsilon-curve-preview:hover{background:#10213a;border-color:#3b82f6!important;box-shadow:0 0 0 1px rgba(59,130,246,.28),0 0 18px rgba(59,130,246,.18)}
      body.theme-dark .epsilon-overview-dialog{background:#0f172a;border-color:#263449;box-shadow:0 22px 58px rgba(0,0,0,.48)}
      body.theme-dark .epsilon-overview-header{background:#111c2f;border-bottom-color:#263449}
      body.theme-dark .epsilon-overview-title,
      body.theme-dark .epsilon-overview-body h3,
      body.theme-dark .epsilon-story-copy h4,
      body.theme-dark .epsilon-svg-title,
      body.theme-dark .epsilon-svg-equation,
      body.theme-dark .epsilon-metric-value{color:#e5edf7}
      body.theme-dark .epsilon-overview-nav{border-right-color:#263449}
      body.theme-dark .epsilon-overview-nav a{color:#94a3b8}
      body.theme-dark .epsilon-overview-nav a:hover{background:#111827;color:#e5edf7}
      body.theme-dark .epsilon-overview-nav a.active{background:#10213a;color:#93c5fd;border-left-color:#3b82f6}
      body.theme-dark .epsilon-overview-body,
      body.theme-dark .epsilon-overview-lead,
      body.theme-dark .epsilon-overview-definitions,
      body.theme-dark .epsilon-story-lead,
      body.theme-dark .epsilon-story-copy p,
      body.theme-dark .epsilon-svg-muted,
      body.theme-dark .epsilon-metric-label{color:#94a3b8}
      body.theme-dark .epsilon-inspector-context,
      body.theme-dark .epsilon-classification-kicker,
      body.theme-dark .epsilon-classification-subtitle{color:#94a3b8}
      body.theme-dark .epsilon-classification-main{color:#e5edf7}
      body.theme-dark .epsilon-streamflow-completeness{border-color:#263449}
      body.theme-dark .epsilon-signal-panel{background:#111827;border-color:#263449}
      body.theme-dark .epsilon-shift-panel,
      body.theme-dark .epsilon-attribution-panel,
      body.theme-dark .epsilon-trend-panel{border-color:#263449}
      body.theme-dark .epsilon-shift-row+.epsilon-shift-row{border-top-color:#263449}
      body.theme-dark .epsilon-shift-effect,
      body.theme-dark .epsilon-shift-ci{color:#e5edf7}
      body.theme-dark .epsilon-overview-body section + section{border-top-color:#263449}
      body.theme-dark .epsilon-overview-definition-title{color:#e5edf7}
      body.theme-dark .epsilon-overview-attribution{border-top-color:#263449}
      body.theme-dark .epsilon-evidence-item{background:#111827;border-color:#263449}
      body.theme-dark .epsilon-evidence-item--primary{border-top-color:#4ade80}
      body.theme-dark .epsilon-evidence-value,
      body.theme-dark .epsilon-evidence-association-value{color:#86efac}
      body.theme-dark .epsilon-evidence-item>strong,
      body.theme-dark .epsilon-evidence-association strong,
      body.theme-dark .epsilon-evidence-guardrail strong{color:#e5edf7}
      body.theme-dark .epsilon-evidence-kicker,
      body.theme-dark .epsilon-evidence-item p,
      body.theme-dark .epsilon-evidence-item small,
      body.theme-dark .epsilon-evidence-association p,
      body.theme-dark .epsilon-evidence-guardrail,
      body.theme-dark .epsilon-evidence-method{color:#94a3b8}
      body.theme-dark .epsilon-evidence-association{border-color:#263449}
      body.theme-dark .epsilon-method-facts>div,
      body.theme-dark .epsilon-driver-guide{border-color:#263449}
      body.theme-dark .epsilon-method-facts strong,
      body.theme-dark .epsilon-equation-card>span,
      body.theme-dark .epsilon-equation-card code,
      body.theme-dark .epsilon-driver-guide code,
      body.theme-dark .epsilon-driver-guide-grid strong,
      body.theme-dark .epsilon-method-caution strong{color:#e5edf7}
      body.theme-dark .epsilon-method-facts span,
      body.theme-dark .epsilon-driver-guide-grid span,
      body.theme-dark .epsilon-method-caution{color:#94a3b8}
      body.theme-dark .epsilon-equation-card{background:#111827;border-color:#263449}
      body.theme-dark .epsilon-attribution-panel,
      body.theme-dark .epsilon-trend-panel{background:transparent;border-color:#263449}
      body.theme-dark .epsilon-attribution-swatch,
      body.theme-dark .epsilon-driver-swatch{border-color:rgba(255,255,255,.28)}
      body.theme-dark .epsilon-continuous-key{background:#111827;border-color:#263449}
      body.theme-dark .epsilon-continuous-labels{color:#94a3b8}
      body.theme-dark .epsilon-attribution-title,
      body.theme-dark .epsilon-driver-legend-title,
      body.theme-dark .epsilon-attribution-driver,
      body.theme-dark .epsilon-trend-class{color:#e5edf7}
      body.theme-dark .epsilon-attribution-row,
      body.theme-dark .epsilon-attribution-regime,
      body.theme-dark .epsilon-attribution-values,
      body.theme-dark .epsilon-attribution-note,
      body.theme-dark .epsilon-driver-legend-item,
      body.theme-dark .epsilon-driver-legend-note,
      body.theme-dark .epsilon-trend-row,
      body.theme-dark .epsilon-trend-regime{color:#94a3b8}
      body.theme-dark .epsilon-trend-row+.epsilon-trend-row{border-top-color:#263449}
      body.theme-dark .epsilon-overview-close:hover{background:#1e293b;color:#f8fafc}
      body.theme-dark .epsilon-metric-card,
      body.theme-dark .epsilon-streamflow-completeness,
      body.theme-dark .epsilon-story-figure{background:#111827;border-color:#263449}
      body.theme-dark .epsilon-metric-value--skill{color:var(--epsilon-skill-dark)}
      body.theme-dark .epsilon-overview-filter{background:#111827;border-color:#263449}
      body.theme-dark .epsilon-filter-title{color:#e5edf7}
      body.theme-dark .epsilon-filter-field{color:#94a3b8}
      body.theme-dark .epsilon-filter-field select,
      body.theme-dark .epsilon-filter-field input{background:#0f172a;border-color:#334155;color:#e5edf7}
      body.theme-dark .epsilon-filter-count{color:#94a3b8}
      body.theme-dark .epsilon-field-mode-note{background:#0f172a;color:#94a3b8}
      body.theme-dark .epsilon-field-mode-note strong{color:#e5edf7}
      body.theme-dark .epsilon-streamflow-completeness,
      body.theme-dark .epsilon-completeness-title,
      body.theme-dark .epsilon-completeness-row span:last-child,
      body.theme-dark .epsilon-completeness-subtitle,
      body.theme-dark .epsilon-gap-list li,
      body.theme-dark .epsilon-gap-list li span:last-child,
      body.theme-dark .epsilon-completeness-note{color:#94a3b8}
      body.theme-dark .epsilon-completeness-subtitle{border-top-color:#263449}
      body.theme-dark .epsilon-story-index{box-shadow:0 0 0 5px rgba(15,23,42,.97)}
      body.theme-dark .epsilon-svg-box,
      body.theme-dark .epsilon-svg-formula{fill:#0f172a;stroke:#334155}
      body.theme-dark .epsilon-svg-pill{fill:#1e293b;stroke:#3b82f6}
      body.theme-dark .epsilon-svg-grid{stroke:#334155}
      body.theme-dark .epsilon-svg-arrow{stroke:#64748b}
      body.theme-dark .epsilon-svg-arrow-head{fill:#64748b}
      @media (max-width:760px){.epsilon-toolbar{top:auto;bottom:12px;left:12px;right:12px;transform:none;max-width:none;justify-content:center;padding:5px}.epsilon-toolbar-label{display:none}.epsilon-toolbar-segments{flex:1;display:grid;grid-template-columns:repeat(3,1fr)}.epsilon-toolbar-segments button,.epsilon-toolbar-overview{padding:0 6px}.epsilon-overview-body{padding-top:0}.epsilon-overview-layout{display:block}.epsilon-overview-nav{z-index:2;display:flex;gap:3px;margin:-2px 0 14px;padding:4px 0 8px;border-right:0;border-bottom:1px solid #e2e8f0;background:#fff;overflow-x:auto}.epsilon-overview-nav-title{display:none}.epsilon-overview-nav a{flex:0 0 auto;border-left:0;border-bottom:2px solid transparent;padding:6px 8px}.epsilon-overview-nav a.active{border-left:0;border-bottom-color:#2563eb}.epsilon-overview-content{padding-left:0}.epsilon-filter-grid{grid-template-columns:1fr}.epsilon-overview-metrics,.epsilon-method-facts,.epsilon-equation-grid,.epsilon-driver-guide-grid,.epsilon-evidence-grid{grid-template-columns:1fr}.epsilon-evidence-association{grid-template-columns:1fr}.epsilon-evidence-association-value{margin-top:4px}.epsilon-evidence-association p{grid-column:auto}.epsilon-equation-card:last-child{grid-column:auto}.epsilon-story-panel{grid-template-columns:1fr}.epsilon-overview-dialog{width:calc(100vw - 28px);max-height:calc(100vh - 28px)}body.theme-dark .epsilon-overview-nav{background:#0f172a;border-bottom-color:#263449}}
    `;
    document.head.appendChild(style);
  }

  ensureDistributionModal() {
    if (this.distributionModal) return;
    if (!document.getElementById("epsilon-distribution-styles")) {
      const style = document.createElement("style");
      style.id = "epsilon-distribution-styles";
      style.textContent = `
        .epsilon-modal{position:fixed;inset:0;z-index:420;display:none;align-items:center;justify-content:center;background:rgba(15,23,42,.36);padding:26px}
        .epsilon-modal.visible{display:flex}
        .epsilon-dialog{width:min(1060px,calc(100vw - 52px));height:min(760px,calc(100vh - 52px));background:#fff;border-radius:8px;box-shadow:0 22px 58px rgba(15,23,42,.28);display:flex;flex-direction:column;overflow:hidden}
        .epsilon-dialog-header{height:58px;padding:0 18px;border-bottom:1px solid #e2e8f0;display:flex;align-items:center;justify-content:space-between;gap:16px}
        .epsilon-dialog-title{font-size:15px;font-weight:700;color:#0f172a}
        .epsilon-dialog-subtitle{font-size:11px;color:#64748b;margin-top:3px}
        .epsilon-close{width:30px;height:30px;border:0;background:transparent;border-radius:4px;cursor:pointer;font-size:0;color:#64748b;line-height:1;position:relative;padding:0}
        .epsilon-close:hover{background:#f1f5f9;color:#0f172a}
        .epsilon-close::before,.epsilon-close::after{content:"";position:absolute;left:50%;top:50%;width:12px;height:1.5px;border-radius:999px;background:currentColor;transform-origin:center}
        .epsilon-close::before{transform:translate(-50%,-50%) rotate(45deg)}
        .epsilon-close::after{transform:translate(-50%,-50%) rotate(-45deg)}
        .epsilon-chart-area{flex:1;min-height:0;padding:14px 18px 18px;display:grid;grid-template-rows:repeat(3,1fr);gap:12px}
        .epsilon-chart-card{position:relative;border:1px solid #e2e8f0;border-radius:6px;background:#f8fafc;overflow:hidden}
        .epsilon-chart-card canvas{display:block;width:100%;height:100%}
        .epsilon-readout{position:absolute;right:42px;bottom:52px;width:138px;padding:7px 9px;border:1px solid #dbe3ef;border-radius:6px;background:rgba(255,255,255,.92);font-size:11px;color:#334155;line-height:1.42;box-shadow:0 8px 20px rgba(15,23,42,.08);pointer-events:none}
        .epsilon-readout:empty{display:none}
        .epsilon-readout strong{color:#0f172a}
        body.theme-dark .epsilon-dialog{background:#0f172a;box-shadow:0 22px 58px rgba(0,0,0,.52)}
        body.theme-dark .epsilon-dialog-header{border-bottom-color:#263449}
        body.theme-dark .epsilon-dialog-title{color:#e5edf7}
        body.theme-dark .epsilon-dialog-subtitle{color:#94a3b8}
        body.theme-dark .epsilon-close:hover{background:#1e293b;color:#f8fafc}
        body.theme-dark .epsilon-chart-card{background:#111827;border-color:#263449}
        body.theme-dark .epsilon-readout{background:rgba(15,23,42,.94);border-color:#334155;color:#cbd5e1;box-shadow:0 8px 20px rgba(0,0,0,.25)}
        body.theme-dark .epsilon-readout strong{color:#f8fafc}
      `;
      document.head.appendChild(style);
    }

    const modal = document.createElement("div");
    modal.className = "epsilon-modal";
    modal.id = "epsilon-distribution-modal";
    modal.innerHTML = `
      <div class="epsilon-dialog">
        <div class="epsilon-dialog-header">
          <div>
            <div class="epsilon-dialog-title" id="epsilon-modal-title">Epsilon distribution</div>
            <div class="epsilon-dialog-subtitle" id="epsilon-modal-subtitle">CDF</div>
          </div>
          <button class="epsilon-close" id="epsilon-modal-close" type="button" aria-label="Close"></button>
        </div>
        <div class="epsilon-chart-area">
          ${this.displayRegimes.map((regime) => `
            <div class="epsilon-chart-card cdf" data-regime="${regime}">
              <canvas id="epsilon-cdf-canvas-${regime}" data-regime="${regime}"></canvas>
              <div class="epsilon-readout" id="epsilon-cdf-readout-${regime}"></div>
            </div>
          `).join("")}
        </div>
      </div>
    `;
    document.body.appendChild(modal);
    modal.onclick = (event) => {
      if (event.target === modal) this.closeDistributionModal();
    };
    modal.querySelector("#epsilon-modal-close").onclick = () => this.closeDistributionModal();
    modal.querySelectorAll("canvas").forEach((canvas) => {
      canvas.addEventListener("mousemove", this.handleModalPointer);
      canvas.addEventListener("mouseleave", () => {
        if (!this.activeDistribution) return;
        const regime = canvas.dataset.regime || "all";
        delete this.activeDistribution.hover[regime];
        this.drawDistributionModal();
      });
    });
    this.distributionModal = modal;
  }

  openDistributionModal(gcin, regime) {
    const basin = this.byId.get(String(gcin));
    const curves = this.data.curves?.[String(gcin)] || {};
    if (!basin || !this.displayRegimes.some((name) => curves[name]?.x?.length)) return;
    this.ensureDistributionModal();
    this.activeDistribution = { basin, curves, hover: {} };
    if (regime && curves[regime]?.x?.length) {
      this.activeDistribution.initialRegime = regime;
    }
    this.distributionModal.querySelector("#epsilon-modal-title").textContent = `GCIN ${basin.GCIN} - CDF panels`;
    this.distributionModal.querySelector("#epsilon-modal-subtitle").textContent =
      "Pre 1950-1990 vs post 1991-2019; rows show all recession, low-flow, and high-flow epsilon distributions.";
    this.distributionModal.classList.add("visible");
    this.drawDistributionModal();
  }

  closeDistributionModal() {
    this.distributionModal?.classList.remove("visible");
    this.activeDistribution = null;
  }

  onDistributionPointer(event) {
    if (!this.activeDistribution) return;
    const canvas = event.currentTarget;
    const regime = canvas.dataset.regime || "all";
    const curve = this.activeDistribution.curves?.[regime];
    if (!curve?.x?.length) return;
    const rect = canvas.getBoundingClientRect();
    const plot = this.distributionPlot(rect.width, rect.height);
    const x = curve.x.map(Number);
    const minX = Math.min(...x);
    const maxX = Math.max(...x, minX + 1e-12);
    const px = event.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, (px - plot.left) / Math.max(1, plot.right - plot.left)));
    const target = minX + ratio * (maxX - minX);
    let closest = 0;
    let closestDistance = Infinity;
    for (let i = 0; i < x.length; i++) {
      const distance = Math.abs(x[i] - target);
      if (distance < closestDistance) {
        closest = i;
        closestDistance = distance;
      }
    }
    this.activeDistribution.hover[regime] = { index: closest, epsilon: target };
    this.drawDistributionModal();
  }

  drawDistributionModal() {
    if (!this.activeDistribution) return;
    for (const regime of this.displayRegimes) {
      this.drawDistributionCanvas(
        this.distributionModal.querySelector(`#epsilon-cdf-canvas-${regime}`),
        this.distributionModal.querySelector(`#epsilon-cdf-readout-${regime}`),
        regime
      );
    }
  }

  drawDistributionCanvas(canvas, readout, regime) {
    const { basin, curves, hover } = this.activeDistribution;
    const curve = curves?.[regime];
    if (!canvas || !readout || !curve?.x?.length) return;
    const hoverState = hover?.[regime] || {};
    const hoverIndex = hoverState.index;
    const hoverEpsilon = hoverState.epsilon;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const width = rect.width;
    const height = rect.height;
    const plot = this.distributionPlot(width, height);
    const x = curve.x.map(Number);
    const pre = curve.preCdf.map(Number);
    const post = curve.postCdf.map(Number);
    const minX = Math.min(...x);
    const maxX = Math.max(...x, minX + 1e-12);
    const maxY = 1;
    const theme = this.themeColors();
    const xAt = (value) => plot.left + ((value - minX) / Math.max(1e-12, maxX - minX)) * (plot.right - plot.left);
    const yAt = (value) => plot.bottom - (value / maxY) * (plot.bottom - plot.top);

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = theme.card;
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = theme.grid;
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = plot.top + (i / 4) * (plot.bottom - plot.top);
      ctx.beginPath();
      ctx.moveTo(plot.left, y);
      ctx.lineTo(plot.right, y);
      ctx.stroke();
      const value = maxY - (i / 4) * maxY;
      ctx.fillStyle = theme.tick;
      ctx.font = "10px sans-serif";
      ctx.textAlign = "right";
      ctx.fillText(this.formatSmall(value), plot.left - 8, y + 3);
    }
    for (let i = 0; i <= 4; i++) {
      const xx = plot.left + (i / 4) * (plot.right - plot.left);
      ctx.beginPath();
      ctx.moveTo(xx, plot.top);
      ctx.lineTo(xx, plot.bottom);
      ctx.stroke();
      const value = minX + (i / 4) * (maxX - minX);
      ctx.fillStyle = theme.tick;
      ctx.font = "10px sans-serif";
      ctx.textAlign = i === 0 ? "left" : i === 4 ? "right" : "center";
      ctx.fillText(this.formatSmall(value), xx, plot.bottom + 16);
    }

    ctx.strokeStyle = theme.axis;
    ctx.beginPath();
    ctx.moveTo(plot.left, plot.top);
    ctx.lineTo(plot.left, plot.bottom);
    ctx.lineTo(plot.right, plot.bottom);
    ctx.stroke();

    this.drawLine(ctx, x, pre, xAt, yAt, "#2563eb");
    this.drawLine(ctx, x, post, xAt, yAt, "#b84235");

    ctx.fillStyle = theme.text;
    ctx.font = "13px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(`${this.regimeShortLabel(regime)} CDF`, plot.left, 18);
    ctx.fillStyle = theme.muted;
    ctx.font = "11px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("Pre", plot.right - 82, 18);
    ctx.fillText("Post", plot.right - 34, 18);
    ctx.fillStyle = "#2563eb";
    ctx.fillRect(plot.right - 102, 11, 14, 3);
    ctx.fillStyle = "#b84235";
    ctx.fillRect(plot.right - 56, 11, 14, 3);
    ctx.textAlign = "right";
    ctx.fillStyle = theme.muted;
    ctx.fillText("epsilon", plot.right, height - 8);

    if (hoverIndex == null) {
      readout.innerHTML = "";
    } else {
      const epsilon = Number.isFinite(Number(hoverEpsilon)) ? Number(hoverEpsilon) : x[hoverIndex];
      const px = xAt(epsilon);
      const preValue = this.interpolateCurve(x, pre, epsilon);
      const postValue = this.interpolateCurve(x, post, epsilon);
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = theme.cursor;
      ctx.beginPath();
      ctx.moveTo(px, plot.top);
      ctx.lineTo(px, plot.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      for (const [value, color] of [[preValue, "#2563eb"], [postValue, "#b84235"]]) {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(px, yAt(value), 4, 0, Math.PI * 2);
        ctx.fill();
      }
      const delta = Number(basin[`${regime}_relative_delta_pct`]);
      readout.innerHTML = `
        <strong>${this.regimeShortLabel(regime)} CDF</strong><br>
        epsilon: ${this.formatSmall(epsilon)}<br>
        pre: ${this.formatSmall(preValue)}<br>
        post: ${this.formatSmall(postValue)}<br>
        raw daily-mean delta: ${this.formatPct(delta)}
      `;
    }
  }

  drawLine(ctx, x, values, xAt, yAt, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    x.forEach((value, index) => {
      const px = xAt(value);
      const py = yAt(values[index] || 0);
      if (index === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
  }

  interpolateCurve(x, values, target) {
    if (!x.length) return 0;
    if (target <= x[0]) return Number(values[0] || 0);
    const last = x.length - 1;
    if (target >= x[last]) return Number(values[last] || 0);
    for (let i = 1; i < x.length; i++) {
      if (target <= x[i]) {
        const x0 = x[i - 1];
        const x1 = x[i];
        const y0 = Number(values[i - 1] || 0);
        const y1 = Number(values[i] || 0);
        const ratio = x1 === x0 ? 0 : (target - x0) / (x1 - x0);
        return y0 + ratio * (y1 - y0);
      }
    }
    return Number(values[last] || 0);
  }

  distributionPlot(width, height) {
    return {
      left: 64,
      right: width - 32,
      top: 34,
      bottom: height - 46
    };
  }

  ensureLegend() {
    if (this.analysisView === "decomposition") {
      this.app.registerLegend?.(this.legendId, {
        title: `${this.focusTitle()} GQ / Q decomposition`,
        html: this.renderDriverLegend(true)
      });
      return;
    }
    this.app.registerLegend?.(this.legendId, {
      title: this.analysisView === "trend"
        ? `${this.focusTitle()} epsilon slope`
        : `${this.focusTitle()} epsilon era change`,
      html: `
        ${this.renderContinuousLegendBar()}
        <div class="epsilon-map-legend-note">${this.analysisView === "trend"
          ? "Fold-centered Theil-Sen slope in percent per decade. FDR evidence is shown in the inspector."
          : "Fold-adjusted post-1990 annual-median effect. FDR evidence is shown in the inspector; gray means near zero, not Unresolved."}</div>
      `
    });
  }

  colorFor(value, maxAbs) {
    if (!Number.isFinite(value)) return "#cbd5e1";
    const t = Math.max(-1, Math.min(1, value / maxAbs));
    if (t < 0) return this.mix("#2563eb", "#f1e8c9", t + 1);
    return this.mix("#f1e8c9", "#b84235", t);
  }

  mix(a, b, t) {
    const ca = this.hex(a);
    const cb = this.hex(b);
    const c = ca.map((value, index) => Math.round(value + (cb[index] - value) * t));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }

  hex(value) {
    return [1, 3, 5].map((index) => parseInt(value.slice(index, index + 2), 16));
  }

  mean(values) {
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : NaN;
  }

  isDarkMode() {
    return document.body.classList.contains("theme-dark");
  }

  themeColors() {
    if (this.isDarkMode()) {
      return {
        card: "#111827",
        grid: "#263449",
        axis: "#334155",
        text: "#e5edf7",
        muted: "#94a3b8",
        tick: "#8aa0ba",
        cursor: "#cbd5e1"
      };
    }
    return {
      card: "#f8fafc",
      grid: "#e2e8f0",
      axis: "#cbd5e1",
      text: "#0f172a",
      muted: "#64748b",
      tick: "#94a3b8",
      cursor: "#475569"
    };
  }

  median(values) {
    if (!values.length) return NaN;
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }

  formatPct(value) {
    const number = Number(value);
    return Number.isFinite(number) ? `${number.toFixed(1)}%` : "NA";
  }

  formatSignedPct(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "NA";
    const rounded = Math.abs(number) < 0.05 ? 0 : number;
    return `${rounded > 0 ? "+" : ""}${rounded.toFixed(1)}%`;
  }

  formatPValue(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "NA";
    return number >= 0.001 ? number.toFixed(3) : number.toExponential(2);
  }

  formatSigned(value, digits = 3) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "NA";
    const rounded = Math.abs(number) < 0.5 * 10 ** (-digits) ? 0 : number;
    return `${rounded > 0 ? "+" : ""}${rounded.toFixed(digits)}`;
  }

  formatNumber(value, digits = 2) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(digits) : "NA";
  }

  formatInteger(value) {
    const number = Number(value);
    return Number.isFinite(number) ? Math.round(number).toLocaleString("en-US") : "";
  }

  formatSmall(value) {
    if (typeof value === "string") return value;
    const number = Number(value);
    if (!Number.isFinite(number)) return "NA";
    const abs = Math.abs(number);
    if (abs >= 1000) return number.toFixed(0);
    if (abs >= 1) return number.toFixed(2);
    if (abs >= 0.01) return number.toFixed(3);
    return number.toExponential(2);
  }

  regimeLabel(regime) {
    return {
      all: "All recession days",
      low: "Low flow (Q <= Q10)",
      high: "High flow (Q >= Q90)"
    }[regime] || regime;
  }

  regimeShortLabel(regime) {
    return {
      all: "All recession",
      low: "Low flow",
      high: "High flow"
    }[regime] || this.regimeLabel(regime);
  }

  escape(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }
};
