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
    this.basins = [];
    this.byId = new Map();
    this.selected = null;
    this.viewMode = manifest.viewMode || "bivariate";
    this.focusRegime = manifest.focusRegime || null;
    this.dataFile = manifest.dataFile || manifest.datasets?.[0]?.file || "./data/epsilon-catchment-distributions.json";
    this.layerId = `${manifest.id || "epsilon-change"}-catchments`;
    this.overviewLayerId = `${manifest.id || "epsilon-change"}-overview`;
    this.legendId = `${manifest.id || "epsilon-change"}-legend`;
    this.overviewModal = null;
    this.distributionModal = null;
    this.activeDistribution = null;
    this.stableThresholdPct = 5;
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
  }

  async onLoad() {
    this.data = await this.fetchJson(this.resolve(this.dataFile));
    this.basins = (this.data.basins || [])
      .filter((basin) => Number.isFinite(Number(basin.lon)) && Number.isFinite(Number(basin.lat)))
      .map((basin) => ({
        ...basin,
        id: String(basin.GCIN),
        lon: Number(basin.lon),
        lat: Number(basin.lat),
        area_km2: Number(basin.area_km2 || 0),
        low_change_state: this.changeState(basin.low_relative_delta_pct),
        high_change_state: this.changeState(basin.high_relative_delta_pct)
      }));
    this.colorScaleExtent = this.computeContinuousExtent();
    this.byId = new Map(this.basins.map((basin) => [basin.id, basin]));
    this.addLayer();
    this.ensurePreviewStyles();
    this.ensureLegend();
    this.showOverview();
    Foundation.eventBus.on(Foundation.Events.FEATURE_CLICK, this.handleFeatureClick);
    Foundation.eventBus.on(Foundation.Events.LAYER_TOGGLE, this.handleLayerToggle);
    this.app.draw?.();
  }

  onUnload() {
    this.app.layerManager.removeLayer(this.layerId);
    this.app.layerManager.removeLayer(this.overviewLayerId);
    this.app.unregisterLegend?.(this.legendId);
    Foundation.eventBus.off(Foundation.Events.FEATURE_CLICK, this.handleFeatureClick);
    Foundation.eventBus.off(Foundation.Events.LAYER_TOGGLE, this.handleLayerToggle);
    this.selected = null;
    this.destroyModals();
  }

  getLayerIds() {
    return [this.layerId, this.overviewLayerId];
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
    const all = this.basins.map((basin) => Number(basin.all_relative_delta_pct)).filter(Number.isFinite);
    const low = this.basins.map((basin) => Number(basin.low_relative_delta_pct)).filter(Number.isFinite);
    const high = this.basins.map((basin) => Number(basin.high_relative_delta_pct)).filter(Number.isFinite);
    const counts = this.categoryCounts();
    const layer = this.app.layerManager.getLayer?.(this.overviewLayerId);
    if (layer && !layer.visible) return;
    this.ensureOverviewModal();
    this.overviewModal.querySelector(".epsilon-overview-body").innerHTML = `
      <section>
        <p class="epsilon-overview-lead">${this.escape(this.overviewText())}</p>
        <div class="epsilon-overview-metrics">
          ${this.metricCard("Catchments", this.basins.length.toLocaleString())}
          ${this.metricCard("All-recession mean", this.formatPct(this.mean(all)), this.mean(all))}
          ${this.metricCard("Low-flow mean", this.formatPct(this.mean(low)), this.mean(low))}
          ${this.metricCard("High-flow mean", this.formatPct(this.mean(high)), this.mean(high))}
        </div>
        ${this.renderOverviewLegend(counts)}
        <p class="epsilon-overview-note">${this.escape(this.legendNote())}</p>
      </section>
      <section>
        <h3>Data and inference</h3>
        <p>
          The map summarizes catchment-level daily epsilon, defined as the inferred daily GQ/Q ratio. Streamflow comes from the matched legacy Event_Typology forcing records. Meteorological forcing and land-state variables come from ERA5-Land daily catchment reductions, so each row is a catchment-day time series record rather than a gridded raster.
        </p>
        <p>
          Pre-change is 1950-1990 and post-change is 1991-2019. Low-flow and high-flow regimes are defined within each catchment using its own Q10 and Q90 streamflow thresholds.
        </p>
        <p>
          Epsilon was inferred with the Ara-style physics-informed LSTM-epsilon workflow. The model directly predicts epsilon inside the recession differential equation; it does not first predict GQ and then divide by Q.
        </p>
      </section>
      ${this.renderWorkflowDiagram()}
      ${this.renderMethodDetails()}
    `;
    this.overviewModal.classList.add("visible");
  }

  closeOverview() {
    this.overviewModal?.classList.remove("visible");
  }

  destroyModals() {
    this.activeDistribution = null;
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

  renderWorkflowDiagram() {
    const steps = [
      ["Legacy streamflow", "Observed Q by matched catchment"],
      ["ERA5-Land forcing", "Daily catchment precipitation, temperature, PET, soil moisture"],
      ["Recession filter", "Declining-flow sequences, first day dropped, cold days removed"],
      ["LSTM-epsilon", "365-day context plus static attributes"],
      ["Physics solver", "AET term and recession equation constrain Q path"],
      ["Epsilon contrast", "Pre/post CDFs, Q10 low flow, Q90 high flow"]
    ];
    return `
      <section>
        <h3>Workflow</h3>
        <div class="epsilon-workflow">
          ${steps.map(([title, text], index) => `
            <div class="epsilon-workflow-step">
              <div class="epsilon-workflow-index">${index + 1}</div>
              <div>
                <div class="epsilon-workflow-title">${this.escape(title)}</div>
                <div class="epsilon-workflow-text">${this.escape(text)}</div>
              </div>
            </div>
          `).join("")}
        </div>
      </section>
    `;
  }

  renderMethodDetails() {
    return `
      <section>
        <h3>Model and analysis details</h3>
        <div class="epsilon-method-grid">
          <div>
            <h4>Inputs</h4>
            <p>Daily dynamic inputs are precipitation, temperature, PET and root-zone soil moisture. Static catchment attributes include location, area, long-term precipitation, temperature, PET, AET, aridity, soil moisture summaries, radiation and precipitation-event frequencies.</p>
          </div>
          <div>
            <h4>Recession mask</h4>
            <p>Training and inference are restricted to recession days. A valid recession has at least four declining days, the first decline day is dropped, a decreasing-rate filter is applied, and days with mean temperature at or below 0 C are removed as a snow proxy.</p>
          </div>
          <div>
            <h4>Physics-informed LSTM</h4>
            <p>The model reads a 365-day context window and outputs daily epsilon, q_base, alpha, LP and gamma. AET is computed inside the model from PET and soil moisture using bounded LP/gamma parameters.</p>
          </div>
          <div>
            <h4>Recession equation</h4>
            <p><code>dQ/dt = -epsilon * Q^2 - epsilon * alpha * AET * Q</code>. The solved recession streamflow path is supervised against observed streamflow; epsilon is therefore inferred directly inside the physical equation.</p>
          </div>
          <div>
            <h4>Training run</h4>
            <p>The production run uses five cross-fitted folds, 150 epochs, batch size 256, hidden size 256, one LSTM layer, dropout 0.4, sequence length 365 and learning rate 1e-4.</p>
          </div>
          <div>
            <h4>Loss</h4>
            <p>The objective combines path error, right-hand-side tendency consistency, epsilon smoothness and reset-flow alignment: <code>L = 25 L_path + 10 L_rhs + 0.1 L_smooth + 5 L_q0</code>.</p>
          </div>
          <div>
            <h4>Flow regimes</h4>
            <p>Low flow is Qobs <= each catchment's recession-day Q10. High flow is Qobs >= each catchment's recession-day Q90. These thresholds are catchment-specific, not pooled globally.</p>
          </div>
          <div>
            <h4>Displayed result</h4>
            <p>The map shows 1,149 catchments with cross-fitted daily recession epsilon summaries. The bivariate class combines low-flow and high-flow relative epsilon change; stable means within +/-${this.stableThresholdPct}%.</p>
          </div>
        </div>
      </section>
    `;
  }

  showInspector(basin) {
    const title = `GCIN ${basin.GCIN}`;
    const curves = this.data.curves?.[String(basin.GCIN)] || {};
    const sourceId = Number.isFinite(Number(basin.force_code))
      ? `<div style="margin:-4px 0 12px;font-size:11px;color:#64748b">Legacy force code: <strong style="color:#334155">${this.escape(basin.force_code)}</strong></div>`
      : "";
    const cards = `
      <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-bottom:14px">
        ${this.metricCard("Area", `${this.formatNumber(basin.area_km2, 1)} km2`)}
        ${this.metricCard("Aridity", this.formatNumber(basin.Aridity, 3))}
        ${this.metricCard("Precip.", `${this.formatNumber(basin.Prec_mm, 1)} mm`)}
        ${this.metricCard("Temp.", `${this.formatNumber(basin.Temp_C, 1)} C`)}
      </div>
      ${this.categoryBanner(basin)}
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:16px">
        ${this.metricCard("All", this.formatPct(basin.all_relative_delta_pct), basin.all_relative_delta_pct)}
        ${this.metricCard("Low", this.formatPct(basin.low_relative_delta_pct), basin.low_relative_delta_pct)}
        ${this.metricCard("High", this.formatPct(basin.high_relative_delta_pct), basin.high_relative_delta_pct)}
      </div>
    `;

    const cdfPreview = `
      <div
        class="epsilon-curve-preview"
        data-gcin="${this.escape(basin.GCIN)}"
        data-regime="all"
        aria-label="Open CDF panels"
        style="display:block;margin:2px 0 16px;cursor:pointer"
      >
        ${this.renderCombinedCdfSvg(curves)}
      </div>
    `;

    const sections = this.displayRegimes.map((regime) => `
      <section style="margin-top:10px;padding-top:12px;border-top:1px solid #e2e8f0">
        <h3 style="margin:0 0 8px;font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#64748b">${this.regimeLabel(regime)}</h3>
        ${this.renderStatsTable(basin, regime)}
      </section>
    `).join("");

    this.app.showInspector?.(title, `
      <p style="margin:0 0 14px;color:#64748b;font-size:12px;line-height:1.6">
        Epsilon is the modeled daily ratio GQ/Q. This panel compares the inferred epsilon distribution before and after 1990.
      </p>
      ${sourceId}
      ${cards}
      ${cdfPreview}
      ${sections}
    `);
    this.bindCurvePreviews();
  }

  metricCard(label, value, signedValue = null) {
    const color = Number.isFinite(Number(signedValue))
      ? Number(signedValue) < 0 ? "#2563eb" : "#b84235"
      : "#0f172a";
    return `
      <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px">
        <div style="font-size:17px;font-weight:700;color:${color}">${this.escape(value)}</div>
        <div style="font-size:11px;color:#64748b;margin-top:3px">${this.escape(label)}</div>
      </div>
    `;
  }

  categoryBanner(basin) {
    const color = this.basinColor(basin);
    return `
      <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px;margin-bottom:10px">
        <div style="display:flex;align-items:center;gap:8px;font-size:13px;font-weight:700;color:#0f172a;line-height:1.35">
          <span style="width:15px;height:15px;border-radius:50%;background:${color};border:1px solid rgba(15,23,42,.24);flex:0 0 auto"></span>
          <span>${this.escape(this.basinLabel(basin))}</span>
        </div>
        <div style="font-size:11px;color:#64748b;margin-top:5px">${this.escape(this.basinLabelSubtitle())}</div>
      </div>
    `;
  }

  renderOverviewLegend(counts) {
    if (this.viewMode === "low" || this.viewMode === "high") {
      return this.renderContinuousOverviewLegend();
    }
    return this.renderCategoryMatrix(counts);
  }

  renderContinuousOverviewLegend() {
    const regime = this.focusRegime || this.viewMode;
    const values = this.basins
      .map((basin) => Number(basin[`${regime}_relative_delta_pct`]))
      .filter(Number.isFinite);
    const mean = this.mean(values);
    const median = this.median(values);
    const negativeShare = values.length
      ? values.filter((value) => value < 0).length / values.length * 100
      : NaN;
    return `
      <div style="margin:0 0 14px">
        <div style="font-size:12px;font-weight:700;color:#0f172a;margin-bottom:8px">${this.escape(this.focusTitle())} relative epsilon change</div>
        ${this.renderContinuousLegendBar()}
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:10px">
          ${this.metricCard("Mean", this.formatPct(mean), mean)}
          ${this.metricCard("Median", this.formatPct(median), median)}
          ${this.metricCard("Decrease share", this.formatPct(negativeShare), -negativeShare)}
        </div>
      </div>
    `;
  }

  renderCategoryMatrix(counts) {
    return `
      <div style="margin:0 0 14px">
        <div style="font-size:12px;font-weight:700;color:#0f172a;margin-bottom:8px">Low-flow x high-flow classes</div>
        ${this.renderBivariateMatrix(counts, false)}
        <div style="font-size:11px;color:#64748b;margin-top:7px">${counts.insufficient || 0} catchments have insufficient low/high information for this bivariate class.</div>
      </div>
    `;
  }

  renderBivariateMatrix(counts, compact = true) {
    const states = ["decrease", "stable", "increase"];
    const cell = compact ? 48 : 72;
    const gap = compact ? 5 : 7;
    const rowHeight = compact ? 28 : 38;
    const fontSize = compact ? 10 : 12;
    const labelLeft = compact ? -38 : -58;
    const labelWidth = compact ? 32 : 50;
    const headerPrefix = compact ? "H " : "High ";
    const rowPrefix = compact ? "L " : "Low ";
    const matrixWidth = cell * 3 + gap * 2;
    return `
      <div style="position:relative;width:${matrixWidth}px;margin:0 auto;font-size:${compact ? 9 : 11}px;color:#64748b">
        <div style="display:grid;grid-template-columns:repeat(3,${cell}px);gap:${gap}px;margin-left:0;margin-bottom:${gap}px">
          ${states.map((state) => `<div style="text-align:center">${headerPrefix}${this.stateShortLabel(state)}</div>`).join("")}
        </div>
        <div style="display:grid;grid-template-columns:repeat(3,${cell}px);gap:${gap}px">
          ${states.map((low, rowIndex) => `
            <div style="position:absolute;left:${labelLeft}px;top:${(compact ? 17 : 22) + rowIndex * (rowHeight + gap)}px;width:${labelWidth}px;text-align:right">${rowPrefix}${this.stateShortLabel(low)}</div>
            ${states.map((high) => `
              <div title="low ${this.stateLabel(low)} / high ${this.stateLabel(high)}" style="height:${rowHeight}px;border-radius:4px;background:${this.categoryColorByStates(low, high)};border:1px solid rgba(15,23,42,.16);display:flex;align-items:center;justify-content:center;color:${this.categoryTextColor(low, high)};font-weight:700;font-size:${fontSize}px">
                ${counts[`${low}_${high}`] || 0}
              </div>
            `).join("")}
          `).join("")}
        </div>
      </div>
    `;
  }

  categoryCounts() {
    const counts = { insufficient: 0 };
    for (const basin of this.basins) {
      const low = basin.low_change_state;
      const high = basin.high_change_state;
      if (!low || !high) {
        counts.insufficient += 1;
      } else {
        const key = `${low}_${high}`;
        counts[key] = (counts[key] || 0) + 1;
      }
    }
    return counts;
  }

  changeState(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return null;
    if (number < -this.stableThresholdPct) return "decrease";
    if (number > this.stableThresholdPct) return "increase";
    return "stable";
  }

  layerName() {
    if (this.viewMode === "low") return "Low-flow epsilon change";
    if (this.viewMode === "high") return "High-flow epsilon change";
    return "Catchment epsilon change";
  }

  moduleTitle() {
    if (this.viewMode === "low") return "Epsilon Change (Low Flow)";
    if (this.viewMode === "high") return "Epsilon Change (High Flow)";
    return "Epsilon Change";
  }

  focusTitle() {
    if (this.focusRegime === "low" || this.viewMode === "low") return "Low-flow";
    if (this.focusRegime === "high" || this.viewMode === "high") return "High-flow";
    return "All-recession";
  }

  overviewText() {
    if (this.viewMode === "low") {
      return "Cross-fitted daily epsilon inference summarized by catchment. Points are catchment centroids colored by continuous low-flow relative epsilon change after 1990.";
    }
    if (this.viewMode === "high") {
      return "Cross-fitted daily epsilon inference summarized by catchment. Points are catchment centroids colored by continuous high-flow relative epsilon change after 1990.";
    }
    return "Cross-fitted daily epsilon inference summarized by catchment. Points are catchment centroids; color classifies each catchment by low-flow and high-flow epsilon change after 1990.";
  }

  legendNote() {
    if (this.viewMode === "low") {
      return "Low-flow epsilon uses recession days with Q_obs at or below each catchment's Q10. Cyan-blue indicates lower post-1990 epsilon, neutral gray indicates little change, and magenta indicates higher post-1990 epsilon.";
    }
    if (this.viewMode === "high") {
      return "High-flow epsilon uses recession days with Q_obs at or above each catchment's Q90. Cyan-blue indicates lower post-1990 epsilon, neutral gray indicates little change, and magenta indicates higher post-1990 epsilon.";
    }
    return `Stable means relative epsilon change within +/-${this.stableThresholdPct}%. Low-flow epsilon uses recession days with Q_obs at or below each catchment's Q10; high-flow uses days at or above Q90.`;
  }

  categoryLabel(basin) {
    const low = basin.low_change_state;
    const high = basin.high_change_state;
    if (!low || !high) return "insufficient";
    return `Low-flow ${this.stateLabel(low)} / high-flow ${this.stateLabel(high)}`;
  }

  basinLabel(basin) {
    if (this.viewMode === "low" || this.viewMode === "high") {
      const regime = this.focusRegime || this.viewMode;
      const value = Number(basin[`${regime}_relative_delta_pct`]);
      return `${this.focusTitle()} epsilon change: ${this.formatPct(value)}`;
    }
    return this.categoryLabel(basin);
  }

  basinLabelSubtitle() {
    if (this.viewMode === "low" || this.viewMode === "high") {
      return "Continuous color scale from post-1990 relative epsilon change.";
    }
    return "Bivariate class from low-flow and high-flow relative epsilon change.";
  }

  stateLabel(state) {
    return {
      decrease: "lower",
      stable: "stable",
      increase: "higher"
    }[state] || "insufficient";
  }

  stateShortLabel(state) {
    return {
      decrease: "low",
      stable: "stb",
      increase: "high"
    }[state] || "NA";
  }

  basinColor(basin) {
    if (this.viewMode === "low" || this.viewMode === "high") {
      const regime = this.focusRegime || this.viewMode;
      return this.continuousColor(Number(basin[`${regime}_relative_delta_pct`]));
    }
    return this.categoryColorByStates(basin.low_change_state, basin.high_change_state);
  }

  computeContinuousExtent() {
    const regime = this.focusRegime || this.viewMode;
    if (!(regime === "low" || regime === "high")) return 50;
    const values = this.basins
      .map((basin) => Number(basin[`${regime}_relative_delta_pct`]))
      .filter(Number.isFinite)
      .map(Math.abs)
      .sort((a, b) => a - b);
    if (!values.length) return 50;
    const p95 = values[Math.min(values.length - 1, Math.floor(values.length * 0.95))];
    return Math.max(10, Math.min(80, p95 || 50));
  }

  continuousColor(value) {
    if (!Number.isFinite(value)) return "#d8dee8";
    const extent = this.colorScaleExtent || 50;
    const t = Math.max(-1, Math.min(1, value / extent));
    if (Math.abs(t) < 0.02) return "#cbd5e1";
    if (t < 0) return this.mix("#00d7ff", "#cbd5e1", t + 1);
    return this.mix("#cbd5e1", "#ff3bbd", t);
  }

  renderContinuousLegendBar() {
    const extent = this.colorScaleExtent || 50;
    return `
      <div style="border:1px solid #e2e8f0;border-radius:6px;background:#f8fafc;padding:10px">
        <div style="height:12px;border-radius:999px;background:linear-gradient(90deg,#00d7ff 0%,#cbd5e1 50%,#ff3bbd 100%);border:1px solid rgba(15,23,42,.12)"></div>
        <div style="display:flex;justify-content:space-between;font-size:10px;color:#64748b;margin-top:6px">
          <span>${this.formatPct(-extent)}</span>
          <span>0%</span>
          <span>${this.formatPct(extent)}</span>
        </div>
      </div>
    `;
  }

  categoryColorByStates(low, high) {
    const colors = {
      decrease_decrease: "#1e3a8a",
      decrease_stable: "#3b82f6",
      decrease_increase: "#22c1d6",
      stable_decrease: "#64748b",
      stable_stable: "#cbd5e1",
      stable_increase: "#d97706",
      increase_decrease: "#7c3aed",
      increase_stable: "#ef4444",
      increase_increase: "#7f1d1d"
    };
    return colors[`${low}_${high}`] || "#d8dee8";
  }

  categoryTextColor(low, high) {
    return low === "stable" && high === "stable" ? "#334155" : "#ffffff";
  }

  renderStatsTable(basin, regime) {
    const rows = [
      ["Mean", basin[`${regime}_pre_mean`], basin[`${regime}_post_mean`]],
      ["IQR", `${this.formatSmall(basin[`${regime}_pre_q25`])}-${this.formatSmall(basin[`${regime}_pre_q75`])}`, `${this.formatSmall(basin[`${regime}_post_q25`])}-${this.formatSmall(basin[`${regime}_post_q75`])}`],
      ["N", basin[`${regime}_pre_n`], basin[`${regime}_post_n`]]
    ];
    return `
      <table style="width:100%;border-collapse:collapse;font-size:11px">
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
    const width = 300;
    const rowHeight = 86;
    const height = rowHeight * this.displayRegimes.length + 18;
    const margin = { left: 34, right: 10, top: 20, bottom: 18 };
    const rows = this.displayRegimes.map((regime, rowIndex) => {
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
        <line x1="${margin.left}" y1="${plotBottom}" x2="${width - margin.right}" y2="${plotBottom}" stroke="#dbe3ef"/>
        <line x1="${margin.left}" y1="${plotTop}" x2="${margin.left}" y2="${plotBottom}" stroke="#dbe3ef"/>
        <text x="${margin.left}" y="${plotTop - 5}" fill="#0f172a" font-size="10" font-weight="700">${this.regimeShortLabel(regime)}</text>
        <text x="${width - margin.right}" y="${plotTop - 5}" fill="#64748b" font-size="9" text-anchor="end">${this.formatSmall(minX)}-${this.formatSmall(maxX)}</text>
        <path d="${path(pre)}" fill="none" stroke="#2563eb" stroke-width="1.6"/>
        <path d="${path(post)}" fill="none" stroke="#b84235" stroke-width="1.6"/>
      `;
    }).join("");
    return `
      <svg viewBox="0 0 ${width} ${height}" style="display:block;width:100%;height:auto;background:#f8fafc;pointer-events:none">
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
      .epsilon-curve-preview{box-sizing:border-box;border:1px solid #e2e8f0;border-radius:6px;overflow:hidden;background:#f8fafc;transition:border-color .16s ease,box-shadow .16s ease}
      .epsilon-curve-preview:hover{border-color:#60a5fa!important;box-shadow:0 0 0 1px rgba(96,165,250,.28)}
      .epsilon-overview-modal{position:fixed;inset:0;display:none;align-items:center;justify-content:center;z-index:150;pointer-events:none}
      .epsilon-overview-modal.visible{display:flex}
      .epsilon-overview-dialog{width:min(820px,calc(100vw - 64px));max-height:min(760px,calc(100vh - 64px));background:rgba(255,255,255,.96);border:1px solid #dbe3ef;border-radius:8px;box-shadow:0 22px 58px rgba(15,23,42,.24);display:flex;flex-direction:column;overflow:hidden;pointer-events:auto}
      .epsilon-overview-header{height:58px;padding:0 18px;border-bottom:1px solid #e2e8f0;display:flex;align-items:center;justify-content:space-between;gap:16px}
      .epsilon-overview-title{font-size:18px;font-weight:750;color:#0f172a;letter-spacing:0}
      .epsilon-overview-close{width:32px;height:32px;border:0;background:transparent;color:#64748b;font-size:0;line-height:1;cursor:pointer;border-radius:6px;position:relative;padding:0}
      .epsilon-overview-close:hover{background:#eef2f7;color:#0f172a}
      .epsilon-overview-close::before,.epsilon-overview-close::after{content:"";position:absolute;left:50%;top:50%;width:12px;height:1.5px;border-radius:999px;background:currentColor;transform-origin:center}
      .epsilon-overview-close::before{transform:translate(-50%,-50%) rotate(45deg)}
      .epsilon-overview-close::after{transform:translate(-50%,-50%) rotate(-45deg)}
      .epsilon-overview-body{padding:18px;overflow:auto;color:#334155;font-size:13px;line-height:1.65}
      .epsilon-overview-body section + section{margin-top:18px;padding-top:16px;border-top:1px solid #e2e8f0}
      .epsilon-overview-body h3{margin:0 0 8px;font-size:13px;color:#0f172a;letter-spacing:.03em;text-transform:uppercase}
      .epsilon-overview-body p{margin:0 0 10px}
      .epsilon-overview-lead{color:#475569}
      .epsilon-overview-note{font-size:12px;color:#475569;margin-top:10px}
      .epsilon-overview-metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:16px 0}
      .epsilon-workflow{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}
      .epsilon-workflow-step{position:relative;min-height:96px;border:1px solid #dbe3ef;border-radius:6px;background:#f8fafc;padding:12px 12px 12px 42px}
      .epsilon-workflow-step::after{content:"";position:absolute;right:-10px;top:50%;width:10px;height:1px;background:#cbd5e1}
      .epsilon-workflow-step:nth-child(3)::after,.epsilon-workflow-step:nth-child(6)::after{display:none}
      .epsilon-workflow-index{position:absolute;left:12px;top:12px;width:20px;height:20px;border-radius:50%;background:#1d4ed8;color:white;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700}
      .epsilon-workflow-title{font-size:12px;font-weight:750;color:#0f172a;margin-bottom:5px}
      .epsilon-workflow-text{font-size:11px;color:#64748b;line-height:1.45}
      .epsilon-method-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}
      .epsilon-method-grid>div{border:1px solid #dbe3ef;border-radius:6px;background:#f8fafc;padding:11px}
      .epsilon-method-grid h4{margin:0 0 5px;font-size:12px;color:#0f172a}
      .epsilon-method-grid p{margin:0;font-size:11.5px;line-height:1.55;color:#475569}
      .epsilon-method-grid code{font-family:Consolas,monospace;font-size:11px;color:#0f172a;background:#eef2f7;border-radius:3px;padding:1px 3px}
      @media (max-width:760px){.epsilon-overview-metrics,.epsilon-method-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.epsilon-workflow{grid-template-columns:1fr}.epsilon-workflow-step::after{display:none}.epsilon-overview-dialog{width:calc(100vw - 28px);max-height:calc(100vh - 28px)}}
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
    const xAt = (value) => plot.left + ((value - minX) / Math.max(1e-12, maxX - minX)) * (plot.right - plot.left);
    const yAt = (value) => plot.bottom - (value / maxY) * (plot.bottom - plot.top);

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#f8fafc";
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = plot.top + (i / 4) * (plot.bottom - plot.top);
      ctx.beginPath();
      ctx.moveTo(plot.left, y);
      ctx.lineTo(plot.right, y);
      ctx.stroke();
      const value = maxY - (i / 4) * maxY;
      ctx.fillStyle = "#94a3b8";
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
      ctx.fillStyle = "#94a3b8";
      ctx.font = "10px sans-serif";
      ctx.textAlign = i === 0 ? "left" : i === 4 ? "right" : "center";
      ctx.fillText(this.formatSmall(value), xx, plot.bottom + 16);
    }

    ctx.strokeStyle = "#cbd5e1";
    ctx.beginPath();
    ctx.moveTo(plot.left, plot.top);
    ctx.lineTo(plot.left, plot.bottom);
    ctx.lineTo(plot.right, plot.bottom);
    ctx.stroke();

    this.drawLine(ctx, x, pre, xAt, yAt, "#2563eb");
    this.drawLine(ctx, x, post, xAt, yAt, "#b84235");

    ctx.fillStyle = "#0f172a";
    ctx.font = "600 13px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(`${this.regimeShortLabel(regime)} CDF`, plot.left, 18);
    ctx.fillStyle = "#64748b";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("Pre", plot.right - 82, 18);
    ctx.fillText("Post", plot.right - 34, 18);
    ctx.fillStyle = "#2563eb";
    ctx.fillRect(plot.right - 102, 11, 14, 3);
    ctx.fillStyle = "#b84235";
    ctx.fillRect(plot.right - 56, 11, 14, 3);
    ctx.textAlign = "right";
    ctx.fillStyle = "#64748b";
    ctx.fillText("epsilon", plot.right, height - 8);

    if (hoverIndex == null) {
      readout.innerHTML = "";
    } else {
      const epsilon = Number.isFinite(Number(hoverEpsilon)) ? Number(hoverEpsilon) : x[hoverIndex];
      const px = xAt(epsilon);
      const preValue = this.interpolateCurve(x, pre, epsilon);
      const postValue = this.interpolateCurve(x, post, epsilon);
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = "#475569";
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
        regime delta: ${this.formatPct(delta)}
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
    if (this.viewMode === "low" || this.viewMode === "high") {
      this.app.registerLegend?.(this.legendId, {
        title: `${this.focusTitle()} epsilon change`,
        html: `
          ${this.renderContinuousLegendBar()}
          <div style="display:flex;align-items:center;gap:6px;font-size:10px;color:#64748b;margin-top:8px">
            <span style="width:12px;height:12px;border-radius:50%;background:#d8dee8;border:1px solid rgba(15,23,42,.16)"></span>
            <span>Insufficient ${this.focusTitle().toLowerCase()} data</span>
          </div>
          <div style="font-size:10px;color:#64748b;margin-top:8px">Relative epsilon change after 1990; values are clipped to the displayed scale.</div>
        `
      });
      return;
    }
    const counts = this.categoryCounts();
    this.app.registerLegend?.(this.legendId, {
      title: "Low x high epsilon class",
      html: `
        ${this.renderBivariateMatrix(counts, true)}
        <div style="display:flex;align-items:center;gap:6px;font-size:10px;color:#64748b;margin-top:8px">
          <span style="width:12px;height:12px;border-radius:50%;background:#d8dee8;border:1px solid rgba(15,23,42,.16)"></span>
          <span>Insufficient low/high data</span>
        </div>
        <div style="font-size:10px;color:#64748b;margin-top:8px">Stable: relative change within +/-${this.stableThresholdPct}%.</div>
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

  formatNumber(value, digits = 2) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(digits) : "NA";
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
