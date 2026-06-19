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
    this.layerId = "epsilon-catchments";
    this.legendId = `${manifest.id || "epsilon-change"}-legend`;
    this.distributionModal = null;
    this.activeDistribution = null;
    this.handleModalPointer = (event) => this.onDistributionPointer(event);
    this.handleFeatureClick = (payload) => {
      if (payload.layer?.id !== this.layerId || payload.layer?.moduleId !== this.manifest.id) return;
      this.selected = payload.feature;
      this.showInspector(payload.feature);
      this.app.draw?.();
    };
  }

  async onLoad() {
    this.data = await this.fetchJson(this.resolve("./data/epsilon-catchment-distributions.json"));
    this.basins = (this.data.basins || [])
      .filter((basin) => Number.isFinite(Number(basin.lon)) && Number.isFinite(Number(basin.lat)))
      .map((basin) => ({
        ...basin,
        id: String(basin.GCIN),
        lon: Number(basin.lon),
        lat: Number(basin.lat),
        area_km2: Number(basin.area_km2 || 0)
      }));
    this.byId = new Map(this.basins.map((basin) => [basin.id, basin]));
    this.addLayer();
    this.ensurePreviewStyles();
    this.ensureLegend();
    this.showOverview();
    Foundation.eventBus.on(Foundation.Events.FEATURE_CLICK, this.handleFeatureClick);
    this.app.draw?.();
  }

  onUnload() {
    this.app.layerManager.removeLayer(this.layerId);
    this.app.unregisterLegend?.(this.legendId);
    Foundation.eventBus.off(Foundation.Events.FEATURE_CLICK, this.handleFeatureClick);
    this.selected = null;
    this.closeDistributionModal();
  }

  getLayerIds() {
    return [this.layerId];
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
      name: "Epsilon change",
      type: "vector",
      visible: true,
      interactive: true,
      moduleId: this.manifest.id,
      groupPath: ["epsilon"],
      metadata: {
        periods: this.data?.meta?.periods,
        regimes: this.data?.meta?.regimes
      },
      renderer: (ctx, _layer, viewport) => this.render(ctx, viewport),
      hitTest: (lon, lat, viewport) => this.hitTest(lon, lat, viewport)
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
    const maxAbs = 35;

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
        ctx.fillStyle = this.colorFor(Number(basin.all_relative_delta_pct), maxAbs);
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
    const mid = this.basins.map((basin) => Number(basin.mid_relative_delta_pct)).filter(Number.isFinite);
    const high = this.basins.map((basin) => Number(basin.high_relative_delta_pct)).filter(Number.isFinite);
    const content = `
      <p style="margin:0 0 14px;color:#64748b;font-size:12px;line-height:1.6">
        Cross-fitted daily epsilon inference summarized by catchment. Points are catchment centroids; color shows all-recession relative epsilon change from 1982-1990 to 1991-2019.
      </p>
      <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-bottom:16px">
        ${this.metricCard("Catchments", this.basins.length.toLocaleString())}
        ${this.metricCard("All-recession mean", this.formatPct(this.mean(all)), this.mean(all))}
        ${this.metricCard("Low-flow mean", this.formatPct(this.mean(low)), this.mean(low))}
        ${this.metricCard("Mid-flow mean", this.formatPct(this.mean(mid)), this.mean(mid))}
        ${this.metricCard("High-flow mean", this.formatPct(this.mean(high)), this.mean(high))}
      </div>
      <div style="font-size:12px;color:#475569;line-height:1.65">
        Low-flow epsilon uses recession days with Q_obs at or below each catchment's Q10. Mid-flow uses Q10 to Q90. High-flow uses days at or above Q90.
      </div>
    `;
    this.app.showInspector?.("Epsilon Change", content);
  }

  showInspector(basin) {
    const title = `GCIN ${basin.GCIN}`;
    const curves = this.data.curves?.[String(basin.GCIN)] || {};
    const cards = `
      <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-bottom:14px">
        ${this.metricCard("Area", `${this.formatNumber(basin.area_km2, 1)} km2`)}
        ${this.metricCard("Aridity", this.formatNumber(basin.Aridity, 3))}
        ${this.metricCard("Precip.", `${this.formatNumber(basin.Prec_mm, 1)} mm`)}
        ${this.metricCard("Temp.", `${this.formatNumber(basin.Temp_C, 1)} C`)}
      </div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:16px">
        ${this.metricCard("All", this.formatPct(basin.all_relative_delta_pct), basin.all_relative_delta_pct)}
        ${this.metricCard("Low", this.formatPct(basin.low_relative_delta_pct), basin.low_relative_delta_pct)}
        ${this.metricCard("Mid", this.formatPct(basin.mid_relative_delta_pct), basin.mid_relative_delta_pct)}
        ${this.metricCard("High", this.formatPct(basin.high_relative_delta_pct), basin.high_relative_delta_pct)}
      </div>
    `;

    const sections = ["all", "low", "mid", "high"].map((regime) => `
      <section style="margin-top:16px;padding-top:14px;border-top:1px solid #e2e8f0">
        <h3 style="margin:0 0 8px;font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#64748b">${this.regimeLabel(regime)}</h3>
        ${this.renderStatsTable(basin, regime)}
        <div
          class="epsilon-curve-preview"
          data-gcin="${this.escape(basin.GCIN)}"
          data-regime="${this.escape(regime)}"
          title="Open density and CDF"
          style="display:grid;grid-template-columns:1fr;gap:8px;margin-top:10px;cursor:pointer"
        >
          ${this.renderCurveSvg(curves[regime], "density")}
          ${this.renderCurveSvg(curves[regime], "cdf")}
          <div class="epsilon-preview-hint">Open density and CDF</div>
        </div>
      </section>
    `).join("");

    this.app.showInspector?.(title, `
      <p style="margin:0 0 14px;color:#64748b;font-size:12px;line-height:1.6">
        Epsilon is the modeled daily ratio GQ/Q. This panel compares the inferred epsilon distribution before and after 1990.
      </p>
      ${cards}
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

  renderStatsTable(basin, regime) {
    const rows = [
      ["Mean", basin[`${regime}_pre_mean`], basin[`${regime}_post_mean`]],
      ["Median", basin[`${regime}_pre_q50`], basin[`${regime}_post_q50`]],
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

  renderCurveSvg(curve, mode) {
    if (!curve?.x?.length) return `<div style="font-size:11px;color:#64748b">No ${mode} data.</div>`;
    const width = 300;
    const height = 122;
    const margin = { left: 34, right: 8, top: 16, bottom: 22 };
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;
    const x = curve.x.map(Number);
    const pre = mode === "density" ? curve.preDensity.map(Number) : curve.preCdf.map(Number);
    const post = mode === "density" ? curve.postDensity.map(Number) : curve.postCdf.map(Number);
    const maxX = Math.max(...x, 1e-12);
    const maxY = mode === "density" ? Math.max(...pre, ...post, 1e-12) : 1;
    const sx = (value) => margin.left + (value / maxX) * plotW;
    const sy = (value) => margin.top + plotH - (value / maxY) * plotH;
    const path = (values) => x.map((value, index) => `${index ? "L" : "M"}${sx(value).toFixed(1)},${sy(values[index] || 0).toFixed(1)}`).join(" ");
    const label = mode === "density" ? "Density" : "CDF";
    return `
      <svg viewBox="0 0 ${width} ${height}" style="display:block;width:100%;height:auto;background:#f8fafc;border:1px solid #e2e8f0;border-radius:4px;pointer-events:none">
        <line x1="${margin.left}" y1="${margin.top + plotH}" x2="${width - margin.right}" y2="${margin.top + plotH}" stroke="#cbd5e1"/>
        <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + plotH}" stroke="#cbd5e1"/>
        <text x="${margin.left}" y="11" fill="#64748b" font-size="10">${label}</text>
        <text x="${width - margin.right}" y="${height - 6}" fill="#64748b" font-size="9" text-anchor="end">epsilon</text>
        <path d="${path(pre)}" fill="none" stroke="#2563eb" stroke-width="1.8"/>
        <path d="${path(post)}" fill="none" stroke="#b84235" stroke-width="1.8"/>
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
      .epsilon-curve-preview{border:1px solid transparent;border-radius:6px;padding:6px;transition:border-color .16s ease,background .16s ease,box-shadow .16s ease,transform .16s ease}
      .epsilon-curve-preview:hover{border-color:#93c5fd!important;background:#eff6ff!important;box-shadow:0 10px 22px rgba(37,99,235,.12);transform:translateY(-1px)}
      .epsilon-curve-preview:hover svg{border-color:#93c5fd!important;background:#f8fbff!important}
      .epsilon-preview-hint{font-size:10px;color:#64748b;text-align:right;transition:color .16s ease,font-weight .16s ease}
      .epsilon-curve-preview:hover .epsilon-preview-hint{color:#1d4ed8!important;font-weight:700}
    `;
    document.head.appendChild(style);
  }

  ensureDistributionModal() {
    if (this.distributionModal) return;

    const style = document.createElement("style");
    style.textContent = `
      .epsilon-modal{position:fixed;inset:0;z-index:420;display:none;align-items:center;justify-content:center;background:rgba(15,23,42,.36);padding:26px}
      .epsilon-modal.visible{display:flex}
      .epsilon-dialog{width:min(1060px,calc(100vw - 52px));height:min(760px,calc(100vh - 52px));background:#fff;border-radius:8px;box-shadow:0 22px 58px rgba(15,23,42,.28);display:flex;flex-direction:column;overflow:hidden}
      .epsilon-dialog-header{height:58px;padding:0 18px;border-bottom:1px solid #e2e8f0;display:flex;align-items:center;justify-content:space-between;gap:16px}
      .epsilon-dialog-title{font-size:15px;font-weight:700;color:#0f172a}
      .epsilon-dialog-subtitle{font-size:11px;color:#64748b;margin-top:3px}
      .epsilon-close{width:30px;height:30px;border:0;background:transparent;border-radius:4px;cursor:pointer;font-size:22px;color:#64748b;line-height:1}
      .epsilon-close:hover{background:#f1f5f9;color:#0f172a}
      .epsilon-chart-area{flex:1;min-height:0;padding:14px 18px 18px;display:grid;grid-template-rows:1fr 1fr;gap:12px}
      .epsilon-chart-card{position:relative;border:1px solid #e2e8f0;border-radius:6px;background:#f8fafc;overflow:hidden}
      .epsilon-chart-card canvas{display:block;width:100%;height:100%}
      .epsilon-readout{position:absolute;right:12px;top:10px;min-width:210px;padding:8px 10px;border:1px solid #dbe3ef;border-radius:6px;background:rgba(255,255,255,.94);font-size:11px;color:#334155;line-height:1.45;box-shadow:0 8px 20px rgba(15,23,42,.08);pointer-events:none}
      .epsilon-chart-card.cdf .epsilon-readout{top:auto;bottom:12px}
      .epsilon-readout strong{color:#0f172a}
    `;
    document.head.appendChild(style);

    const modal = document.createElement("div");
    modal.className = "epsilon-modal";
    modal.id = "epsilon-distribution-modal";
    modal.innerHTML = `
      <div class="epsilon-dialog">
        <div class="epsilon-dialog-header">
          <div>
            <div class="epsilon-dialog-title" id="epsilon-modal-title">Epsilon distribution</div>
            <div class="epsilon-dialog-subtitle" id="epsilon-modal-subtitle">Density and CDF</div>
          </div>
          <button class="epsilon-close" id="epsilon-modal-close" type="button" aria-label="Close">x</button>
        </div>
        <div class="epsilon-chart-area">
          <div class="epsilon-chart-card density">
            <canvas id="epsilon-density-canvas"></canvas>
            <div class="epsilon-readout" id="epsilon-density-readout"></div>
          </div>
          <div class="epsilon-chart-card cdf">
            <canvas id="epsilon-cdf-canvas"></canvas>
            <div class="epsilon-readout" id="epsilon-cdf-readout"></div>
          </div>
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
        this.activeDistribution.hoverIndex = null;
        this.drawDistributionModal();
      });
    });
    this.distributionModal = modal;
  }

  openDistributionModal(gcin, regime) {
    const basin = this.byId.get(String(gcin));
    const curve = this.data.curves?.[String(gcin)]?.[regime];
    if (!basin || !curve?.x?.length) return;
    this.ensureDistributionModal();
    this.activeDistribution = { basin, regime, curve, hoverIndex: null };
    this.distributionModal.querySelector("#epsilon-modal-title").textContent = `GCIN ${basin.GCIN} - ${this.regimeLabel(regime)}`;
    this.distributionModal.querySelector("#epsilon-modal-subtitle").textContent =
      "Pre 1982-1990 vs post 1991-2019; move the cursor across either panel to read matched density and CDF values.";
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
    const rect = canvas.getBoundingClientRect();
    const plot = this.distributionPlot(rect.width, rect.height);
    const x = event.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, (x - plot.left) / Math.max(1, plot.right - plot.left)));
    const n = this.activeDistribution.curve.x.length;
    this.activeDistribution.hoverIndex = Math.max(0, Math.min(n - 1, Math.round(ratio * (n - 1))));
    this.drawDistributionModal();
  }

  drawDistributionModal() {
    if (!this.activeDistribution) return;
    this.drawDistributionCanvas(
      this.distributionModal.querySelector("#epsilon-density-canvas"),
      this.distributionModal.querySelector("#epsilon-density-readout"),
      "density"
    );
    this.drawDistributionCanvas(
      this.distributionModal.querySelector("#epsilon-cdf-canvas"),
      this.distributionModal.querySelector("#epsilon-cdf-readout"),
      "cdf"
    );
  }

  drawDistributionCanvas(canvas, readout, mode) {
    const { basin, regime, curve, hoverIndex } = this.activeDistribution;
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
    const pre = mode === "density" ? curve.preDensity.map(Number) : curve.preCdf.map(Number);
    const post = mode === "density" ? curve.postDensity.map(Number) : curve.postCdf.map(Number);
    const maxX = Math.max(...x, 1e-12);
    const maxY = mode === "density" ? Math.max(...pre, ...post, 1e-12) : 1;
    const xAt = (value) => plot.left + (value / maxX) * (plot.right - plot.left);
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
    ctx.fillText(mode === "density" ? "Density" : "CDF", plot.left, 18);
    ctx.fillStyle = "#64748b";
    ctx.font = "11px sans-serif";
    ctx.fillText("Pre", plot.left + 82, 18);
    ctx.fillText("Post", plot.left + 132, 18);
    ctx.fillStyle = "#2563eb";
    ctx.fillRect(plot.left + 62, 11, 14, 3);
    ctx.fillStyle = "#b84235";
    ctx.fillRect(plot.left + 110, 11, 14, 3);
    ctx.textAlign = "right";
    ctx.fillStyle = "#64748b";
    ctx.fillText("epsilon", plot.right, height - 10);
    ctx.fillText(this.formatSmall(maxX), plot.right, plot.bottom + 16);
    ctx.textAlign = "left";
    ctx.fillText("0", plot.left, plot.bottom + 16);

    const idx = hoverIndex == null ? Math.floor(x.length / 2) : hoverIndex;
    const epsilon = x[idx];
    const px = xAt(epsilon);
    const preValue = pre[idx] || 0;
    const postValue = post[idx] || 0;

    if (hoverIndex != null) {
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
    }

    const delta = Number(basin[`${regime}_relative_delta_pct`]);
    readout.innerHTML = `
      <strong>${mode === "density" ? "Density" : "CDF"}</strong><br>
      epsilon: ${this.formatSmall(epsilon)}<br>
      pre: ${this.formatSmall(preValue)}<br>
      post: ${this.formatSmall(postValue)}<br>
      regime delta: ${this.formatPct(delta)}
    `;
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

  distributionPlot(width, height) {
    return {
      left: 64,
      right: width - 32,
      top: 34,
      bottom: height - 34
    };
  }

  ensureLegend() {
    this.app.registerLegend?.(this.legendId, {
      title: "Epsilon relative change",
      html: `
        <div style="display:grid;grid-template-columns:auto 1fr auto;gap:8px;align-items:center;font-size:11px;color:#64748b">
          <span>Decrease</span>
          <span style="height:10px;border-radius:999px;background:linear-gradient(90deg,#2563eb,#f1e8c9,#b84235)"></span>
          <span>Increase</span>
        </div>
        <div style="font-size:11px;color:#64748b;margin-top:8px">Color uses all-recession relative epsilon change, clipped for display at +/-35%.</div>
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
      mid: "Mid flow (Q10 < Q < Q90)",
      high: "High flow (Q >= Q90)"
    }[regime] || regime;
  }

  escape(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }
};
