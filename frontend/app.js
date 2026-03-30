const FIXED_WINDOW_SIZE = 25;
const STATIC_PAYLOAD_VERSION = "20260330c";
const ACTUAL_COLOR = "#2b78e4";
const PREDICTED_COLOR = "#d98c3f";
const MODEL_COLOR_PALETTE = {
  actual: "#2b78e4",
  hybrid_coordinated: "#d94801",
  gru: "#4c6ef5",
  bilstm: "#228be6",
  random_forest: "#2b8a3e",
  gradient_boosting: "#c77d00",
  linear_regression: "#6f42c1",
  svr_rbf: "#1f7a8c",
  knn_regressor: "#8f5f00",
  xgboost: "#b5179e",
};

const state = {
  dashboard: null,
  cropIndex: 0,
  timeIndex: 0,
  isPlaying: false,
  playTimer: null,
  windowStart: 0,
  chartPlotPoints: [],
  hoverIndex: null,
  modelChartPlotPoints: [],
  modelHoverIndex: null,
  seriesVisible: {
    actual: true,
    predicted: true,
  },
  modelSeriesVisible: {},
  thresholds: {
    low_threshold_c: 22.0,
    high_threshold_c: 29.0,
    spray_threshold_c: 31.0,
  },
};

const els = {
  toastContainer: document.getElementById("toastContainer"),
  samplesTableBody: document.getElementById("samplesTableBody"),
  modelComparisonBody: document.getElementById("modelComparisonBody"),
  primaryGraphNote: document.getElementById("primaryGraphNote"),
  modelGraphNote: document.getElementById("modelGraphNote"),
  cropSelect: document.getElementById("cropSelect"),
  refreshBtn: document.getElementById("refreshBtn"),
  playBtn: document.getElementById("playBtn"),
  timeSlider: document.getElementById("timeSlider"),
  sliderText: document.getElementById("sliderText"),
  timestampLabel: document.getElementById("timestampLabel"),
  predictedTemp: document.getElementById("predictedTemp"),
  actualTemp: document.getElementById("actualTemp"),
  absError: document.getElementById("absError"),
  tempGaugeFill: document.getElementById("tempGaugeFill"),
  temperatureHint: document.getElementById("temperatureHint"),
  actionLabel: document.getElementById("actionLabel"),
  fanWidget: document.getElementById("fanWidget"),
  fanLabel: document.getElementById("fanLabel"),
  sprayWidget: document.getElementById("sprayWidget"),
  fanCard: document.getElementById("fanCard"),
  sprayCard: document.getElementById("sprayCard"),
  sprayLabel: document.getElementById("sprayLabel"),
  maeMetric: document.getElementById("maeMetric"),
  rmseMetric: document.getElementById("rmseMetric"),
  mapeMetric: document.getElementById("mapeMetric"),
  r2Metric: document.getElementById("r2Metric"),
  modelCoordination: document.getElementById("modelCoordination"),
  bestModelLabel: document.getElementById("bestModelLabel"),
  hybridGainLabel: document.getElementById("hybridGainLabel"),
  weightsText: document.getElementById("weightsText"),
  windowScroll: document.getElementById("windowScroll"),
  windowRangeLabel: document.getElementById("windowRangeLabel"),
  ruleHint: document.getElementById("ruleHint"),
  chart: document.getElementById("temperatureChart"),
  chartCanvasWrap: document.getElementById("chartCanvasWrap"),
  chartTooltip: document.getElementById("chartTooltip"),
  legendActual: document.getElementById("legendActual"),
  legendPredicted: document.getElementById("legendPredicted"),
  modelLegend: document.getElementById("modelLegend"),
  modelChart: document.getElementById("modelTemperatureChart"),
  modelChartCanvasWrap: document.getElementById("modelChartCanvasWrap"),
  modelChartTooltip: document.getElementById("modelChartTooltip"),
};

function clamp(value, minValue, maxValue) {
  return Math.max(minValue, Math.min(value, maxValue));
}

function formatTemp(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return Number(value).toFixed(2);
}

function formatTimestampLabel(rawTs) {
  const ts = String(rawTs || "").replace("T", " ");
  if (!ts) return "--";
  if (ts.length >= 16) {
    return `${ts.slice(5, 10)} ${ts.slice(11, 16)}`;
  }
  return ts.slice(0, 16);
}

function prettyModelName(modelKey) {
  const text = String(modelKey || "").replace(/_/g, " ").trim();
  if (!text) return "--";
  return text
    .split(" ")
    .map((part) => (part.length <= 2 ? part.toUpperCase() : part.charAt(0).toUpperCase() + part.slice(1)))
    .join(" ");
}

function formatCoordinationText(value) {
  const raw = String(value || "").trim();
  if (!raw || raw === "--") return "--";

  const openIdx = raw.indexOf("[");
  const closeIdx = raw.lastIndexOf("]");
  if (openIdx === -1 || closeIdx === -1 || closeIdx <= openIdx) {
    return raw.replace(/_/g, " ");
  }

  const method = raw.slice(0, openIdx).replace(/_/g, " ").trim();
  const weightsRaw = raw.slice(openIdx + 1, closeIdx).trim();
  if (!weightsRaw) return method || "--";

  const formattedWeights = weightsRaw
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => {
      const [model, weight] = item.split(":");
      if (!model) return item;
      const prettyModel = prettyModelName(model.trim());
      const parsedWeight = Number(weight);
      const prettyWeight = Number.isFinite(parsedWeight) ? parsedWeight.toFixed(2) : String(weight || "").trim();
      return `${prettyModel}: ${prettyWeight}`;
    })
    .join("\n");

  return `${method}\n${formattedWeights}`;
}

function rmseForModel(crop, modelKey) {
  const rows = Array.isArray(crop.model_comparison) ? crop.model_comparison : [];
  const hit = rows.find((row) => String(row.model || "").toLowerCase() === String(modelKey || "").toLowerCase());
  return hit ? Number(hit.rmse) : null;
}

function setLoadingState(message) {
  els.predictedTemp.textContent = "--";
  els.actualTemp.textContent = "-- °C";
  els.absError.textContent = "-- °C";
  els.temperatureHint.textContent = message;
  if (els.bestModelLabel) els.bestModelLabel.textContent = "--";
  if (els.hybridGainLabel) els.hybridGainLabel.textContent = "--";
  if (els.weightsText) els.weightsText.textContent = "--";
  if (els.modelComparisonBody) {
    els.modelComparisonBody.innerHTML = "<tr><td colspan='5'>Loading model comparison...</td></tr>";
  }
  if (els.primaryGraphNote) {
    els.primaryGraphNote.textContent = "This graph is the final operational output used by fan/spray decision logic.";
  }
  if (els.modelGraphNote) {
    els.modelGraphNote.textContent = "This graph compares each model against actual values.";
  }
  hideTooltip();
  hideModelTooltip();
}

function showToast(message, type = "success") {
  if (!els.toastContainer) return;
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  els.toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.remove();
  }, 2600);
}

function getCurrentCrop() {
  if (!state.dashboard || !state.dashboard.crops || state.dashboard.crops.length === 0) {
    return null;
  }
  return state.dashboard.crops[state.cropIndex];
}

function updateCropSelect() {
  const crops = state.dashboard?.crops ?? [];
  els.cropSelect.innerHTML = "";
  crops.forEach((crop, idx) => {
    const option = document.createElement("option");
    option.value = String(idx);
    option.textContent = `Class ${crop.crop_type}`;
    els.cropSelect.appendChild(option);
  });
  els.cropSelect.value = String(state.cropIndex);
}

function updateActionChip(action) {
  const normalized = String(action || "unknown").toLowerCase();
  els.actionLabel.classList.remove("neutral", "good", "warn", "danger");

  if (normalized === "fan_and_spray_on") {
    els.actionLabel.classList.add("danger");
    els.actionLabel.textContent = "Fan + Spray ON";
    return;
  }
  if (normalized === "fan_on") {
    els.actionLabel.classList.add("warn");
    els.actionLabel.textContent = "Fan ON";
    return;
  }
  if (normalized === "idle") {
    els.actionLabel.classList.add("good");
    els.actionLabel.textContent = "Stable (Idle)";
    return;
  }
  if (normalized === "cooling_off") {
    els.actionLabel.classList.add("neutral");
    els.actionLabel.textContent = "Cooling OFF";
    return;
  }
  els.actionLabel.classList.add("neutral");
  els.actionLabel.textContent = normalized.replace(/_/g, " ");
}

function updateDevices(fanOn, sprayOn) {
  els.fanWidget.classList.toggle("on", fanOn === 1);
  els.fanWidget.classList.toggle("off", fanOn !== 1);
  els.fanLabel.textContent = fanOn === 1 ? "ON (Rotating)" : "OFF (Static)";
  els.fanLabel.classList.toggle("on", fanOn === 1);
  els.fanLabel.classList.toggle("off", fanOn !== 1);
  els.fanCard.classList.toggle("on-state", fanOn === 1);
  els.fanCard.classList.toggle("off-state", fanOn !== 1);

  els.sprayWidget.classList.toggle("on", sprayOn === 1);
  els.sprayWidget.classList.toggle("off", sprayOn !== 1);
  els.sprayLabel.textContent = sprayOn === 1 ? "ON (Mist Active)" : "OFF (Static)";
  els.sprayLabel.classList.toggle("on", sprayOn === 1);
  els.sprayLabel.classList.toggle("off", sprayOn !== 1);
  els.sprayCard.classList.toggle("on-state", sprayOn === 1);
  els.sprayCard.classList.toggle("off-state", sprayOn !== 1);
}

function updateMetrics(crop) {
  const metrics = crop.metrics || {};
  els.maeMetric.textContent = metrics.mae !== null && metrics.mae !== undefined ? metrics.mae.toFixed(3) : "--";
  els.rmseMetric.textContent = metrics.rmse !== null && metrics.rmse !== undefined ? metrics.rmse.toFixed(3) : "--";
  els.mapeMetric.textContent =
    metrics.mape_pct !== null && metrics.mape_pct !== undefined ? `${metrics.mape_pct.toFixed(2)}%` : "--";
  els.r2Metric.textContent = metrics.r2 !== null && metrics.r2 !== undefined ? metrics.r2.toFixed(3) : "--";
  els.modelCoordination.textContent = formatCoordinationText(crop.model_coordination || metrics.model_coordination);
}

function updateModelInsights(crop) {
  if (!els.bestModelLabel || !els.hybridGainLabel || !els.weightsText) return;

  const bestModel = crop.best_model || crop.metrics?.best_model || "--";
  els.bestModelLabel.textContent = prettyModelName(bestModel);

  const hybridRmse = Number(crop.metrics?.rmse);
  const bestModelRmse = rmseForModel(crop, bestModel);
  if (Number.isFinite(hybridRmse) && Number.isFinite(bestModelRmse)) {
    const diff = bestModelRmse - hybridRmse;
    const sign = diff >= 0 ? "+" : "";
    els.hybridGainLabel.textContent = `${sign}${diff.toFixed(3)} vs best RMSE`;
  } else {
    els.hybridGainLabel.textContent = "--";
  }

  const weights = crop.model_weights || {};
  const pairs = Object.entries(weights);
  if (!pairs.length) {
    els.weightsText.textContent = "No model weights available in this crop payload.";
    return;
  }

  const text = pairs
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .map(([key, value]) => `${prettyModelName(key)}: ${Number(value).toFixed(2)}`)
    .join(" | ");
  els.weightsText.textContent = text;
}

function renderModelComparisonRows(crop) {
  if (!els.modelComparisonBody) return;
  const rows = Array.isArray(crop.model_comparison) ? crop.model_comparison : [];
  if (!rows.length) {
    els.modelComparisonBody.innerHTML = "<tr><td colspan='5'>No model comparison rows found.</td></tr>";
    return;
  }

  const sortedRows = [...rows].sort((a, b) => Number(a.rank) - Number(b.rank));
  els.modelComparisonBody.innerHTML = "";
  sortedRows.forEach((row) => {
    const tr = document.createElement("tr");
    const modelKey = String(row.model || "").toLowerCase();
    if (Number(row.rank) === 1) tr.classList.add("row-best");
    if (modelKey === "hybrid_coordinated") tr.classList.add("row-hybrid");
    tr.innerHTML =
      `<td>${Number(row.rank)}</td>`
      + `<td>${prettyModelName(row.model)}</td>`
      + `<td>${formatTemp(row.mae)}</td>`
      + `<td>${formatTemp(row.rmse)}</td>`
      + `<td>${Number(row.r2).toFixed(3)}</td>`;
    els.modelComparisonBody.appendChild(tr);
  });
}

function updateImprovementValues(crop) {
  const bestModel = crop.best_model || crop.metrics?.best_model || "";

  if (els.primaryGraphNote) {
    els.primaryGraphNote.textContent =
      `Graph 1 shows Actual vs Hybrid Final prediction (used for control). Best individual model: ${prettyModelName(bestModel) || "--"}.`;
  }
  if (els.modelGraphNote) {
    els.modelGraphNote.textContent =
      "Graph 2 compares Actual, Hybrid, and each individual model. Use legend toggles to focus on any model.";
  }
}

function renderSampleRows(crop) {
  if (!els.samplesTableBody) return;

  const start = clamp(state.timeIndex - 7, 0, Math.max(0, crop.timestamps.length - 1));
  const end = clamp(state.timeIndex + 1, 0, crop.timestamps.length);
  const rows = [];
  for (let i = start; i < end; i += 1) {
    rows.push({
      ts: crop.timestamps[i],
      actual: crop.actual_temperature_c[i],
      predicted: crop.predicted_temperature_c[i],
      error: crop.absolute_error_c[i],
      action: crop.actions[i],
    });
  }

  if (!rows.length) {
    els.samplesTableBody.innerHTML = "<tr><td colspan='5'>No rows available.</td></tr>";
    return;
  }

  els.samplesTableBody.innerHTML = "";
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td>${String(row.ts || "").replace("T", " ")}</td>`
      + `<td>${formatTemp(row.actual)}</td>`
      + `<td>${formatTemp(row.predicted)}</td>`
      + `<td>${formatTemp(row.error)}</td>`
      + `<td>${String(row.action || "").replace(/_/g, " ")}</td>`;
    els.samplesTableBody.appendChild(tr);
  });
}

function updateLegendControls() {
  const actualOn = state.seriesVisible.actual;
  const predictedOn = state.seriesVisible.predicted;

  els.legendActual.classList.toggle("active", actualOn);
  els.legendActual.setAttribute("aria-pressed", actualOn ? "true" : "false");
  els.legendPredicted.classList.toggle("active", predictedOn);
  els.legendPredicted.setAttribute("aria-pressed", predictedOn ? "true" : "false");
}

function getWindowSize(totalPoints) {
  return Math.max(1, Math.min(FIXED_WINDOW_SIZE, totalPoints));
}

function syncWindowController(crop) {
  const total = crop.timestamps.length;
  const windowSize = getWindowSize(total);
  const maxStart = Math.max(0, total - windowSize);

  state.windowStart = clamp(state.windowStart, 0, maxStart);
  els.windowScroll.disabled = maxStart === 0;
  els.windowScroll.min = "0";
  els.windowScroll.max = String(maxStart);
  els.windowScroll.value = String(state.windowStart);

  const start = state.windowStart;
  const end = Math.min(total - 1, start + windowSize - 1);
  els.windowRangeLabel.textContent = `Points ${start + 1}-${end + 1} / ${total}`;
}

function keepCurrentStepVisible(crop) {
  const total = crop.timestamps.length;
  const windowSize = getWindowSize(total);
  if (state.timeIndex < state.windowStart) {
    state.windowStart = state.timeIndex;
    return;
  }
  const currentEnd = state.windowStart + windowSize - 1;
  if (state.timeIndex > currentEnd) {
    state.windowStart = state.timeIndex - windowSize + 1;
  }
}

function ensureCanvasSizeFor(canvasEl, desktopHeight = 560) {
  const rect = canvasEl.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const logicalWidth = Math.max(760, Math.floor(rect.width || 1200));
  const logicalHeight = window.innerWidth <= 760 ? 420 : desktopHeight;
  const pixelWidth = Math.floor(logicalWidth * dpr);
  const pixelHeight = Math.floor(logicalHeight * dpr);

  if (canvasEl.width !== pixelWidth || canvasEl.height !== pixelHeight) {
    canvasEl.width = pixelWidth;
    canvasEl.height = pixelHeight;
    canvasEl.style.height = `${logicalHeight}px`;
  }

  const ctx = canvasEl.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width: logicalWidth, height: logicalHeight };
}

function getVisibleSlice(crop) {
  const total = crop.timestamps.length;
  const windowSize = getWindowSize(total);
  const start = clamp(state.windowStart, 0, Math.max(0, total - windowSize));
  const endExclusive = Math.min(total, start + windowSize);

  return {
    total,
    windowSize,
    start,
    endExclusive,
    timestamps: crop.timestamps.slice(start, endExclusive),
    actual: crop.actual_temperature_c.slice(start, endExclusive).map(Number),
    predicted: crop.predicted_temperature_c.slice(start, endExclusive).map(Number),
  };
}

function getModelSeries(crop) {
  const rawSeries = crop.model_prediction_series || {};
  const allSeries = {
    actual: crop.actual_temperature_c.map(Number),
    hybrid_coordinated: crop.predicted_temperature_c.map(Number),
  };

  Object.entries(rawSeries).forEach(([key, values]) => {
    if (!Array.isArray(values) || values.length !== crop.timestamps.length) return;
    allSeries[String(key)] = values.map(Number);
  });

  return allSeries;
}

function ensureModelSeriesVisibility(crop) {
  const series = getModelSeries(crop);
  const bestModel = String(crop.best_model || "").toLowerCase();
  Object.keys(series).forEach((key) => {
    if (state.modelSeriesVisible[key] === undefined) {
      state.modelSeriesVisible[key] = (
        key === "actual"
        || key === "hybrid_coordinated"
        || key === bestModel
      );
    }
  });
}

function renderModelLegend(crop) {
  if (!els.modelLegend) return;
  const series = getModelSeries(crop);
  const keys = Object.keys(series);
  ensureModelSeriesVisibility(crop);

  els.modelLegend.innerHTML = "";
  keys.forEach((key) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "legend-item";
    if (state.modelSeriesVisible[key]) {
      button.classList.add("active");
    }
    button.setAttribute("aria-pressed", state.modelSeriesVisible[key] ? "true" : "false");
    button.innerHTML =
      `<span class="legend-swatch" style="background:${MODEL_COLOR_PALETTE[key] || "#60716d"}"></span>`
      + `<span>${prettyModelName(key)}</span>`;
    button.addEventListener("click", () => {
      state.modelSeriesVisible[key] = !state.modelSeriesVisible[key];
      renderModelLegend(crop);
      drawModelChart(crop, state.timeIndex);
    });
    els.modelLegend.appendChild(button);
  });
}

function drawSeries(ctx, values, xAt, yAt, color) {
  if (values.length < 2) return;

  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.6;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";

  values.forEach((value, i) => {
    const x = xAt(i);
    const y = yAt(value);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });

  ctx.stroke();
}

function drawChart(crop, currentIndex) {
  const { ctx, width, height } = ensureCanvasSizeFor(els.chart, 560);
  ctx.clearRect(0, 0, width, height);

  const visible = getVisibleSlice(crop);
  if (visible.actual.length === 0 || visible.predicted.length === 0) {
    ctx.fillStyle = "#6d7f7a";
    ctx.font = "16px Avenir Next";
    ctx.fillText("No chart data available.", 28, 38);
    return;
  }

  const left = 84;
  const right = 20;
  const top = 24;
  const bottom = 72;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;

  const values = [];
  if (state.seriesVisible.actual) values.push(...visible.actual);
  if (state.seriesVisible.predicted) values.push(...visible.predicted);
  if (values.length === 0) values.push(...visible.actual, ...visible.predicted);

  const yMinRaw = Math.min(...values);
  const yMaxRaw = Math.max(...values);
  const yPadding = Math.max(0.5, (yMaxRaw - yMinRaw) * 0.2);
  const yMin = yMinRaw - yPadding;
  const yMax = yMaxRaw + yPadding;
  const ySpan = Math.max(0.0001, yMax - yMin);

  const xAt = (idx) => left + (idx / Math.max(visible.windowSize - 1, 1)) * plotWidth;
  const yAt = (value) => top + (1 - (Number(value) - yMin) / ySpan) * plotHeight;

  ctx.strokeStyle = "rgba(126, 139, 136, 0.20)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i += 1) {
    const y = top + (i / 5) * plotHeight;
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(left + plotWidth, y);
    ctx.stroke();
  }

  for (let i = 0; i <= 5; i += 1) {
    const x = left + (i / 5) * plotWidth;
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, top + plotHeight);
    ctx.strokeStyle = "rgba(126, 139, 136, 0.10)";
    ctx.stroke();
  }

  ctx.fillStyle = "#5b6f6a";
  ctx.font = "12px Avenir Next";
  ctx.textAlign = "right";
  for (let i = 0; i <= 5; i += 1) {
    const value = yMax - (i / 5) * ySpan;
    const y = top + (i / 5) * plotHeight + 4;
    ctx.fillText(`${value.toFixed(1)}°C`, left - 8, y);
  }

  ctx.fillStyle = "#5b6f6a";
  ctx.font = "12px Avenir Next";
  ctx.textAlign = "left";
  ctx.fillText(
    `Visible: ${visible.start + 1}-${visible.endExclusive} (${visible.endExclusive - visible.start} points)`,
    left,
    top - 6,
  );

  if (state.seriesVisible.actual) {
    drawSeries(ctx, visible.actual, xAt, yAt, ACTUAL_COLOR);
  }
  if (state.seriesVisible.predicted) {
    drawSeries(ctx, visible.predicted, xAt, yAt, PREDICTED_COLOR);
  }

  if (currentIndex >= visible.start && currentIndex < visible.endExclusive) {
    const rel = currentIndex - visible.start;
    const markerX = xAt(rel);
    ctx.save();
    ctx.setLineDash([4, 5]);
    ctx.strokeStyle = "rgba(49, 66, 63, 0.45)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(markerX, top);
    ctx.lineTo(markerX, top + plotHeight);
    ctx.stroke();
    ctx.restore();
  }

  const tickCount = 4;
  ctx.fillStyle = "#5b6f6a";
  ctx.font = "12px Avenir Next";
  ctx.textAlign = "center";
  for (let t = 0; t <= tickCount; t += 1) {
    const rel = Math.round((t / tickCount) * Math.max(visible.windowSize - 1, 1));
    const tsIdx = clamp(rel, 0, visible.timestamps.length - 1);
    ctx.fillText(formatTimestampLabel(visible.timestamps[tsIdx]), xAt(rel), top + plotHeight + 24);
  }

  ctx.fillStyle = "#5b6f6a";
  ctx.font = "12px Avenir Next";
  ctx.textAlign = "center";
  ctx.fillText("Time", left + plotWidth / 2, height - 10);

  ctx.save();
  ctx.translate(18, top + plotHeight / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = "#5b6f6a";
  ctx.font = "12px Avenir Next";
  ctx.textAlign = "center";
  ctx.fillText("Temperature (°C)", 0, 0);
  ctx.restore();

  state.chartPlotPoints = visible.actual.map((value, relIdx) => {
    const idx = visible.start + relIdx;
    return {
      idx,
      x: xAt(relIdx),
      yActual: yAt(value),
      yPredicted: yAt(visible.predicted[relIdx]),
    };
  });

  if (state.hoverIndex !== null) {
    const hovered = state.chartPlotPoints.find((point) => point.idx === state.hoverIndex);
    if (hovered) {
      if (state.seriesVisible.actual) {
        ctx.fillStyle = ACTUAL_COLOR;
        ctx.beginPath();
        ctx.arc(hovered.x, hovered.yActual, 3.2, 0, Math.PI * 2);
        ctx.fill();
      }
      if (state.seriesVisible.predicted) {
        ctx.fillStyle = PREDICTED_COLOR;
        ctx.beginPath();
        ctx.arc(hovered.x, hovered.yPredicted, 3.2, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  if (!state.seriesVisible.actual && !state.seriesVisible.predicted) {
    ctx.fillStyle = "#7a8b87";
    ctx.font = "14px Avenir Next";
    ctx.textAlign = "left";
    ctx.fillText("Enable at least one series in legend to view the chart.", left, top + 22);
  }
}

function drawModelChart(crop, currentIndex) {
  if (!els.modelChart) return;
  const { ctx, width, height } = ensureCanvasSizeFor(els.modelChart, 520);
  ctx.clearRect(0, 0, width, height);

  const visible = getVisibleSlice(crop);
  if (visible.timestamps.length === 0) {
    ctx.fillStyle = "#6d7f7a";
    ctx.font = "16px Avenir Next";
    ctx.fillText("No model chart data available.", 28, 38);
    return;
  }

  ensureModelSeriesVisibility(crop);
  const modelSeries = getModelSeries(crop);
  const visibleKeys = Object.keys(modelSeries).filter((key) => state.modelSeriesVisible[key]);
  const activeKeys = visibleKeys.length ? visibleKeys : ["actual", "hybrid_coordinated"];

  const left = 84;
  const right = 20;
  const top = 24;
  const bottom = 70;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;

  const visibleSeries = {};
  activeKeys.forEach((key) => {
    const values = modelSeries[key];
    if (!Array.isArray(values)) return;
    visibleSeries[key] = values.slice(visible.start, visible.endExclusive).map(Number);
  });

  const allValues = Object.values(visibleSeries).flat();
  if (!allValues.length) {
    ctx.fillStyle = "#6d7f7a";
    ctx.font = "16px Avenir Next";
    ctx.fillText("No visible model series selected.", 28, 38);
    return;
  }

  const yMinRaw = Math.min(...allValues);
  const yMaxRaw = Math.max(...allValues);
  const yPadding = Math.max(0.5, (yMaxRaw - yMinRaw) * 0.2);
  const yMin = yMinRaw - yPadding;
  const yMax = yMaxRaw + yPadding;
  const ySpan = Math.max(0.0001, yMax - yMin);

  const xAt = (idx) => left + (idx / Math.max(visible.windowSize - 1, 1)) * plotWidth;
  const yAt = (value) => top + (1 - (Number(value) - yMin) / ySpan) * plotHeight;

  ctx.strokeStyle = "rgba(126, 139, 136, 0.18)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i += 1) {
    const y = top + (i / 5) * plotHeight;
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(left + plotWidth, y);
    ctx.stroke();
  }

  ctx.fillStyle = "#5b6f6a";
  ctx.font = "12px Avenir Next";
  ctx.textAlign = "right";
  for (let i = 0; i <= 5; i += 1) {
    const value = yMax - (i / 5) * ySpan;
    const y = top + (i / 5) * plotHeight + 4;
    ctx.fillText(`${value.toFixed(1)}°C`, left - 8, y);
  }

  Object.entries(visibleSeries).forEach(([key, values]) => {
    drawSeries(ctx, values, xAt, yAt, MODEL_COLOR_PALETTE[key] || "#60716d");
  });

  if (currentIndex >= visible.start && currentIndex < visible.endExclusive) {
    const rel = currentIndex - visible.start;
    const markerX = xAt(rel);
    ctx.save();
    ctx.setLineDash([4, 5]);
    ctx.strokeStyle = "rgba(49, 66, 63, 0.45)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(markerX, top);
    ctx.lineTo(markerX, top + plotHeight);
    ctx.stroke();
    ctx.restore();
  }

  const tickCount = 4;
  ctx.fillStyle = "#5b6f6a";
  ctx.font = "12px Avenir Next";
  ctx.textAlign = "center";
  for (let t = 0; t <= tickCount; t += 1) {
    const rel = Math.round((t / tickCount) * Math.max(visible.windowSize - 1, 1));
    const tsIdx = clamp(rel, 0, visible.timestamps.length - 1);
    ctx.fillText(formatTimestampLabel(visible.timestamps[tsIdx]), xAt(rel), top + plotHeight + 22);
  }

  state.modelChartPlotPoints = visible.timestamps.map((_, relIdx) => ({
    idx: visible.start + relIdx,
    x: xAt(relIdx),
  }));
}

function hideTooltip() {
  state.hoverIndex = null;
  els.chartTooltip.classList.add("hidden");
}

function hideModelTooltip() {
  state.modelHoverIndex = null;
  if (els.modelChartTooltip) {
    els.modelChartTooltip.classList.add("hidden");
  }
}

function findNearestPlotPoint(mouseX) {
  if (!state.chartPlotPoints.length) return null;

  let nearest = null;
  let minDistance = Number.POSITIVE_INFINITY;

  state.chartPlotPoints.forEach((point) => {
    const distance = Math.abs(point.x - mouseX);
    if (distance < minDistance) {
      minDistance = distance;
      nearest = point;
    }
  });

  if (minDistance > 24) return null;
  return nearest;
}

function findNearestModelPlotPoint(mouseX) {
  if (!state.modelChartPlotPoints.length) return null;

  let nearest = null;
  let minDistance = Number.POSITIVE_INFINITY;
  state.modelChartPlotPoints.forEach((point) => {
    const distance = Math.abs(point.x - mouseX);
    if (distance < minDistance) {
      minDistance = distance;
      nearest = point;
    }
  });

  if (minDistance > 24) return null;
  return nearest;
}

function showTooltip(point) {
  const crop = getCurrentCrop();
  if (!crop) return;

  const ts = crop.timestamps[point.idx];
  const actual = crop.actual_temperature_c[point.idx];
  const predicted = crop.predicted_temperature_c[point.idx];

  els.chartTooltip.innerHTML =
    `<strong>${String(ts).replace("T", " ")}</strong><br>`
    + `<span>Actual: ${formatTemp(actual)}°C</span><br>`
    + `<span>Predicted: ${formatTemp(predicted)}°C</span>`;

  const wrapRect = els.chartCanvasWrap.getBoundingClientRect();
  const canvasRect = els.chart.getBoundingClientRect();
  const baseLeft = canvasRect.left - wrapRect.left;

  let left = baseLeft + point.x + 12;
  const top = 10;
  const tooltipWidth = 210;

  if (left + tooltipWidth > wrapRect.width - 8) {
    left = wrapRect.width - tooltipWidth - 8;
  }
  if (left < 8) {
    left = 8;
  }

  els.chartTooltip.style.left = `${left}px`;
  els.chartTooltip.style.top = `${top}px`;
  els.chartTooltip.classList.remove("hidden");
}

function showModelTooltip(point) {
  if (!els.modelChartTooltip || !els.modelChartCanvasWrap || !els.modelChart) return;
  const crop = getCurrentCrop();
  if (!crop) return;

  const ts = crop.timestamps[point.idx];
  const modelSeries = getModelSeries(crop);
  const visibleKeys = Object.keys(modelSeries).filter((key) => state.modelSeriesVisible[key]);
  const keys = visibleKeys.length ? visibleKeys : ["actual", "hybrid_coordinated"];

  const valuesText = keys
    .slice(0, 7)
    .map((key) => `${prettyModelName(key)}: ${formatTemp(modelSeries[key][point.idx])}°C`)
    .join("<br>");

  els.modelChartTooltip.innerHTML = `<strong>${String(ts).replace("T", " ")}</strong><br>${valuesText}`;

  const wrapRect = els.modelChartCanvasWrap.getBoundingClientRect();
  const canvasRect = els.modelChart.getBoundingClientRect();
  const baseLeft = canvasRect.left - wrapRect.left;

  let left = baseLeft + point.x + 12;
  const top = 10;
  const tooltipWidth = 230;
  if (left + tooltipWidth > wrapRect.width - 8) {
    left = wrapRect.width - tooltipWidth - 8;
  }
  if (left < 8) {
    left = 8;
  }

  els.modelChartTooltip.style.left = `${left}px`;
  els.modelChartTooltip.style.top = `${top}px`;
  els.modelChartTooltip.classList.remove("hidden");
}

function updateRuleHint() {
  const low = Number(state.thresholds.low_threshold_c);
  const high = Number(state.thresholds.high_threshold_c);
  const spray = Number(state.thresholds.spray_threshold_c);

  els.ruleHint.textContent =
    `OFF highlight: Fan OFF and Spray OFF when predicted temperature is below ${high.toFixed(1)}C. `
    + `Specifically, ${low.toFixed(1)}C to ${high.toFixed(1)}C is Idle (OFF), and below ${low.toFixed(1)}C is Cooling OFF. `
    + `Spray turns ON only at or above ${spray.toFixed(1)}C.`;
}

function drawCurrentChart() {
  const crop = getCurrentCrop();
  if (!crop) return;
  drawChart(crop, state.timeIndex);
  drawModelChart(crop, state.timeIndex);
}

function renderCurrentStep() {
  const crop = getCurrentCrop();
  if (!crop) return;

  const n = crop.timestamps.length;
  if (n === 0) return;

  state.timeIndex = clamp(state.timeIndex, 0, n - 1);
  keepCurrentStepVisible(crop);

  const idx = state.timeIndex;
  const predicted = Number(crop.predicted_temperature_c[idx]);
  const actual = Number(crop.actual_temperature_c[idx]);
  const absErr = Number(crop.absolute_error_c[idx]);
  const action = crop.actions[idx];
  const fanOn = Number(crop.fan_on[idx]);
  const sprayOn = Number(crop.spray_on[idx]);

  els.timeSlider.max = String(n - 1);
  els.timeSlider.value = String(idx);
  els.sliderText.textContent = `Step ${idx + 1} / ${n}`;
  els.timestampLabel.textContent = `Timestamp: ${crop.timestamps[idx]}`;

  els.predictedTemp.textContent = formatTemp(predicted);
  els.actualTemp.textContent = `${formatTemp(actual)} °C`;
  els.absError.textContent = `${formatTemp(absErr)} °C`;

  const gaugePercent = clamp(((predicted - 15) / 25) * 100, 0, 100);
  els.tempGaugeFill.style.width = `${gaugePercent}%`;

  const lowThreshold = Number(state.thresholds.low_threshold_c);
  const highThreshold = Number(state.thresholds.high_threshold_c);
  const sprayThreshold = Number(state.thresholds.spray_threshold_c);

  if (predicted >= sprayThreshold) {
    els.temperatureHint.textContent = "Critical heat: spray and fan should be active.";
  } else if (predicted >= highThreshold) {
    els.temperatureHint.textContent = "High heat: fan cooling is recommended.";
  } else if (predicted <= lowThreshold) {
    els.temperatureHint.textContent = "Lower range: active cooling can remain off.";
  } else {
    els.temperatureHint.textContent = "Comfort zone: greenhouse condition is stable.";
  }

  updateActionChip(action);
  updateDevices(fanOn, sprayOn);
  updateMetrics(crop);
  updateModelInsights(crop);
  renderModelComparisonRows(crop);
  updateImprovementValues(crop);
  renderModelLegend(crop);
  renderSampleRows(crop);
  syncWindowController(crop);
  drawChart(crop, idx);
  drawModelChart(crop, idx);
}

function stopPlayback() {
  if (state.playTimer) {
    clearInterval(state.playTimer);
    state.playTimer = null;
  }
  state.isPlaying = false;
  els.playBtn.textContent = "Play";
}

function togglePlayback() {
  const crop = getCurrentCrop();
  if (!crop || crop.timestamps.length === 0) return;

  if (state.isPlaying) {
    stopPlayback();
    return;
  }

  state.isPlaying = true;
  els.playBtn.textContent = "Pause";
  state.playTimer = setInterval(() => {
    const maxIndex = crop.timestamps.length - 1;
    state.timeIndex = state.timeIndex >= maxIndex ? 0 : state.timeIndex + 1;
    renderCurrentStep();
  }, 520);
}

function onCropChange() {
  stopPlayback();
  hideTooltip();
  hideModelTooltip();

  state.cropIndex = Number(els.cropSelect.value);
  const crop = getCurrentCrop();
  if (!crop || crop.timestamps.length === 0) {
    state.timeIndex = 0;
    state.windowStart = 0;
    return;
  }

  state.modelSeriesVisible = {};
  state.timeIndex = crop.timestamps.length - 1;
  state.windowStart = Math.max(0, crop.timestamps.length - getWindowSize(crop.timestamps.length));
  renderCurrentStep();
}

function getDashboardEndpoints(refresh = false) {
  const apiEndpoint = refresh ? "/api/dashboard?refresh=1" : "/api/dashboard";
  const staticEndpoint = refresh
    ? `/dashboard_payload.static.json?v=${STATIC_PAYLOAD_VERSION}&r=${Date.now()}`
    : `/dashboard_payload.static.json?v=${STATIC_PAYLOAD_VERSION}`;
  const hostname = String(window.location.hostname || "").toLowerCase();
  const isLocalhost = hostname === "127.0.0.1" || hostname === "localhost";

  // Local development prefers live API; hosted demos prefer static payload first.
  return isLocalhost ? [apiEndpoint, staticEndpoint] : [staticEndpoint, apiEndpoint];
}

async function fetchDashboardPayload(refresh = false) {
  const endpoints = getDashboardEndpoints(refresh);
  let lastError = null;

  for (const endpoint of endpoints) {
    try {
      const response = await fetch(endpoint, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status} (${endpoint})`);
      }
      const payload = await response.json();
      if (!payload || !Array.isArray(payload.crops) || payload.crops.length === 0) {
        throw new Error(`No crop data in payload (${endpoint})`);
      }
      return { payload, endpoint };
    } catch (error) {
      lastError = error;
    }
  }

  throw (lastError || new Error("No dashboard endpoint is available."));
}

async function fetchDashboard(refresh = false) {
  try {
    setLoadingState(refresh ? "Refreshing model outputs..." : "Loading dashboard data...");
    const { payload, endpoint } = await fetchDashboardPayload(refresh);

    state.dashboard = payload;
    state.thresholds = payload.thresholds || state.thresholds;
    state.cropIndex = 0;
    state.seriesVisible.actual = true;
    state.seriesVisible.predicted = true;
    state.modelSeriesVisible = {};

    updateCropSelect();
    updateLegendControls();
    updateRuleHint();

    const firstCrop = payload.crops[0];
    state.timeIndex = Math.max(0, firstCrop.timestamps.length - 1);
    state.windowStart = Math.max(0, firstCrop.timestamps.length - getWindowSize(firstCrop.timestamps.length));

    hideTooltip();
    renderCurrentStep();
    if (refresh) {
      const loadedFromStatic = endpoint.includes("dashboard_payload.static.json");
      showToast(loadedFromStatic ? "Loaded static dashboard snapshot." : "Dashboard data refreshed.");
    }
  } catch (error) {
    stopPlayback();
    setLoadingState(`Dashboard load failed: ${error.message}`);
    showToast(`Load failed: ${error.message}`, "error");
  }
}

els.cropSelect.addEventListener("change", onCropChange);
els.playBtn.addEventListener("click", togglePlayback);
els.refreshBtn.addEventListener("click", () => fetchDashboard(true));

els.windowScroll.addEventListener("input", () => {
  const crop = getCurrentCrop();
  if (!crop) return;

  stopPlayback();
  hideTooltip();
  state.windowStart = Number(els.windowScroll.value);
  syncWindowController(crop);
  drawCurrentChart();
});

els.timeSlider.addEventListener("input", () => {
  stopPlayback();
  hideTooltip();
  state.timeIndex = Number(els.timeSlider.value);
  renderCurrentStep();
});

els.legendActual.addEventListener("click", () => {
  state.seriesVisible.actual = !state.seriesVisible.actual;
  updateLegendControls();
  drawCurrentChart();
});

els.legendPredicted.addEventListener("click", () => {
  state.seriesVisible.predicted = !state.seriesVisible.predicted;
  updateLegendControls();
  drawCurrentChart();
});

els.chart.addEventListener("mousemove", (event) => {
  const crop = getCurrentCrop();
  if (!crop) return;

  const rect = els.chart.getBoundingClientRect();
  const mouseX = event.clientX - rect.left;
  const nearest = findNearestPlotPoint(mouseX);

  if (!nearest) {
    hideTooltip();
    drawCurrentChart();
    return;
  }

  state.hoverIndex = nearest.idx;
  showTooltip(nearest);
  drawCurrentChart();
});

els.chart.addEventListener("mouseleave", () => {
  hideTooltip();
  drawCurrentChart();
});

if (els.modelChart) {
  els.modelChart.addEventListener("mousemove", (event) => {
    const crop = getCurrentCrop();
    if (!crop) return;

    const rect = els.modelChart.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const nearest = findNearestModelPlotPoint(mouseX);

    if (!nearest) {
      hideModelTooltip();
      drawCurrentChart();
      return;
    }

    state.modelHoverIndex = nearest.idx;
    showModelTooltip(nearest);
    drawCurrentChart();
  });

  els.modelChart.addEventListener("mouseleave", () => {
    hideModelTooltip();
    drawCurrentChart();
  });
}

window.addEventListener("resize", () => {
  drawCurrentChart();
});

fetchDashboard(false);
