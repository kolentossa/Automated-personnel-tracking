const statusEl = document.querySelector("#service-state");
const cameraPill = document.querySelector("#camera-pill");
const resetButton = document.querySelector("#reset-button");
const saveCountingButton = document.querySelector("#save-counting-button");
const videoEl = document.querySelector("#video-stream");
const overlayEl = document.querySelector("#counting-overlay");
const overlayCtx = overlayEl.getContext("2d");
const directionButtons = Array.from(document.querySelectorAll(".direction-option"));

const fields = {
  currentOccupancy: document.querySelector("#current-occupancy"),
  totalEntered: document.querySelector("#total-entered"),
  totalExited: document.querySelector("#total-exited"),
  activeTracks: document.querySelector("#active-tracks"),
  fps: document.querySelector("#fps"),
  totalLatencyMs: document.querySelector("#total-latency-ms"),
  inferenceMs: document.querySelector("#inference-ms"),
  behaviorAlerts: document.querySelector("#behavior-alerts"),
  cameraStatus: document.querySelector("#camera-status"),
  source: document.querySelector("#source"),
  detector: document.querySelector("#detector"),
  npuEnabled: document.querySelector("#npu-enabled"),
  privacy: document.querySelector("#privacy"),
  cigaretteCandidates: document.querySelector("#cigarette-candidates"),
  behaviorStatus: document.querySelector("#behavior-status"),
  phoneDetector: document.querySelector("#phone-detector"),
  smokingDetector: document.querySelector("#smoking-detector"),
  events: document.querySelector("#events"),
  behaviorEvents: document.querySelector("#behavior-events"),
  countingLine: document.querySelector("#counting-line"),
  countingState: document.querySelector("#counting-config-state"),
};

const countingConfig = {
  line: null,
  direction: "left_to_right",
  frameSize: { width: 1280, height: 720 },
  draftStart: null,
};

async function getJson(path, options = {}) {
  const response = await fetch(path, { cache: "no-store", ...options });
  if (!response.ok) throw new Error(`${path} ${response.status}`);
  return response.json();
}

function setStatus(stats, error = "") {
  const cameraStatus = String(stats.camera_status || "offline");
  fields.cameraStatus.textContent = cameraStatus;
  cameraPill.textContent = cameraStatus.charAt(0).toUpperCase() + cameraStatus.slice(1);
  cameraPill.classList.toggle("online", cameraStatus === "online");
  cameraPill.classList.toggle("error", cameraStatus === "error");
  if (error) statusEl.textContent = "Service unavailable";
  else if (stats.last_error) statusEl.textContent = stats.last_error;
  else statusEl.textContent = "Local service running";
}

function renderEvents(events) {
  fields.events.innerHTML = "";
  if (!events.length) {
    const item = document.createElement("li");
    item.className = "empty";
    item.textContent = "No crossing events yet";
    fields.events.appendChild(item);
    return;
  }
  for (const event of events) {
    const item = document.createElement("li");
    const type = String(event.event_type || "").toLowerCase();
    item.innerHTML = `
      <span class="event-time">${event.time || "--:--:--"}</span>
      <span class="event-type ${type}">${event.event_type || "EVENT"}</span>
      <span class="event-detail">ID ${event.track_id} | now ${event.current_occupancy}</span>
    `;
    fields.events.appendChild(item);
  }
}

function renderBehaviorEvents(events) {
  fields.behaviorEvents.innerHTML = "";
  if (!events.length) {
    const item = document.createElement("li");
    item.className = "empty";
    item.textContent = "No behavior alerts yet";
    fields.behaviorEvents.appendChild(item);
    return;
  }
  for (const event of events.slice(0, 10)) {
    const item = document.createElement("li");
    const timestamp = event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : "--:--:--";
    const type = String(event.event_type || "event").replaceAll("_", " ");
    item.innerHTML = `
      <span class="event-time">${timestamp}</span>
      <span class="event-type behavior">${type}</span>
      <span class="event-detail">ID ${event.track_id ?? "-"} | ${Math.round((event.confidence || 0) * 100)}%</span>
    `;
    fields.behaviorEvents.appendChild(item);
  }
}

function renderStats(stats) {
  fields.currentOccupancy.textContent = stats.current_occupancy ?? 0;
  fields.totalEntered.textContent = stats.total_entered ?? 0;
  fields.totalExited.textContent = stats.total_exited ?? 0;
  fields.activeTracks.textContent = stats.active_tracks ?? 0;
  fields.fps.textContent = stats.fps ?? 0;
  fields.totalLatencyMs.textContent = stats.total_latency_ms ?? stats.latency_ms ?? 0;
  fields.inferenceMs.textContent = stats.inference_ms ?? 0;
  const behaviorCounts = stats.behavior_event_counts || {};
  fields.behaviorAlerts.textContent = Object.values(behaviorCounts).reduce((sum, value) => sum + Number(value || 0), 0);
  fields.source.textContent = stats.source || "-";
  fields.detector.textContent = stats.detector || "-";
  fields.npuEnabled.textContent = stats.npu_enabled ? "ON" : "OFF";
  fields.privacy.textContent = stats.privacy_mode === "no_mosaic" ? "NO MOSAIC / UNREDACTED" : "UNKNOWN";
  fields.cigaretteCandidates.textContent = `${stats.raw_cigarette_candidates || 0} raw / ${stats.verified_cigarette_candidates || 0} verified`;
  fields.behaviorStatus.textContent = stats.behavior_status || "-";
  fields.phoneDetector.textContent = stats.phone_detection_available ? "READY" : "UNAVAILABLE";
  fields.smokingDetector.textContent = stats.smoking_detection_available ? "READY" : "MODEL REQUIRED";
  renderEvents(stats.recent_events || []);
  renderBehaviorEvents(stats.behavior_events || []);
  setStatus(stats);
}

async function refreshStats() {
  try {
    renderStats(await getJson("/api/stats"));
  } catch (error) {
    setStatus({}, error.message);
  }
}

function applyCountingConfig(config) {
  countingConfig.line = normalizeLine(config.line);
  countingConfig.direction = config.direction || "left_to_right";
  countingConfig.frameSize = normalizeFrameSize(config.frame_size);
  countingConfig.draftStart = null;
  renderCountingConfig();
}

function normalizeLine(line) {
  if (!line) return null;
  if (Array.isArray(line) && line.length === 2) {
    return {
      x1: Number(line[0][0]),
      y1: Number(line[0][1]),
      x2: Number(line[1][0]),
      y2: Number(line[1][1]),
    };
  }
  return {
    x1: Number(line.x1),
    y1: Number(line.y1),
    x2: Number(line.x2),
    y2: Number(line.y2),
  };
}

function normalizeFrameSize(frameSize) {
  const width = Number(frameSize?.width) || countingConfig.frameSize.width || 1280;
  const height = Number(frameSize?.height) || countingConfig.frameSize.height || 720;
  return { width, height };
}

function renderCountingConfig(message = "") {
  if (countingConfig.line) {
    const { x1, y1, x2, y2 } = countingConfig.line;
    fields.countingLine.textContent = `(${roundCoord(x1)},${roundCoord(y1)})-(${roundCoord(x2)},${roundCoord(y2)})`;
  } else {
    fields.countingLine.textContent = "-";
  }
  for (const button of directionButtons) {
    const active = button.dataset.direction === countingConfig.direction;
    button.classList.toggle("active", active);
    button.setAttribute("aria-checked", active ? "true" : "false");
  }
  fields.countingState.textContent = message;
  drawCountingOverlay();
}

function roundCoord(value) {
  return Math.round(Number(value) || 0);
}

function drawCountingOverlay() {
  resizeOverlay();
  const rect = overlayEl.getBoundingClientRect();
  overlayCtx.clearRect(0, 0, rect.width, rect.height);
  if (countingConfig.line) {
    const start = frameToCanvas(countingConfig.line.x1, countingConfig.line.y1, rect);
    const end = frameToCanvas(countingConfig.line.x2, countingConfig.line.y2, rect);
    overlayCtx.strokeStyle = "#ffd43b";
    overlayCtx.lineWidth = 3;
    overlayCtx.beginPath();
    overlayCtx.moveTo(start.x, start.y);
    overlayCtx.lineTo(end.x, end.y);
    overlayCtx.stroke();
    drawHandle(start);
    drawHandle(end);
  }
  if (countingConfig.draftStart) {
    drawHandle(frameToCanvas(countingConfig.draftStart.x, countingConfig.draftStart.y, rect), "#ffffff");
  }
}

function drawHandle(point, fill = "#ffd43b") {
  overlayCtx.fillStyle = fill;
  overlayCtx.strokeStyle = "#1f272d";
  overlayCtx.lineWidth = 2;
  overlayCtx.beginPath();
  overlayCtx.arc(point.x, point.y, 6, 0, Math.PI * 2);
  overlayCtx.fill();
  overlayCtx.stroke();
}

function resizeOverlay() {
  const rect = videoEl.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (overlayEl.width !== width || overlayEl.height !== height) {
    overlayEl.width = width;
    overlayEl.height = height;
    overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
}

function frameToCanvas(x, y, rect) {
  return {
    x: (Number(x) / countingConfig.frameSize.width) * rect.width,
    y: (Number(y) / countingConfig.frameSize.height) * rect.height,
  };
}

function eventToFramePoint(event) {
  const rect = overlayEl.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * countingConfig.frameSize.width;
  const y = ((event.clientY - rect.top) / rect.height) * countingConfig.frameSize.height;
  return { x: clamp(x, 0, countingConfig.frameSize.width), y: clamp(y, 0, countingConfig.frameSize.height) };
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

async function loadCountingConfig() {
  try {
    applyCountingConfig(await getJson("/api/config/counting"));
  } catch (error) {
    fields.countingState.textContent = `Config unavailable: ${error.message}`;
  }
}

async function saveCountingConfig(message = "Configuration saved") {
  if (!countingConfig.line) return;
  saveCountingButton.disabled = true;
  try {
    const payload = { line: countingConfig.line, direction: countingConfig.direction };
    const updated = await getJson("/api/config/counting", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    applyCountingConfig(updated);
    renderCountingConfig(message);
  } catch (error) {
    fields.countingState.textContent = `Save failed: ${error.message}`;
  } finally {
    saveCountingButton.disabled = false;
  }
}

overlayEl.addEventListener("click", (event) => {
  const point = eventToFramePoint(event);
  if (!countingConfig.draftStart) {
    countingConfig.draftStart = point;
    renderCountingConfig("Start point selected");
    return;
  }
  countingConfig.line = {
    x1: countingConfig.draftStart.x,
    y1: countingConfig.draftStart.y,
    x2: point.x,
    y2: point.y,
  };
  countingConfig.draftStart = null;
  renderCountingConfig("Line ready to save");
});

for (const button of directionButtons) {
  button.addEventListener("click", async () => {
    countingConfig.direction = button.dataset.direction;
    renderCountingConfig("Saving direction");
    await saveCountingConfig("Direction saved");
  });
}

saveCountingButton.addEventListener("click", async () => {
  await saveCountingConfig();
});

resetButton.addEventListener("click", async () => {
  resetButton.disabled = true;
  try {
    renderStats(await getJson("/api/reset-stats", { method: "POST" }));
  } catch (error) {
    setStatus({}, error.message);
  } finally {
    resetButton.disabled = false;
  }
});

videoEl.addEventListener("load", drawCountingOverlay);
window.addEventListener("resize", drawCountingOverlay);
if ("ResizeObserver" in window) {
  new ResizeObserver(drawCountingOverlay).observe(videoEl);
}

loadCountingConfig();
refreshStats();
setInterval(refreshStats, 1000);
