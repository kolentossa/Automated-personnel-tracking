// 页面脚本：启动、刷新人数、保存计数设置、绘制计数线。
// 中英文文字单独放在 i18n.js；浏览器只显示结果，不计算人员轨迹或人数。
import { initLanguage, t } from "./i18n.js?v=20261103d";

const serviceState = document.querySelector("#service-state");
const cameraPill = document.querySelector("#camera-pill");
const resetButton = document.querySelector("#reset-button");
const fields = {
  currentOccupancy: document.querySelector("#current-occupancy"),
  totalEntered: document.querySelector("#total-entered"),
  totalExited: document.querySelector("#total-exited"),
  cameraStatus: document.querySelector("#camera-status"),
  privacy: document.querySelector("#privacy"),
  events: document.querySelector("#events"),
};

let latestStats = null;
let currentStatus = { stats: {}, error: "" };

const saveButton = document.querySelector("#save-counting-button");
const video = document.querySelector("#video-stream");
const overlay = document.querySelector("#counting-overlay");
const context = overlay.getContext("2d");
const directionButtons = Array.from(document.querySelectorAll(".direction-option"));
const anchorInput = document.querySelector("#counting-anchor");
const deadZoneInput = document.querySelector("#counting-inner");
const confirmationInput = document.querySelector("#counting-outer");
const warning = document.querySelector("#counting-warning");
const lineReadout = document.querySelector("#counting-line");
const countingStatus = document.querySelector("#counting-config-state");
const settingsPanel = document.querySelector("#counting-settings");
let countingMessageKey = "";
let countingMessageDetail = "";
let savingCountingSettings = false;

const countingSettings = {
  line: null,
  direction: "left_to_right",
  frameSize: { width: 1280, height: 720 },
  draftStart: null,
  anchor: "auto",
  deadZone: 64,
  confirmationDistance: 96,
};

// 1. 页面启动：把按钮和事件连到下面的普通函数，然后开始每秒刷新。
function startPage() {
  resetButton.addEventListener("click", resetStats);
  document.addEventListener("languagechange", refreshLanguage);
  initLanguage();

  overlay.addEventListener("click", selectLinePoint);
  for (const button of directionButtons) {
    button.addEventListener("click", () => changeDirection(button.dataset.direction));
  }
  saveButton.addEventListener("click", () => saveConfig());
  for (const input of [anchorInput, deadZoneInput, confirmationInput]) {
    input.addEventListener("input", updateDraftSettings);
  }

  video.addEventListener("load", drawOverlay);
  window.addEventListener("resize", drawOverlay);
  if ("ResizeObserver" in window) {
    new ResizeObserver(drawOverlay).observe(video);
  }

  loadConfig();
  refreshStats();
  setInterval(refreshStats, 1000);
}

startPage();

// 2. 人数与状态：请求后台快照，再把文字填入页面。

async function getJson(path, options = {}) {
  const response = await fetch(path, { cache: "no-store", ...options });
  if (!response.ok) {
    const problem = await response.json().catch(() => ({}));
    throw new Error(problem.detail || `${path} ${response.status}`);
  }
  return response.json();
}

async function refreshStats() {
  try {
    renderStats(await getJson("/api/stats"));
  } catch (error) {
    setStatus({}, error.message);
  }
}

function renderStats(stats) {
  latestStats = stats;
  fields.currentOccupancy.textContent = stats.current_occupancy ?? 0;
  fields.totalEntered.textContent = stats.total_entered ?? 0;
  fields.totalExited.textContent = stats.total_exited ?? 0;
  fields.privacy.textContent = t(stats.face_mosaic_enabled ? "privacy_on" : "privacy_off");
  renderEvents(stats.recent_events || []);
  setStatus(stats);
}

function setStatus(stats, error = "") {
  currentStatus = { stats, error };
  const cameraStatus = String(stats.camera_status || "offline");
  const label = t(cameraStatus);
  fields.cameraStatus.textContent = label;
  cameraPill.textContent = label;
  cameraPill.classList.toggle("online", cameraStatus === "online");
  cameraPill.classList.toggle("error", cameraStatus === "error");
  serviceState.textContent = error ? t("service_unavailable") : stats.last_error || t("service_running");
}

function renderEvents(events) {
  fields.events.replaceChildren();
  if (!events.length) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = t("no_events");
    fields.events.append(empty);
    return;
  }

  for (const event of events) {
    const item = document.createElement("li");
    const time = document.createElement("span");
    time.className = "event-time";
    time.textContent = event.time || "--:--:--";
    const direction = document.createElement("span");
    direction.className = `event-type ${String(event.event_type || "").toLowerCase()}`;
    const eventLabel = { ENTER: "enter_event", EXIT: "exit_event" }[event.event_type];
    direction.textContent = eventLabel ? t(eventLabel) : event.event_type || t("event");
    const detail = document.createElement("span");
    detail.className = "event-detail";
    detail.textContent = t("event_detail", { id: event.track_id, count: event.current_occupancy });
    // 事件是文本数据，用 textContent 避免把后端内容当作 HTML 执行。
    item.append(time, direction, detail);
    fields.events.append(item);
  }
}

async function resetStats() {
  resetButton.disabled = true;
  try {
    renderStats(await getJson("/api/reset-stats", { method: "POST" }));
  } catch (error) {
    setStatus({}, error.message);
  } finally {
    resetButton.disabled = false;
  }
}

function refreshLanguage() {
  // 只更新文字，不重新读取或保存配置，保留断线前的统计和未保存的画线草稿。
  if (latestStats) {
    fields.privacy.textContent = t(latestStats.face_mosaic_enabled ? "privacy_on" : "privacy_off");
    renderEvents(latestStats.recent_events || []);
  }
  if (latestStats || currentStatus.error) setStatus(currentStatus.stats, currentStatus.error);
  renderSettings(countingMessageKey, countingMessageDetail);
}

// 3. 计数设置：读取、验证和保存；草稿始终使用原图坐标。

async function loadConfig() {
  try {
    applyConfig(await getJson("/api/config/counting"));
  } catch (error) {
    setMessage("config_unavailable", error.message);
  }
}

function applyConfig(config, message = "") {
  countingSettings.line = normalizeLine(config.line);
  countingSettings.direction = config.direction || "left_to_right";
  countingSettings.frameSize = normalizeFrameSize(config.frame_size);
  countingSettings.anchor = config.anchor || "auto";
  countingSettings.deadZone = Number(config.hysteresis_px ?? 64);
  countingSettings.confirmationDistance = Number(config.rearm_distance_px ?? 96);
  anchorInput.value = countingSettings.anchor;
  deadZoneInput.value = countingSettings.deadZone;
  confirmationInput.value = countingSettings.confirmationDistance;
  countingSettings.draftStart = null;
  renderSettings(message);
}

async function saveConfig(message = "settings_saved") {
  if (!countingSettings.line || !validDistances()) return;
  savingCountingSettings = true;
  saveButton.disabled = true;
  try {
    const payload = { line: countingSettings.line, direction: countingSettings.direction,
      anchor: countingSettings.anchor, hysteresis_px: countingSettings.deadZone, rearm_distance_px: countingSettings.confirmationDistance };
    const updated = await getJson("/api/config/counting", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    applyConfig(updated, message);
  } catch (error) {
    setMessage("save_failed", error.message);
  } finally {
    savingCountingSettings = false;
    saveButton.disabled = !validDistances();
  }
}

function selectLinePoint(event) {
  settingsPanel.open = true;
  const point = eventToFramePoint(event);
  if (!countingSettings.draftStart) {
    countingSettings.draftStart = point;
    renderSettings("start_selected");
    return;
  }
  countingSettings.line = {
    x1: countingSettings.draftStart.x,
    y1: countingSettings.draftStart.y,
    x2: point.x,
    y2: point.y,
  };
  countingSettings.draftStart = null;
  renderSettings("line_selected");
}

async function changeDirection(direction) {
  countingSettings.direction = direction;
  renderSettings("saving_direction");
  // 与原界面一致：方向切换立即保存，其他设置由保存按钮提交。
  await saveConfig("direction_saved");
}

function updateDraftSettings() {
  countingSettings.anchor = anchorInput.value;
  countingSettings.deadZone = Number(deadZoneInput.value);
  countingSettings.confirmationDistance = Number(confirmationInput.value);
  renderSettings("unsaved_changes");
}

function renderSettings(message = "", detail = "") {
  const horizontal = isHorizontalLine();
  const forward = ["left_to_right", "top_to_bottom"].includes(countingSettings.direction);
  const modes = horizontal ? ["top_to_bottom", "bottom_to_top"] : ["left_to_right", "right_to_left"];
  countingSettings.direction = modes[forward ? 0 : 1];
  directionButtons.forEach((button, index) => {
    button.dataset.direction = modes[index];
    button.textContent = t(modes[index]);
  });
  anchorInput.options[0].textContent = t(horizontal ? "auto_foot" : "auto_center");
  const invalid = !validDistances();
  warning.textContent = invalid ? t("invalid_distances")
    : confirmationOutsideImage() ? t("outside_image") : "";
  saveButton.disabled = savingCountingSettings || invalid;
  if (countingSettings.line) {
    const { x1, y1, x2, y2 } = countingSettings.line;
    lineReadout.textContent = `(${roundCoord(x1)},${roundCoord(y1)})-(${roundCoord(x2)},${roundCoord(y2)})`;
  } else {
    lineReadout.textContent = "-";
  }
  for (const button of directionButtons) {
    const active = button.dataset.direction === countingSettings.direction;
    button.classList.toggle("active", active);
    button.setAttribute("aria-checked", active ? "true" : "false");
  }
  setMessage(message, detail);
  drawOverlay();
}

function setMessage(key, detail = "") {
  countingMessageKey = key;
  countingMessageDetail = detail;
  countingStatus.textContent = key ? t(key, { detail }) : "";
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
  const width = Number(frameSize?.width) || countingSettings.frameSize.width || 1280;
  const height = Number(frameSize?.height) || countingSettings.frameSize.height || 720;
  return { width, height };
}

function validDistances() {
  return Number.isFinite(countingSettings.deadZone) && Number.isFinite(countingSettings.confirmationDistance)
    && countingSettings.deadZone >= 1 && countingSettings.confirmationDistance > countingSettings.deadZone;
}

function roundCoord(value) {
  return Math.round(Number(value) || 0);
}

// 4. 视频画线：把原图坐标换算到画布，绘制中心线、确认区域和进入箭头。

function drawOverlay() {
  resizeOverlay();
  const rect = overlay.getBoundingClientRect();
  context.clearRect(0, 0, rect.width, rect.height);
  if (countingSettings.line) {
    drawConfirmationZones(rect);
    const start = frameToCanvas(countingSettings.line.x1, countingSettings.line.y1, rect);
    const end = frameToCanvas(countingSettings.line.x2, countingSettings.line.y2, rect);
    context.strokeStyle = "#ffd43b";
    context.lineWidth = 3;
    context.beginPath();
    context.moveTo(start.x, start.y);
    context.lineTo(end.x, end.y);
    context.stroke();
    drawEnterArrow(rect);
    drawHandle(start);
    drawHandle(end);
  }
  if (countingSettings.draftStart) {
    drawHandle(frameToCanvas(countingSettings.draftStart.x, countingSettings.draftStart.y, rect), "#ffffff");
  }
}

function drawConfirmationZones(rect) {
  const geometry = lineGeometry();
  if (!geometry || !validDistances()) return;
  const { width, height } = countingSettings.frameSize;
  const extension = Math.hypot(width, height) * 2;
  const length = Math.hypot(geometry.dx, geometry.dy);
  // 延长到画面以外，再沿法向量偏移，保持防抖区与确认线的原有几何规则。
  const ends = [[geometry.x1 - extension * geometry.dx / length, geometry.y1 - extension * geometry.dy / length],
    [geometry.x2 + extension * geometry.dx / length, geometry.y2 + extension * geometry.dy / length]];
  const offset = (point, distance) => frameToCanvas(point[0] + distance * geometry.nx, point[1] + distance * geometry.ny, rect);
  const corners = [offset(ends[0], countingSettings.deadZone), offset(ends[1], countingSettings.deadZone),
    offset(ends[1], -countingSettings.deadZone), offset(ends[0], -countingSettings.deadZone)];
  context.save();
  context.fillStyle = "rgba(255, 212, 59, 0.09)";
  context.beginPath();
  corners.forEach((p, i) => i ? context.lineTo(p.x, p.y) : context.moveTo(p.x, p.y));
  context.closePath();
  context.fill();
  context.strokeStyle = "#67e8f9";
  context.lineWidth = 1.5;
  context.setLineDash([7, 5]);
  for (const distance of [-countingSettings.confirmationDistance, countingSettings.confirmationDistance]) {
    const a = offset(ends[0], distance);
    const b = offset(ends[1], distance);
    context.beginPath();
    context.moveTo(a.x, a.y);
    context.lineTo(b.x, b.y);
    context.stroke();
  }
  context.restore();
}

function drawEnterArrow(rect) {
  const geometry = lineGeometry();
  if (!geometry) return;
  const desired = { left_to_right: [1, 0], right_to_left: [-1, 0], top_to_bottom: [0, 1], bottom_to_top: [0, -1] }[countingSettings.direction];
  const sign = geometry.nx * desired[0] + geometry.ny * desired[1] >= 0 ? 1 : -1;
  const centre = frameToCanvas((geometry.x1 + geometry.x2) / 2, (geometry.y1 + geometry.y2) / 2, rect);
  const vx = sign * geometry.nx * rect.width / countingSettings.frameSize.width;
  const vy = sign * geometry.ny * rect.height / countingSettings.frameSize.height;
  const length = Math.hypot(vx, vy);
  const ux = vx / length;
  const uy = vy / length;
  const end = { x: centre.x + ux * 26, y: centre.y + uy * 26 };
  context.save();
  context.strokeStyle = "#67e8f9";
  context.lineWidth = 3;
  context.beginPath();
  context.moveTo(centre.x - ux * 20, centre.y - uy * 20);
  context.lineTo(end.x, end.y);
  context.moveTo(end.x - ux * 10 - uy * 6, end.y - uy * 10 + ux * 6);
  context.lineTo(end.x, end.y);
  context.lineTo(end.x - ux * 10 + uy * 6, end.y - uy * 10 - ux * 6);
  context.stroke();
  context.restore();
}

function drawHandle(point, fill = "#ffd43b") {
  context.fillStyle = fill;
  context.strokeStyle = "#1f272d";
  context.lineWidth = 2;
  context.beginPath();
  context.arc(point.x, point.y, 6, 0, Math.PI * 2);
  context.fill();
  context.stroke();
}

function isHorizontalLine() {
  const line = countingSettings.line;
  return line && Math.abs(line.x2 - line.x1) > Math.abs(line.y2 - line.y1);
}

function lineGeometry() {
  const line = countingSettings.line;
  if (!line) return null;
  const dx = line.x2 - line.x1;
  const dy = line.y2 - line.y1;
  const length = Math.hypot(dx, dy);
  if (!length) return null;
  // (nx, ny) 是垂直于计数线的单位向量，用来计算两侧距离。
  return { ...line, dx, dy, nx: -dy / length, ny: dx / length };
}

function confirmationOutsideImage() {
  const geometry = lineGeometry();
  if (!geometry) return false;
  const { width, height } = countingSettings.frameSize;
  const distances = [[0, 0], [width, 0], [0, height], [width, height]]
    .map(([x, y]) => (x - geometry.x1) * geometry.nx + (y - geometry.y1) * geometry.ny);
  return Math.min(...distances) >= -countingSettings.confirmationDistance || Math.max(...distances) <= countingSettings.confirmationDistance;
}

function resizeOverlay() {
  const rect = video.getBoundingClientRect();
  const pixelRatio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * pixelRatio));
  const height = Math.max(1, Math.round(rect.height * pixelRatio));
  if (overlay.width !== width || overlay.height !== height) {
    overlay.width = width;
    overlay.height = height;
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  }
}

function frameToCanvas(x, y, rect) {
  return {
    x: (Number(x) / countingSettings.frameSize.width) * rect.width,
    y: (Number(y) / countingSettings.frameSize.height) * rect.height,
  };
}

function eventToFramePoint(event) {
  const rect = overlay.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * countingSettings.frameSize.width;
  const y = ((event.clientY - rect.top) / rect.height) * countingSettings.frameSize.height;
  return { x: clamp(x, 0, countingSettings.frameSize.width), y: clamp(y, 0, countingSettings.frameSize.height) };
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}
