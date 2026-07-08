const statusEl = document.querySelector("#service-state");
const cameraPill = document.querySelector("#camera-pill");
const resetButton = document.querySelector("#reset-button");
const fields = {
  currentOccupancy: document.querySelector("#current-occupancy"),
  totalEntered: document.querySelector("#total-entered"),
  totalExited: document.querySelector("#total-exited"),
  activeTracks: document.querySelector("#active-tracks"),
  fps: document.querySelector("#fps"),
  latencyMs: document.querySelector("#latency-ms"),
  cameraStatus: document.querySelector("#camera-status"),
  source: document.querySelector("#source"),
  detector: document.querySelector("#detector"),
  privacy: document.querySelector("#privacy"),
  events: document.querySelector("#events"),
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

function renderStats(stats) {
  fields.currentOccupancy.textContent = stats.current_occupancy ?? 0;
  fields.totalEntered.textContent = stats.total_entered ?? 0;
  fields.totalExited.textContent = stats.total_exited ?? 0;
  fields.activeTracks.textContent = stats.active_tracks ?? 0;
  fields.fps.textContent = stats.fps ?? 0;
  fields.latencyMs.textContent = stats.latency_ms ?? 0;
  fields.source.textContent = stats.source || "-";
  fields.detector.textContent = stats.detector || "-";
  fields.privacy.textContent = stats.face_mosaic_enabled ? "ON - face/head mosaic" : "OFF";
  renderEvents(stats.recent_events || []);
  setStatus(stats);
}

async function refreshStats() {
  try {
    renderStats(await getJson("/api/stats"));
  } catch (error) {
    setStatus({}, error.message);
  }
}

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

refreshStats();
setInterval(refreshStats, 1000);