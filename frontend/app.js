const serviceState = document.querySelector("#service-state");
const cameraPill = document.querySelector("#camera-pill");
const fields = {
  currentPeople: document.querySelector("#current-people"),
  cameraStatus: document.querySelector("#camera-status"),
  fps: document.querySelector("#fps"),
  frames: document.querySelector("#frames"),
  entered: document.querySelector("#entered"),
  exited: document.querySelector("#exited"),
  events: document.querySelector("#events"),
};

async function getJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`${path} ${response.status}`);
  }
  return response.json();
}

function formatTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "--:--";
  }
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function setCameraStatus(status) {
  const label = status ? String(status) : "offline";
  const pretty = label.charAt(0).toUpperCase() + label.slice(1);
  fields.cameraStatus.textContent = pretty;
  cameraPill.textContent = pretty;
  cameraPill.classList.toggle("online", label === "online");
}

function renderEvents(items) {
  fields.events.innerHTML = "";
  if (!items.length) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = "No events yet";
    fields.events.appendChild(empty);
    return;
  }
  for (const event of items) {
    const item = document.createElement("li");
    const type = String(event.event_type || "").toLowerCase();
    item.innerHTML = `
      <span class="time">${formatTime(event.timestamp)}</span>
      <span class="type ${type}">${event.event_type || "EVENT"}</span>
      <span class="track">ID ${event.tracking_id}</span>
    `;
    fields.events.appendChild(item);
  }
}

async function refresh() {
  try {
    const [status, statistics, events] = await Promise.all([
      getJson("/status"),
      getJson("/statistics"),
      getJson("/events?limit=10"),
    ]);
    fields.currentPeople.textContent = status.current_people ?? 0;
    fields.fps.textContent = status.fps ?? 0;
    fields.frames.textContent = status.frames_processed ?? 0;
    fields.entered.textContent = statistics.today_entered ?? 0;
    fields.exited.textContent = statistics.today_exited ?? 0;
    setCameraStatus(status.camera_status);
    serviceState.textContent = status.last_error ? status.last_error : "Local service running";
    renderEvents(events.events || []);
  } catch (error) {
    serviceState.textContent = "Service unavailable";
    setCameraStatus("offline");
  }
}

refresh();
setInterval(refresh, 2000);
