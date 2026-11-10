// 页面只支持两种语言；文字集中放在这里，切换时不改业务状态或计数配置。
const messages = {
  en: {
    page_title: "Personnel tracking · RK3588",
    eyebrow: "RK3588 · Local personnel tracking",
    title: "Personnel tracking",
    language: "Language",
    connecting_service: "Connecting to service",
    connecting: "Connecting",
    counts_label: "Entry and exit counts",
    current: "Current occupancy",
    entered: "Total entered",
    exited: "Total exited",
    people: "people",
    live_video: "Live view",
    local_processing: "Local processing",
    video_alt: "Live camera view with anonymous person tracking",
    overlay_label: "Counting line editor: select two points on the video",
    privacy_loading: "Reading privacy settings",
    temporary_ids: "IDs are temporary tracking labels",
    recent_events: "Recent crossings",
    latest_records: "Latest events",
    no_events: "No crossing events yet",
    counting_settings: "Counting settings",
    counting_help: "Select two points on the video, then save the line. Changing the entry direction saves immediately.",
    counting_line: "Counting line",
    entry_direction: "Entry direction",
    left_to_right: "Left → Right",
    right_to_left: "Right → Left",
    top_to_bottom: "Top → Bottom",
    bottom_to_top: "Bottom → Top",
    anchor: "Counting point",
    auto: "Auto",
    center: "Box centre",
    bottom_center: "Foot point",
    auto_center: "Auto (box centre)",
    auto_foot: "Auto (foot point)",
    dead_zone: "Dead zone (px)",
    confirmation: "Confirmation (px)",
    save: "Save settings",
    reset_help: "Reset clears counts and recent events. The video keeps running.",
    reset: "Reset counts",
    online: "Online",
    offline: "Offline",
    error: "Error",
    service_unavailable: "Service connection lost",
    service_running: "Local service running",
    enter_event: "Entered",
    exit_event: "Exited",
    event: "Event",
    event_detail: "ID {id} · Current occupancy: {count}",
    privacy_on: "Face / head masking enabled",
    privacy_off: "Privacy masking disabled",
    invalid_distances: "Confirmation must exceed the dead zone; the dead zone must be at least 1 px.",
    outside_image: "One confirmation zone is outside the image",
    config_unavailable: "Could not load counting settings: {detail}",
    settings_saved: "Settings saved",
    save_failed: "Could not save settings: {detail}",
    start_selected: "Start point selected. Select the end point.",
    line_selected: "Line selected. Save to apply it.",
    saving_direction: "Saving entry direction",
    direction_saved: "Entry direction saved",
    unsaved_changes: "Unsaved changes",
  },
  zh: {
    page_title: "人员出入统计 · RK3588",
    eyebrow: "RK3588 · 本地人员跟踪",
    title: "人员出入统计",
    language: "语言",
    connecting_service: "正在连接服务",
    connecting: "正在连接",
    counts_label: "出入人数",
    current: "当前人数",
    entered: "累计进入",
    exited: "累计离开",
    people: "人",
    live_video: "实时画面",
    local_processing: "本地处理",
    video_alt: "摄像头实时人员跟踪画面",
    overlay_label: "计数线编辑区，点击两个位置选择端点",
    privacy_loading: "正在读取隐私设置",
    temporary_ids: "编号仅用于临时跟踪",
    recent_events: "最近出入",
    latest_records: "最新记录",
    no_events: "暂无出入记录",
    counting_settings: "计数设置",
    counting_help: "在画面上点击两个位置确定计数线，然后保存。切换进入方向会立即保存。",
    counting_line: "计数线",
    entry_direction: "进入方向",
    left_to_right: "左 → 右",
    right_to_left: "右 → 左",
    top_to_bottom: "上 → 下",
    bottom_to_top: "下 → 上",
    anchor: "计数参考点",
    auto: "自动",
    center: "人体框中心",
    bottom_center: "脚部位置",
    auto_center: "自动（人体框中心）",
    auto_foot: "自动（脚部位置）",
    dead_zone: "缓冲距离（像素）",
    confirmation: "确认距离（像素）",
    save: "保存设置",
    reset_help: "重置会清空人数和出入记录，视频继续运行。",
    reset: "重置统计",
    online: "在线",
    offline: "离线",
    error: "异常",
    service_unavailable: "服务连接中断",
    service_running: "本地服务运行中",
    enter_event: "进入",
    exit_event: "离开",
    event: "事件",
    event_detail: "编号 {id} · 当前 {count} 人",
    privacy_on: "人脸 / 头部遮挡已开启",
    privacy_off: "画面遮挡未开启",
    invalid_distances: "确认距离须大于缓冲距离，缓冲距离至少为 1 像素。",
    outside_image: "有一侧确认区域超出画面",
    config_unavailable: "无法读取计数设置：{detail}",
    settings_saved: "设置已保存",
    save_failed: "保存失败：{detail}",
    start_selected: "已选起点，请再点击一次选择终点",
    line_selected: "计数线已选好，点击保存后生效",
    saving_direction: "正在保存进入方向",
    direction_saved: "进入方向已保存",
    unsaved_changes: "有尚未保存的修改",
  },
};

const storageKey = "person-tracking-language";
let language = "en";
try {
  if (localStorage.getItem(storageKey) === "zh") language = "zh";
} catch {
  // 浏览器不允许本地存储时仍默认英文，并允许当前页面切换语言。
}

export function t(key, values = {}) {
  const text = messages[language][key] ?? messages.en[key] ?? key;
  return text.replace(/\{(\w+)\}/g, (match, name) => values[name] ?? match);
}

function applyLanguage() {
  document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
  document.querySelector("#language-select").value = language;
  for (const element of document.querySelectorAll("[data-i18n]")) {
    element.textContent = t(element.dataset.i18n);
  }
  for (const element of document.querySelectorAll("[data-i18n-aria]")) {
    element.setAttribute("aria-label", t(element.dataset.i18nAria));
  }
  for (const element of document.querySelectorAll("[data-i18n-alt]")) {
    element.alt = t(element.dataset.i18nAlt);
  }
}

export function initLanguage() {
  applyLanguage();
  document.querySelector("#language-select").addEventListener("change", (event) => {
    language = event.target.value === "zh" ? "zh" : "en";
    try {
      localStorage.setItem(storageKey, language);
    } catch {
      // 偏好无法保存不影响界面切换。
    }
    applyLanguage();
    document.dispatchEvent(new Event("languagechange"));
  });
}
