const api = {
  async get(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
  async post(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
  async delete(url) {
    const res = await fetch(url, { method: "DELETE" });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
};

const state = {
  datasets: [],
  checkpoints: [],
  archives: [],
  items: [],
  datasetId: null,
  itemId: null,
  tool: "brush",
  drawing: false,
  image: null,
  trainingTimer: null,
  activeBatch: null,
  activeQueue: [],
  activeIndex: -1,
  zoom: 1,
  nextCycleJobs: new Set(),
  contextByLabel: {},
  activeModelLabelId: null,
  inference: null,
  modelBlocked: false,
};

const el = {
  health: document.getElementById("health"),
  checkpointSelect: document.getElementById("checkpointSelect"),
  threshold: document.getElementById("threshold"),
  datasetName: document.getElementById("datasetName"),
  defectClass: document.getElementById("defectClass"),
  confirmProjectName: document.getElementById("confirmProjectName"),
  confirmDefectClass: document.getElementById("confirmDefectClass"),
  labelSelect: document.getElementById("labelSelect"),
  labelIdInput: document.getElementById("labelIdInput"),
  labelNameInput: document.getElementById("labelNameInput"),
  labelColorInput: document.getElementById("labelColorInput"),
  createDataset: document.getElementById("createDataset"),
  datasetSelect: document.getElementById("datasetSelect"),
  datasetSummary: document.getElementById("datasetSummary"),
  archiveDataset: document.getElementById("archiveDataset"),
  deleteDataset: document.getElementById("deleteDataset"),
  archiveSelect: document.getElementById("archiveSelect"),
  restoreArchive: document.getElementById("restoreArchive"),
  datasetManageStatus: document.getElementById("datasetManageStatus"),
  rebuildMetadata: document.getElementById("rebuildMetadata"),
  createSnapshot: document.getElementById("createSnapshot"),
  snapshotStatus: document.getElementById("snapshotStatus"),
  exportActive: document.getElementById("exportActive"),
  exportCoco: document.getElementById("exportCoco"),
  exportFiftyOne: document.getElementById("exportFiftyOne"),
  exportStatus: document.getElementById("exportStatus"),
  sourceDir: document.getElementById("sourceDir"),
  importImages: document.getElementById("importImages"),
  importStatus: document.getElementById("importStatus"),
  activeTopK: document.getElementById("activeTopK"),
  batchPredict: document.getElementById("batchPredict"),
  rankActive: document.getElementById("rankActive"),
  nextActive: document.getElementById("nextActive"),
  activeStatus: document.getElementById("activeStatus"),
  activeQueue: document.getElementById("activeQueue"),
  itemList: document.getElementById("itemList"),
  predictMode: document.getElementById("predictMode"),
  predictBtn: document.getElementById("predictBtn"),
  brushBtn: document.getElementById("brushBtn"),
  eraseBtn: document.getElementById("eraseBtn"),
  clearBtn: document.getElementById("clearBtn"),
  saveBtn: document.getElementById("saveBtn"),
  brushSize: document.getElementById("brushSize"),
  zoomOut: document.getElementById("zoomOut"),
  zoomIn: document.getElementById("zoomIn"),
  zoomReset: document.getElementById("zoomReset"),
  zoomValue: document.getElementById("zoomValue"),
  imageCanvas: document.getElementById("imageCanvas"),
  maskCanvas: document.getElementById("maskCanvas"),
  selectedStatus: document.getElementById("selectedStatus"),
  saveStatus: document.getElementById("saveStatus"),
  trainJob: document.getElementById("trainJob"),
  openTraining: document.getElementById("openTraining"),
  openTesting: document.getElementById("openTesting"),
  trainStatus: document.getElementById("trainStatus"),
  trainEpochs: document.getElementById("trainEpochs"),
  trainSamples: document.getElementById("trainSamples"),
  trainBatch: document.getElementById("trainBatch"),
  trainLr: document.getElementById("trainLr"),
};

const imageCtx = el.imageCanvas.getContext("2d");
const maskCtx = el.maskCanvas.getContext("2d");
const MASK_ALPHA = 88;
const DEFAULT_CHECKPOINT_ID = document.body.dataset.defaultCheckpoint || "baseline-spall-unet-recall-v1";
const WORKSPACE_MODE = document.body.dataset.workspace || "optimization";
const pageParams = new URLSearchParams(window.location.search);
const DEFAULT_INFERENCE = {
  threshold: 0.5,
  tile: 512,
  stride: 384,
};

function applyInferenceSettings(settings) {
  state.inference = { ...DEFAULT_INFERENCE, ...(settings || {}) };
  if (el.threshold && !el.threshold.dataset.platformDefaultApplied) {
    el.threshold.value = state.inference.threshold;
    el.threshold.dataset.platformDefaultApplied = "true";
  }
}

function inferenceParams() {
  const settings = state.inference || DEFAULT_INFERENCE;
  return {
    threshold: Number(el.threshold.value || settings.threshold),
    tile: settings.tile,
    stride: settings.stride,
  };
}

function setStatus(message) {
  el.selectedStatus.textContent = message;
}

function selectedCheckpointId() {
  return el.checkpointSelect.value || DEFAULT_CHECKPOINT_ID;
}

function selectedCheckpoint() {
  return state.checkpoints.find((ckpt) => ckpt.id === selectedCheckpointId()) || null;
}

function isFoundationStage() {
  return selectedCheckpoint()?.role === "foundation";
}

function checkpointLabelId(ckpt = selectedCheckpoint()) {
  return ckpt?.label_id || (ckpt?.id === "baseline-spall-unet-recall-v1" ? "spall" : null);
}

function checkpointLabelName(ckpt = selectedCheckpoint()) {
  return ckpt?.label_name || (checkpointLabelId(ckpt) ? checkpointLabelId(ckpt) : "未绑定类别");
}

function checkpointLabelColor(ckpt = selectedCheckpoint()) {
  return ckpt?.label_color || (checkpointLabelId(ckpt) === "spall" ? "#ff3728" : "#ff9e94");
}

function selectedLabelId() {
  return el.labelSelect?.value || "spall";
}

function currentDataset() {
  return state.datasets.find((ds) => ds.id === state.datasetId) || null;
}

function currentDatasetLabel() {
  const ds = currentDataset();
  return ds?.labels?.[0] || null;
}

function datasetLabelId(ds = currentDataset()) {
  return ds?.labels?.[0]?.id || null;
}

function contextLabelId() {
  return currentDatasetLabel()?.id || checkpointLabelId() || el.labelIdInput?.value || selectedLabelId();
}

function syncDatasetForm(meta) {
  if (!meta) return;
  if (el.datasetName) el.datasetName.value = meta.name || "";
  if (el.defectClass) el.defectClass.value = meta.defect_class || "";
  const label = meta.labels?.[0];
  if (label) {
    if (el.labelIdInput) el.labelIdInput.value = label.id || "";
    if (el.labelNameInput) el.labelNameInput.value = label.name || label.id || "";
    if (el.labelColorInput) el.labelColorInput.value = label.color || "#ff9e94";
  }
}

function selectedLabelColor() {
  const option = el.labelSelect?.selectedOptions?.[0];
  return option?.dataset?.color || el.labelColorInput?.value || "#ff9e94";
}

function slugLabel(text) {
  return String(text || "defect")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "") || "defect";
}

function randomLabelColor() {
  const palette = [
    "#ff9e94",
    "#7cc7ff",
    "#91d8a4",
    "#f2c66d",
    "#c7a6ff",
    "#68d6c3",
    "#ffb0cf",
    "#b7d77a",
  ];
  const used = new Set(Array.from(el.labelSelect?.options || []).map((opt) => opt.dataset.color));
  const available = palette.filter((color) => !used.has(color));
  return (available.length ? available : palette)[Math.floor(Math.random() * (available.length ? available.length : palette.length))];
}

function syncStagedLabelFromDefect({ randomizeColor = false } = {}) {
  if (WORKSPACE_MODE !== "generation" || !el.defectClass) return;
  const labelName = el.defectClass.value.trim() || "New Defect";
  const labelId = slugLabel(labelName);
  const color = randomizeColor ? randomLabelColor() : (el.labelColorInput?.value || "#ff9e94");
  if (el.labelIdInput) el.labelIdInput.value = labelId;
  if (el.labelNameInput) el.labelNameInput.value = labelName;
  if (el.labelColorInput) el.labelColorInput.value = color;
  renderLabels([{ id: labelId, name: labelName, color, enabled: true }]);
  if (el.datasetManageStatus) {
    el.datasetManageStatus.textContent = `已确认类别：${labelName}。导入图像后会自动创建项目和第一轮 Active Set。`;
  }
}

function hexToRgb(hex) {
  const normalized = hex.replace("#", "").trim();
  const value = parseInt(normalized.length === 3 ? normalized.split("").map((x) => x + x).join("") : normalized, 16);
  return {
    r: (value >> 16) & 255,
    g: (value >> 8) & 255,
    b: value & 255,
  };
}

function selectedItem() {
  return state.items.find((item) => item.id === state.itemId);
}

function checkpointIdForJob(job) {
  return `run-${job.id}-best`;
}

function hasReviewAnnotation(item) {
  if (!item) return false;
  return Boolean(item.annotations?.[selectedLabelId()] || (selectedLabelId() === "spall" && item.annotation_path));
}

function isCurrentItemInActiveQueue() {
  return Boolean(state.itemId && state.activeQueue.some((row) => row.item_id === state.itemId));
}

function generationRequiresQueueReview() {
  return WORKSPACE_MODE === "generation" && isFoundationStage();
}

function renderDatasets() {
  el.datasetSelect.innerHTML = "";
  if (!state.datasets.length) {
    const opt = document.createElement("option");
    const labelName = checkpointLabelName();
    opt.value = "";
    opt.textContent = WORKSPACE_MODE === "generation" ? "暂无模型生成项目" : `暂无 ${labelName} 优化数据集`;
    el.datasetSelect.appendChild(opt);
    el.datasetSelect.disabled = true;
    return;
  }
  el.datasetSelect.disabled = false;
  for (const ds of state.datasets) {
    const opt = document.createElement("option");
    opt.value = ds.id;
    opt.textContent = `${ds.name} (${ds.n_items})`;
    el.datasetSelect.appendChild(opt);
  }
  if (state.datasetId) el.datasetSelect.value = state.datasetId;
}

function renderArchives() {
  if (!el.archiveSelect) return;
  el.archiveSelect.innerHTML = "";
  const rows = state.archives;
  if (!rows.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "暂无可加载归档";
    el.archiveSelect.appendChild(opt);
    el.archiveSelect.disabled = true;
    return;
  }
  el.archiveSelect.disabled = false;
  for (const archive of rows) {
    const opt = document.createElement("option");
    opt.value = archive.id;
    opt.textContent = `${archive.name || archive.dataset_id} · ${archive.id}`;
    el.archiveSelect.appendChild(opt);
  }
}

function renderCheckpoints() {
  const previous = el.checkpointSelect.value;
  el.checkpointSelect.innerHTML = "";
  const visibleCheckpoints = state.checkpoints.filter((ckpt) => {
    if (WORKSPACE_MODE === "generation") {
      return ckpt.role === "training_run" && ckpt.project_type === "generation";
    }
    if (ckpt.project_type === "generation") {
      return ckpt.model_stage === "generated_baseline";
    }
    return ckpt.role !== "foundation";
  });
  if (WORKSPACE_MODE === "generation" && !visibleCheckpoints.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "第一轮：空白 Active Set（内部使用初始权重）";
    el.checkpointSelect.appendChild(opt);
    return;
  }
  const groups = new Map();
  for (const ckpt of visibleCheckpoints) {
    const labelName = checkpointLabelName(ckpt);
    if (!groups.has(labelName)) groups.set(labelName, []);
    groups.get(labelName).push(ckpt);
  }
  for (const [labelName, rows] of groups) {
    const group = document.createElement("optgroup");
    group.label = labelName;
    rows.sort((a, b) => String(b.created_at || "").localeCompare(String(a.created_at || "")));
    for (const ckpt of rows) {
    const opt = document.createElement("option");
    opt.value = ckpt.id;
      const role = checkpointRoleText(ckpt);
    opt.textContent = `${role}：${ckpt.name}`;
      group.appendChild(opt);
    }
    el.checkpointSelect.appendChild(group);
  }
  if (visibleCheckpoints.some((ckpt) => ckpt.id === previous)) {
    el.checkpointSelect.value = previous;
  } else if (WORKSPACE_MODE === "generation") {
    const latestRun = visibleCheckpoints
      .filter((ckpt) => ckpt.role === "training_run")
      .sort((a, b) => String(b.created_at || "").localeCompare(String(a.created_at || "")))[0];
    if (latestRun) {
      el.checkpointSelect.value = latestRun.id;
    } else if (visibleCheckpoints.some((ckpt) => ckpt.id === DEFAULT_CHECKPOINT_ID)) {
      el.checkpointSelect.value = DEFAULT_CHECKPOINT_ID;
    }
  } else {
    const latestRun = visibleCheckpoints
      .filter((ckpt) => ckpt.role === "training_run")
      .sort((a, b) => String(b.created_at || "").localeCompare(String(a.created_at || "")))[0];
    if (latestRun) {
      el.checkpointSelect.value = latestRun.id;
    } else if (visibleCheckpoints.some((ckpt) => ckpt.id === DEFAULT_CHECKPOINT_ID)) {
      el.checkpointSelect.value = DEFAULT_CHECKPOINT_ID;
    }
  }
}

function archiveQueryForCurrentModel() {
  const params = new URLSearchParams({ project_type: WORKSPACE_MODE });
  const labelId = contextLabelId();
  if (labelId) params.set("label_id", labelId);
  return `/api/dataset-archives?${params.toString()}`;
}

function openDatasetManager() {
  const params = new URLSearchParams({ project_type: WORKSPACE_MODE });
  const labelId = contextLabelId();
  if (labelId) params.set("label_id", labelId);
  if (state.datasetId) params.set("dataset_id", state.datasetId);
  const returnParams = new URLSearchParams({ workspace: WORKSPACE_MODE });
  if (state.datasetId) returnParams.set("dataset_id", state.datasetId);
  if (labelId) returnParams.set("label_id", labelId);
  params.set("return", `${WORKSPACE_MODE === "generation" ? "/generate" : "/optimize"}?${returnParams.toString()}`);
  window.location.href = `/datasets?${params.toString()}`;
}

function contextNavigationUrl(path) {
  const params = new URLSearchParams({ workspace: WORKSPACE_MODE });
  const labelId = contextLabelId();
  if (state.datasetId) params.set("dataset_id", state.datasetId);
  if (selectedCheckpointId()) params.set("checkpoint_id", selectedCheckpointId());
  if (labelId) params.set("label_id", labelId);
  params.set("return", WORKSPACE_MODE === "generation" ? "/generate" : "/optimize");
  return `${path}?${params.toString()}`;
}

function checkpointRoleText(ckpt) {
  if (ckpt.role === "foundation") return "初始权重";
  if (WORKSPACE_MODE === "optimization" && ckpt.model_stage === "generated_baseline") return "基线模型";
  if (ckpt.role === "baseline") return "基线模型";
  return "训练模型";
}

function statusText(status) {
  const map = {
    imported: "已导入",
    predicted: "已预测",
    reviewed: "已 Review",
  };
  return map[status] || status;
}

function compactDict(value) {
  const entries = Object.entries(value || {});
  if (!entries.length) return "-";
  return entries.map(([k, v]) => `${k}:${v}`).join("  ");
}

async function renderDatasetSummary() {
  if (!state.datasetId || !el.datasetSummary) return;
  const summary = await api.get(`/api/datasets/${state.datasetId}/summary`);
  syncDatasetForm(summary);
  renderLabels(summary.labels || []);
  el.datasetSummary.innerHTML = `
    <div class="summary-line"><span>样本数</span><strong>${summary.n_items}</strong></div>
    <div class="summary-line"><span>状态</span><strong>${compactDict(summary.by_status)}</strong></div>
    <div class="summary-line"><span>格式</span><strong>${compactDict(summary.by_format)}</strong></div>
    <div class="summary-line"><span>Review</span><strong>${compactDict(summary.reviewed_by_label)}</strong></div>
    <div class="summary-line"><span>版本</span><strong>${summary.n_versions}</strong></div>
  `;
}

function renderLabels(labels) {
  if (!el.labelSelect || !labels.length) return;
  const previous = el.labelSelect.value;
  el.labelSelect.innerHTML = "";
  for (const label of labels) {
    const opt = document.createElement("option");
    opt.value = label.id;
    opt.dataset.color = label.color || "#ff9e94";
    opt.textContent = label.name || label.id;
    opt.title = `label_id: ${label.id}`;
    el.labelSelect.appendChild(opt);
  }
  if (labels.some((label) => label.id === previous)) {
    el.labelSelect.value = previous;
  }
  const selected = labels.find((label) => label.id === el.labelSelect.value) || labels[0];
  if (selected) {
    if (el.labelColorInput) el.labelColorInput.value = selected.color || "#ff9e94";
    if (el.labelNameInput) el.labelNameInput.value = selected.name || selected.id;
    if (el.labelIdInput) el.labelIdInput.value = selected.id;
    document.querySelectorAll(".swatch.spall").forEach((node) => {
      node.style.background = selected.color || "#ff9e94";
    });
  }
}

function renderItems() {
  el.itemList.innerHTML = "";
  if (!state.datasetId) {
    el.itemList.innerHTML = `<div class="muted">${WORKSPACE_MODE === "generation" ? "请先创建模型生成项目。" : "请先创建或选择模型优化数据集。"}</div>`;
    return;
  }
  for (const item of state.items) {
    const row = document.createElement("div");
    row.className = `item-row ${item.id === state.itemId ? "active" : ""}`;
    row.onclick = () => selectItem(item.id);
    const name = document.createElement("div");
    name.className = "item-name";
    name.textContent = item.original_name;
    const badge = document.createElement("span");
    badge.className = `badge ${item.status}`;
    badge.textContent = statusText(item.status);
    row.append(name, badge);
    el.itemList.appendChild(row);
  }
}

function renderActiveQueue(rows) {
  state.activeQueue = rows || [];
  el.activeQueue.innerHTML = "";
  if (!rows.length) {
    el.activeQueue.innerHTML = '<div class="muted">暂无待 Review 队列，请先全量预测并生成队列。</div>';
    return;
  }
  for (const row of rows) {
    const div = document.createElement("div");
    div.className = `active-row ${row.item_id === state.itemId ? "active" : ""}`;
    div.onclick = () => selectActiveQueueItem(row.item_id);
    div.innerHTML = `
      <strong>${row.original_name}</strong>
      <div class="active-score"><span>#${row.rank || "-"} · 分数 ${row.score.toFixed(3)}</span><span>${row.status === "reviewed" ? "已Review" : "待Review"}</span></div>
      <div class="active-score"><span>不确定 ${row.uncertain_ratio.toFixed(3)}</span><span>碎片 ${row.components}</span></div>
      <div class="active-score"><span>预测 ${row.pred_px} px</span><span>多样性 ${row.diversity.toFixed(3)}</span></div>
    `;
    el.activeQueue.appendChild(div);
  }
}

function setActiveBatch(batch) {
  state.activeBatch = batch || null;
  state.activeQueue = batch?.items || [];
  state.activeIndex = -1;
  renderActiveQueue(state.activeQueue);
}

async function loadActiveBatches() {
  if (!state.datasetId) return;
  const batches = await api.get(`/api/datasets/${state.datasetId}/active-learning/batches`);
  const pendingBatch = batches.find((row) => row.n_pending > 0) || null;
  setActiveBatch(pendingBatch);
  if (pendingBatch) {
    el.activeStatus.textContent = `当前队列 ${pendingBatch.id}：待 Review ${pendingBatch.n_pending}/${pendingBatch.n_items}`;
  } else if (batches.length) {
    el.activeStatus.textContent = "当前没有待 Review 队列；可以训练、重新预测未 Review 图，或生成新队列。";
  }
}

async function selectActiveQueueItem(itemId) {
  const idx = state.activeQueue.findIndex((row) => row.item_id === itemId);
  if (idx >= 0) state.activeIndex = idx;
  await selectItem(itemId);
  renderActiveQueue(state.activeQueue);
}

function nextPendingActiveItem() {
  if (!state.activeQueue.length) return null;
  const start = Math.max(state.activeIndex, -1) + 1;
  for (let i = start; i < state.activeQueue.length; i += 1) {
    if (state.activeQueue[i].status !== "reviewed") {
      state.activeIndex = i;
      return state.activeQueue[i];
    }
  }
  for (let i = 0; i < start; i += 1) {
    if (state.activeQueue[i].status !== "reviewed") {
      state.activeIndex = i;
      return state.activeQueue[i];
    }
  }
  return null;
}

async function selectNextActiveItem() {
  const row = nextPendingActiveItem();
  if (!row) {
    el.activeStatus.textContent = "当前队列已全部 Review";
    return;
  }
  await selectItem(row.item_id);
  renderActiveQueue(state.activeQueue);
  el.activeStatus.textContent = `当前推荐：#${row.rank || state.activeIndex + 1} ${row.original_name}`;
}

async function refresh() {
  const [health, checkpoints, settings] = await Promise.all([
    api.get("/api/health"),
    api.get("/api/checkpoints"),
    api.get("/api/settings"),
  ]);
  applyInferenceSettings(settings.inference);
  el.health.textContent = `就绪 · ${health.device}`;
  state.checkpoints = checkpoints;
  renderCheckpoints();
  state.activeModelLabelId = checkpointLabelId();
  await loadDatasetsForCurrentModel({ restore: true });
}

function saveCurrentModelContext() {
  const labelId = state.activeModelLabelId;
  if (!labelId) return;
  state.contextByLabel[labelId] = {
    datasetId: state.datasetId,
    itemId: state.itemId,
    checkpointId: selectedCheckpointId(),
  };
}

function datasetQueryForCurrentModel() {
  const params = new URLSearchParams({ project_type: WORKSPACE_MODE });
  const current = selectedCheckpoint();
  const shouldFilterByLabel = WORKSPACE_MODE === "optimization" || (WORKSPACE_MODE === "generation" && current?.role !== "foundation");
  const labelId = shouldFilterByLabel ? checkpointLabelId() : null;
  if (labelId) params.set("label_id", labelId);
  return `/api/datasets?${params.toString()}`;
}

async function loadDatasetsForCurrentModel({ restore = false } = {}) {
  const labelId = checkpointLabelId();
  state.datasets = await api.get(datasetQueryForCurrentModel());
  const saved = labelId ? state.contextByLabel[labelId] : null;
  const candidateId = restore && saved?.datasetId ? saved.datasetId : state.datasetId;
  if (candidateId && state.datasets.some((ds) => ds.id === candidateId)) {
    state.datasetId = candidateId;
    state.itemId = restore && saved?.itemId ? saved.itemId : state.itemId;
  } else {
    state.datasetId = state.datasets.length ? state.datasets[0].id : null;
    state.itemId = null;
  }
  state.activeModelLabelId = labelId;
  renderDatasets();
  if (state.datasetId) {
    await loadItems();
    await loadActiveBatches();
    await renderDatasetSummary();
    state.archives = await api.get(archiveQueryForCurrentModel());
    renderArchives();
    if (state.itemId && state.items.some((item) => item.id === state.itemId)) {
      await selectItem(state.itemId);
    }
  } else {
    state.archives = await api.get(archiveQueryForCurrentModel());
    renderArchives();
    clearWorkspaceForModel(labelId);
  }
}

function clearWorkspaceForModel(labelId) {
  state.items = [];
  state.itemId = null;
  setActiveBatch(null);
  renderItems();
  clearMask();
  imageCtx.clearRect(0, 0, el.imageCanvas.width, el.imageCanvas.height);
  el.datasetSummary.innerHTML = "";
  renderModelLabelFromCheckpoint();
  const labelName = checkpointLabelName();
  if (WORKSPACE_MODE === "optimization" && labelId) {
    el.activeStatus.textContent = `当前选择的是 ${labelName} 基线模型；暂无该类别的优化数据集。Spall Review 队列已保留在缓冲区，切回 Spall 模型会恢复。`;
  } else {
    el.activeStatus.textContent = WORKSPACE_MODE === "generation" ? "请先创建独立的模型生成项目。" : "请先选择模型优化数据集。";
  }
  setStatus("未选择图像");
}

function renderModelLabelFromCheckpoint() {
  if (!el.labelSelect) return;
  const labelId = checkpointLabelId() || el.labelIdInput?.value || "new-defect";
  const labelName = checkpointLabelName();
  const color = checkpointLabelColor();
  el.labelSelect.innerHTML = "";
  const opt = document.createElement("option");
  opt.value = labelId;
  opt.dataset.color = color;
  opt.textContent = labelName;
  opt.title = `label_id: ${labelId}`;
  el.labelSelect.appendChild(opt);
  if (el.labelColorInput) el.labelColorInput.value = color;
  if (el.labelNameInput) el.labelNameInput.value = labelName;
  if (el.labelIdInput) el.labelIdInput.value = labelId;
  document.querySelectorAll(".swatch.spall").forEach((node) => {
    node.style.background = color;
  });
}

async function loadItems() {
  state.items = await api.get(`/api/datasets/${state.datasetId}/items`);
  renderItems();
  await renderDatasetSummary();
}

async function selectItem(itemId) {
  state.itemId = itemId;
  renderItems();
  const item = selectedItem();
  setStatus(item.original_name);
  await loadImage(`/api/datasets/${state.datasetId}/items/${item.id}/image`);
  if (hasReviewAnnotation(item)) {
    await loadMask(`/api/datasets/${state.datasetId}/items/${item.id}/annotation?label_id=${selectedLabelId()}`);
  } else {
    const predictionId = item.predictions?.[selectedCheckpointId()] ? selectedCheckpointId() : item.latest_prediction;
    if (predictionId) {
      await loadMask(`/api/datasets/${state.datasetId}/items/${item.id}/prediction-mask?checkpoint_id=${predictionId}`);
    } else {
      clearMask();
    }
  }
}

function fitCanvasToImage(img) {
  el.imageCanvas.width = img.naturalWidth;
  el.imageCanvas.height = img.naturalHeight;
  el.maskCanvas.width = img.naturalWidth;
  el.maskCanvas.height = img.naturalHeight;
  applyZoom();
  imageCtx.clearRect(0, 0, el.imageCanvas.width, el.imageCanvas.height);
  imageCtx.drawImage(img, 0, 0);
}

function applyZoom() {
  if (el.zoomValue) {
    el.zoomValue.textContent = `${Math.round(state.zoom * 100)}%`;
  }
  if (!state.image) return;
  const width = Math.round(state.image.naturalWidth * state.zoom);
  const height = Math.round(state.image.naturalHeight * state.zoom);
  for (const canvas of [el.imageCanvas, el.maskCanvas]) {
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
  }
}

function setZoom(nextZoom) {
  state.zoom = Math.min(4, Math.max(0.25, nextZoom));
  applyZoom();
}

function loadHtmlImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = `${src}${src.includes("?") ? "&" : "?"}t=${Date.now()}`;
  });
}

async function loadImage(src) {
  const img = await loadHtmlImage(src);
  state.image = img;
  fitCanvasToImage(img);
  clearMask();
}

async function loadMask(src) {
  const img = await loadHtmlImage(src);
  clearMask();
  const color = hexToRgb(selectedLabelColor());
  const tmp = document.createElement("canvas");
  tmp.width = el.maskCanvas.width;
  tmp.height = el.maskCanvas.height;
  const tctx = tmp.getContext("2d");
  tctx.drawImage(img, 0, 0, tmp.width, tmp.height);
  const srcData = tctx.getImageData(0, 0, tmp.width, tmp.height);
  const dst = maskCtx.createImageData(tmp.width, tmp.height);
  for (let i = 0; i < srcData.data.length; i += 4) {
    const intensity = Math.max(srcData.data[i], srcData.data[i + 1], srcData.data[i + 2]);
    const hit = intensity > 16;
    if (hit) {
      dst.data[i] = color.r;
      dst.data[i + 1] = color.g;
      dst.data[i + 2] = color.b;
      dst.data[i + 3] = MASK_ALPHA;
    }
  }
  maskCtx.putImageData(dst, 0, 0);
}

function clearMask() {
  maskCtx.clearRect(0, 0, el.maskCanvas.width, el.maskCanvas.height);
}

function clearReviewMask() {
  clearMask();
  el.saveStatus.textContent = "已清空当前掩码";
}

function canvasPoint(evt) {
  const rect = el.maskCanvas.getBoundingClientRect();
  return {
    x: (evt.clientX - rect.left) * (el.maskCanvas.width / rect.width),
    y: (evt.clientY - rect.top) * (el.maskCanvas.height / rect.height),
  };
}

function drawAt(evt) {
  if (!state.drawing) return;
  const p = canvasPoint(evt);
  const size = Number(el.brushSize.value);
  const color = hexToRgb(selectedLabelColor());
  maskCtx.save();
  maskCtx.beginPath();
  maskCtx.arc(p.x, p.y, size / 2, 0, Math.PI * 2);
  if (state.tool === "erase") {
    maskCtx.globalCompositeOperation = "destination-out";
    maskCtx.fillStyle = "rgba(0,0,0,1)";
    maskCtx.fill();
  } else {
    maskCtx.globalCompositeOperation = "destination-out";
    maskCtx.fillStyle = "rgba(0,0,0,1)";
    maskCtx.fill();
    maskCtx.globalCompositeOperation = "source-over";
    maskCtx.fillStyle = `rgba(${color.r},${color.g},${color.b},${MASK_ALPHA / 255})`;
    maskCtx.fill();
  }
  maskCtx.restore();
}

function setTool(tool) {
  state.tool = tool;
  el.brushBtn.classList.toggle("active", tool === "brush");
  el.eraseBtn.classList.toggle("active", tool === "erase");
}

async function predictAllImages() {
  if (!state.datasetId) return;
  if (generationRequiresQueueReview()) {
    const message = "模型生成第一轮不做预测，请先生成空白 Active Set 并完成队列 Review。";
    setStatus(message);
    if (el.activeStatus) el.activeStatus.textContent = message;
    return;
  }
  const message = "全量模型预测中";
  setStatus(message);
  if (el.activeStatus) el.activeStatus.textContent = message;
  const result = await api.post(`/api/datasets/${state.datasetId}/active-learning/batch-predict`, {
    checkpoint_id: selectedCheckpointId(),
    ...inferenceParams(),
    limit: 0,
    only_unreviewed: false,
    force: true,
  });
  await loadItems();
  setActiveBatch(null);
  if (state.itemId) {
    await selectItem(state.itemId);
  }
  const doneText = `全量预测完成 · ${result.predicted} 张`;
  setStatus(doneText);
  if (el.activeStatus) el.activeStatus.textContent = `${doneText}，跳过 ${result.skipped} 张；请生成新队列`;
}

async function predictCurrentImage() {
  if (!state.datasetId || !state.itemId) {
    setStatus("请先选择一张图像");
    return;
  }
  if (generationRequiresQueueReview()) {
    setStatus("模型生成第一轮使用空白 mask 标注，不加载 scratch 预测。");
    clearMask();
    return;
  }
  const item = selectedItem();
  if (hasReviewAnnotation(item)) {
    setStatus("已 Review 图像受保护：模型预测不会覆盖人工 mask");
    await loadMask(`/api/datasets/${state.datasetId}/items/${item.id}/annotation?label_id=${selectedLabelId()}`);
    return;
  }
  setStatus(`单图预测中：${item.original_name}`);
  const result = await api.post(`/api/datasets/${state.datasetId}/items/${state.itemId}/predict`, {
    checkpoint_id: selectedCheckpointId(),
    ...inferenceParams(),
  });
  await loadItems();
  await selectItem(state.itemId);
  setStatus(`单图预测完成 · ${result.pred_px} px`);
}

async function predictByMode() {
  if (el.predictMode?.value === "all") {
    await predictAllImages();
  } else {
    await predictCurrentImage();
  }
}

async function saveReview() {
  if (!state.datasetId || !state.itemId) return;
  if (generationRequiresQueueReview() && !isCurrentItemInActiveQueue()) {
    el.saveStatus.textContent = "第一轮只允许保存 Active Set 队列内的空白标注图。";
    return;
  }
  const reviewedItemId = state.itemId;
  el.saveStatus.textContent = "保存中";
  const dataUrl = el.maskCanvas.toDataURL("image/png");
  const saved = await api.post(`/api/datasets/${state.datasetId}/items/${reviewedItemId}/annotations`, {
    mask_png_base64: dataUrl,
    label_id: selectedLabelId(),
    source: "manual_review",
    reviewer: "local",
  });
  await loadItems();
  const isQueuedReview = state.activeQueue.some((row) => row.item_id === reviewedItemId);
  if (state.activeBatch?.id && isQueuedReview) {
    const batch = await api.post(`/api/datasets/${state.datasetId}/active-learning/batches/${state.activeBatch.id}/items/${reviewedItemId}/reviewed`, {});
    state.activeBatch = batch;
    state.activeQueue = batch.items || [];
    state.activeIndex = state.activeQueue.findIndex((row) => row.item_id === reviewedItemId);
    renderActiveQueue(state.activeQueue);
    await selectNextActiveItem();
  }
  el.saveStatus.textContent = `已保存 · ${saved.annotations?.[selectedLabelId()]?.mask_px ?? saved.annotation?.mask_px ?? 0} px`;
}

function trainingStatusText(job) {
  if (job.status === "queued") return `排队中：${job.id}`;
  if (job.status === "running") {
    const p = job.progress;
    if (!p) return `训练中：${job.id}`;
    return `训练中：${p.epoch}/${p.epochs} · loss ${p.train_loss.toFixed(4)} · Dice ${p.val.dice.toFixed(4)} · Recall ${p.val.recall.toFixed(4)}`;
  }
  if (job.status === "completed") {
    return `训练完成：${job.id} · ${job.metrics?.n_reviewed ?? 0} 张 Review 图`;
  }
  if (job.status === "failed") return `训练失败：${job.error || job.id}`;
  return `${job.status}：${job.id}`;
}

async function pollTrainingJob(jobId) {
  if (state.trainingTimer) clearInterval(state.trainingTimer);
  const poll = async () => {
    const job = await api.get(`/api/training/jobs/${jobId}`);
    el.trainStatus.textContent = trainingStatusText(job);
    if (job.status === "completed" || job.status === "failed") {
      clearInterval(state.trainingTimer);
      state.trainingTimer = null;
      state.checkpoints = await api.get("/api/checkpoints");
      renderCheckpoints();
      if (job.status === "completed") {
        await advanceActiveLearningCycle(job);
      }
    }
  };
  await poll();
  state.trainingTimer = setInterval(() => poll().catch(console.error), 3000);
}

async function advanceActiveLearningCycle(job) {
  const nextCheckpointId = checkpointIdForJob(job);
  if (state.checkpoints.some((ckpt) => ckpt.id === nextCheckpointId)) {
    el.checkpointSelect.value = nextCheckpointId;
  }
  for (let attempt = 0; attempt < 20; attempt += 1) {
    const current = await api.get(`/api/training/jobs/${job.id}`);
    if (current.next_cycle) {
      await loadItems();
      await loadActiveBatches();
      const next = current.next_cycle;
      if (next.status === "created") {
        el.trainStatus.textContent = `训练完成，后端已切换最新模型并生成下一轮队列：${next.batch_id}`;
      } else if (next.status === "no_candidates") {
        el.trainStatus.textContent = "训练完成，没有未 Review 候选";
      } else {
        el.trainStatus.textContent = `训练完成，下一轮状态：${next.status}`;
      }
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  await loadItems();
  await loadActiveBatches();
  el.trainStatus.textContent = "训练完成，等待后端生成下一轮状态";
}

async function createDatasetFromForm() {
  const labelName = el.labelNameInput?.value?.trim() || el.defectClass.value || "Spall";
  const labelId = el.labelIdInput?.value?.trim() || slugLabel(labelName);
  const labelColor = el.labelColorInput?.value || "#ff9e94";
  const ds = await api.post("/api/datasets", {
    name: el.datasetName.value,
    defect_class: el.defectClass.value,
    label_id: labelId,
    label_name: labelName,
    label_color: labelColor,
    project_type: WORKSPACE_MODE,
  });
  state.datasetId = ds.id;
  if (WORKSPACE_MODE === "generation") {
    el.checkpointSelect.value = DEFAULT_CHECKPOINT_ID;
  }
  await refresh();
  if (WORKSPACE_MODE === "generation") {
    el.checkpointSelect.value = DEFAULT_CHECKPOINT_ID;
    state.datasetId = ds.id;
    await loadDatasetsForCurrentModel({ restore: false });
  }
  return ds;
}

async function ensureGenerationDatasetForImport() {
  if (WORKSPACE_MODE !== "generation") return state.datasetId;
  syncStagedLabelFromDefect({ randomizeColor: false });
  if (!el.datasetName.value.trim()) throw new Error("请先填写项目名称");
  if (!el.defectClass.value.trim()) throw new Error("请先填写缺陷类别");
  const stagedName = el.datasetName.value.trim();
  const stagedLabelId = el.labelIdInput?.value?.trim() || slugLabel(el.defectClass.value);
  const currentDataset = state.datasets.find((ds) => ds.id === state.datasetId);
  const currentLabelIds = new Set((currentDataset?.labels || []).map((label) => label.id));
  const matchesCurrentProject = currentDataset
    && currentDataset.name === stagedName
    && currentLabelIds.has(stagedLabelId);
  if (state.datasetId && matchesCurrentProject) return state.datasetId;
  if (state.datasetId && !matchesCurrentProject) {
    state.datasetId = null;
    state.itemId = null;
    state.items = [];
    setActiveBatch(null);
  }
  el.datasetManageStatus.textContent = "正在根据导入参数自动创建项目...";
  const ds = await createDatasetFromForm();
  state.datasetId = ds.id;
  el.datasetManageStatus.textContent = `已创建项目：${ds.name}`;
  return ds.id;
}

async function createInitialActiveSetAfterImport() {
  if (WORKSPACE_MODE !== "generation" || !isFoundationStage() || !state.datasetId) return;
  const topK = Math.max(1, Number(el.activeTopK.value || 12));
  el.activeStatus.textContent = "导入完成，正在生成第一轮空白 Active Set...";
  const result = await api.post(`/api/datasets/${state.datasetId}/active-learning/initial-review-queue`, {
    checkpoint_id: selectedCheckpointId(),
    label_id: selectedLabelId(),
    ...inferenceParams(),
    top_k: topK,
    predict_missing: false,
    create_batch: true,
  });
  if (result.batch && result.ranked > 0) {
    setActiveBatch(result.batch);
    el.activeStatus.textContent = `已自动生成第一轮空白 Active Set：${result.items.length} 张。请只 Review 队列内图像。`;
    await selectNextActiveItem();
  } else {
    setActiveBatch(null);
    el.activeStatus.textContent = "导入完成，但没有可进入第一轮 Review 的图像。";
  }
}

function checkpointsForDatasetLabel() {
  const labelId = currentDataset()?.labels?.[0]?.id || null;
  if (!labelId) return [];
  return state.checkpoints
    .filter((ckpt) => checkpointLabelId(ckpt) === labelId)
    .sort((a, b) => String(b.created_at || "").localeCompare(String(a.created_at || "")));
}

function latestGenerationCheckpointForDataset() {
  return checkpointsForDatasetLabel().find((ckpt) => ckpt.role === "training_run" && ckpt.project_type === "generation") || null;
}

function baselineCheckpointForDataset() {
  return checkpointsForDatasetLabel().find((ckpt) => ckpt.model_stage === "generated_baseline" || ckpt.role === "baseline") || null;
}

function setWorkflowBlocked(blocked, message = "") {
  state.modelBlocked = blocked;
  for (const node of [el.predictBtn, el.batchPredict, el.rankActive, el.trainJob, el.openTraining, el.openTesting]) {
    if (node) node.disabled = blocked;
  }
  if (blocked) {
    if (el.activeStatus) el.activeStatus.textContent = message;
    if (el.trainStatus) el.trainStatus.textContent = message;
  }
}

renderDatasets = function renderDatasets() {
  if (!el.datasetSelect) return;
  el.datasetSelect.innerHTML = "";
  if (!state.datasets.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = WORKSPACE_MODE === "generation" ? "暂无模型生成数据集" : "暂无模型升级数据集";
    el.datasetSelect.appendChild(opt);
    el.datasetSelect.disabled = true;
    return;
  }
  el.datasetSelect.disabled = false;
  for (const ds of state.datasets) {
    const opt = document.createElement("option");
    opt.value = ds.id;
    const label = ds.labels?.[0]?.name || ds.defect_class || "未绑定标签";
    opt.textContent = `${ds.name} | ${label} | ${ds.n_items} 张`;
    el.datasetSelect.appendChild(opt);
  }
  if (state.datasetId) el.datasetSelect.value = state.datasetId;
};

renderArchives = function renderArchives() {
  if (!el.archiveSelect) return;
  el.archiveSelect.innerHTML = "";
  if (!state.archives.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "暂无可加载归档";
    el.archiveSelect.appendChild(opt);
    el.archiveSelect.disabled = true;
    return;
  }
  el.archiveSelect.disabled = false;
  for (const archive of state.archives) {
    const opt = document.createElement("option");
    opt.value = archive.id;
    opt.textContent = `${archive.name || archive.dataset_id} | ${archive.id}`;
    el.archiveSelect.appendChild(opt);
  }
};

renderCheckpoints = function renderCheckpoints() {
  if (!el.checkpointSelect) return;
  el.checkpointSelect.innerHTML = "";
  const ds = currentDataset();
  const label = ds?.labels?.[0] || null;
  const labelName = label?.name || ds?.defect_class || label?.id || "未绑定类别";
  let selected = null;
  if (!ds) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "请先选择数据集";
    el.checkpointSelect.appendChild(opt);
    setWorkflowBlocked(false);
    return;
  }
  if (WORKSPACE_MODE === "generation") {
    selected = latestGenerationCheckpointForDataset()
      || state.checkpoints.find((ckpt) => ckpt.id === DEFAULT_CHECKPOINT_ID)
      || state.checkpoints.find((ckpt) => ckpt.role === "foundation");
  } else {
    selected = baselineCheckpointForDataset();
  }
  if (!selected && WORKSPACE_MODE === "optimization") {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = `${labelName} 缺少基线模型，请先完成模型生成`;
    el.checkpointSelect.appendChild(opt);
    setWorkflowBlocked(true, `${labelName} 暂无基线模型。请先在模型生成工作台完成全部 Review，并训练得到基线模型。`);
    return;
  }
  if (!selected) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "缺少可用初始权重";
    el.checkpointSelect.appendChild(opt);
    setWorkflowBlocked(true, "缺少可用初始权重，请检查 checkpoints 目录。");
    return;
  }
  const opt = document.createElement("option");
  opt.value = selected.id;
  opt.textContent = `${checkpointRoleText(selected)} | ${selected.name}`;
  el.checkpointSelect.appendChild(opt);
  el.checkpointSelect.value = selected.id;
  setWorkflowBlocked(false);
};

datasetQueryForCurrentModel = function datasetQueryForCurrentModel() {
  const params = new URLSearchParams({ project_type: WORKSPACE_MODE });
  return `/api/datasets?${params.toString()}`;
};

loadDatasetsForCurrentModel = async function loadDatasetsForCurrentModel({ restore = false } = {}) {
  state.datasets = await api.get(datasetQueryForCurrentModel());
  const saved = state.activeModelLabelId ? state.contextByLabel[state.activeModelLabelId] : null;
  const requestedDatasetId = pageParams.get("dataset_id");
  const candidateId = requestedDatasetId || (restore && saved?.datasetId ? saved.datasetId : state.datasetId);
  if (candidateId && state.datasets.some((ds) => ds.id === candidateId)) {
    state.datasetId = candidateId;
    state.itemId = restore && saved?.itemId ? saved.itemId : state.itemId;
  } else {
    state.datasetId = state.datasets.length ? state.datasets[0].id : null;
    state.itemId = null;
  }
  renderDatasets();
  renderCheckpoints();
  state.activeModelLabelId = contextLabelId();
  if (state.datasetId) {
    await loadItems();
    await loadActiveBatches();
    await renderDatasetSummary();
    state.archives = await api.get(archiveQueryForCurrentModel());
    renderArchives();
    if (state.itemId && state.items.some((item) => item.id === state.itemId)) {
      await selectItem(state.itemId);
    }
  } else {
    state.archives = await api.get(archiveQueryForCurrentModel());
    renderArchives();
    clearWorkspaceForModel(state.activeModelLabelId);
  }
};

if (el.createDataset) {
  el.createDataset.onclick = createDatasetFromForm;
}

if (el.confirmProjectName) {
  el.confirmProjectName.onclick = () => {
    if (el.datasetManageStatus) el.datasetManageStatus.textContent = `已确认项目名称：${el.datasetName.value.trim() || "未命名项目"}`;
    el.defectClass?.focus();
  };
}

if (el.confirmDefectClass) {
  el.confirmDefectClass.onclick = () => {
    syncStagedLabelFromDefect({ randomizeColor: true });
    el.sourceDir?.focus();
  };
}

if (el.datasetName) {
  el.datasetName.addEventListener("keydown", (evt) => {
    if (WORKSPACE_MODE === "generation" && evt.key === "Enter") {
      evt.preventDefault();
      el.confirmProjectName?.click();
    }
  });
}

if (el.defectClass) {
  el.defectClass.addEventListener("keydown", (evt) => {
    if (WORKSPACE_MODE === "generation" && evt.key === "Enter") {
      evt.preventDefault();
      el.confirmDefectClass?.click();
    }
  });
}

el.datasetSelect.onchange = async () => {
  if (!el.datasetSelect.value) return;
  state.datasetId = el.datasetSelect.value;
  state.itemId = null;
  saveCurrentModelContext();
  renderCheckpoints();
  await loadItems();
  await loadActiveBatches();
  state.archives = await api.get(archiveQueryForCurrentModel());
  renderArchives();
};

el.checkpointSelect.onchange = async () => {
  saveCurrentModelContext();
  await loadDatasetsForCurrentModel({ restore: true });
};

if (el.archiveDataset) {
  el.archiveDataset.textContent = "数据集管理";
  el.archiveDataset.onclick = openDatasetManager;
}

if (el.deleteDataset) {
  el.deleteDataset.textContent = "集中维护";
  el.deleteDataset.onclick = openDatasetManager;
}

if (el.restoreArchive) {
  el.restoreArchive.onclick = async () => {
    const archiveId = el.archiveSelect?.value;
    if (!archiveId) return;
    el.datasetManageStatus.textContent = "加载归档中";
    const restored = await api.post(`/api/dataset-archives/${encodeURIComponent(archiveId)}/restore?project_type=${WORKSPACE_MODE}`, {});
    state.datasetId = restored.id;
    state.itemId = null;
    await loadDatasetsForCurrentModel({ restore: false });
    el.datasetManageStatus.textContent = `已加载归档：${restored.name}`;
  };
}

el.createSnapshot.onclick = async () => {
  if (!state.datasetId) return;
  el.snapshotStatus.textContent = "创建版本中";
  const snapshot = await api.post(`/api/datasets/${state.datasetId}/snapshots`, {
    name: "人工快照",
    note: "从界面创建",
  });
  el.snapshotStatus.textContent = `已创建：${snapshot.id}`;
  await renderDatasetSummary();
};

el.rebuildMetadata.onclick = async () => {
  if (!state.datasetId) return;
  el.snapshotStatus.textContent = "重建元数据中";
  const result = await api.post(`/api/datasets/${state.datasetId}/rebuild-metadata`, {});
  el.snapshotStatus.textContent = `已重建 ${result.updated} 张`;
  await refresh();
};

async function exportDataset(format) {
  if (!state.datasetId) return;
  el.exportStatus.textContent = "导出中";
  const result = await api.post(`/api/datasets/${state.datasetId}/exports`, {
    format,
    label_id: selectedLabelId(),
    include_predictions: true,
  });
  const target = result.loader || result.annotations || result.manifest || result.path;
  el.exportStatus.textContent = `已导出 ${result.format}：${target}`;
}

el.exportActive.onclick = () => exportDataset("active_learning").catch((err) => {
  el.exportStatus.textContent = `导出失败：${err.message}`;
});

el.exportCoco.onclick = () => exportDataset("coco").catch((err) => {
  el.exportStatus.textContent = `导出失败：${err.message}`;
});

el.exportFiftyOne.onclick = () => exportDataset("fiftyone").catch((err) => {
  el.exportStatus.textContent = `导出失败：${err.message}`;
});

if (el.importImages) el.importImages.onclick = async () => {
  if (WORKSPACE_MODE === "generation") {
    await ensureGenerationDatasetForImport();
  }
  if (!state.datasetId) return;
  el.importStatus.textContent = "导入中";
  const result = await api.post(`/api/datasets/${state.datasetId}/import`, { source_dir: el.sourceDir.value });
  el.importStatus.textContent = `已导入 ${result.imported} 张`;
  await loadItems();
  state.datasets = await api.get(datasetQueryForCurrentModel());
  renderDatasets();
  if (WORKSPACE_MODE === "generation" && isFoundationStage()) {
    await createInitialActiveSetAfterImport();
  }
};

el.batchPredict.onclick = async () => {
  if (!state.datasetId) return;
  if (generationRequiresQueueReview()) {
    el.activeStatus.textContent = "模型生成第一轮不做批量预测，请点击“生成 Active Set”创建空白 Review 队列。";
    return;
  }
  el.activeStatus.textContent = "全量模型预测中";
  const result = await api.post(`/api/datasets/${state.datasetId}/active-learning/batch-predict`, {
    checkpoint_id: selectedCheckpointId(),
    ...inferenceParams(),
    limit: 0,
    only_unreviewed: false,
    force: true,
  });
  await loadItems();
  setActiveBatch(null);
  el.activeStatus.textContent = `全量预测完成：${result.predicted} 张，跳过 ${result.skipped}；请生成新队列`;
};

el.rankActive.onclick = async () => {
  if (!state.datasetId) return;
  const firstGenerationRound = WORKSPACE_MODE === "generation" && isFoundationStage();
  el.activeStatus.textContent = firstGenerationRound ? "生成第一轮空白 Review 队列中" : "生成主动学习队列中";
  const endpoint = firstGenerationRound
    ? `/api/datasets/${state.datasetId}/active-learning/initial-review-queue`
    : `/api/datasets/${state.datasetId}/active-learning/rank`;
  const result = await api.post(endpoint, {
    checkpoint_id: selectedCheckpointId(),
    label_id: selectedLabelId(),
    ...inferenceParams(),
    top_k: Number(el.activeTopK.value),
    predict_missing: true,
    create_batch: true,
  });
  if (!result.batch || result.ranked === 0) {
    setActiveBatch(null);
    el.activeStatus.textContent = "没有未 Review 候选图像：可以进行最终训练、导出数据集，或导入新图像进入下一轮。";
    return;
  }
  el.activeStatus.textContent = firstGenerationRound
    ? `已创建第一轮空白 Review 队列 ${result.batch?.id || ""}：请只标注这 ${result.items.length} 张`
    : `已创建队列 ${result.batch?.id || ""}：排序 ${result.ranked} 张，推荐 ${result.items.length} 张`;
  setActiveBatch(result.batch);
};

el.nextActive.onclick = () => {
  selectNextActiveItem().catch((err) => {
    el.activeStatus.textContent = `切换失败：${err.message}`;
  });
};

el.predictBtn.onclick = () => {
  predictByMode().catch((err) => {
    setStatus(`预测失败：${err.message}`);
  });
};
el.brushBtn.onclick = () => setTool("brush");
el.eraseBtn.onclick = () => setTool("erase");
el.clearBtn.onclick = clearReviewMask;
el.saveBtn.onclick = saveReview;
el.zoomOut.onclick = () => setZoom(state.zoom / 1.25);
el.zoomIn.onclick = () => setZoom(state.zoom * 1.25);
el.zoomReset.onclick = () => setZoom(1);

el.trainJob.onclick = async () => {
  if (!state.datasetId) return;
  el.trainStatus.textContent = "训练任务创建中";
  const job = await api.post("/api/training/jobs", {
    dataset_id: state.datasetId,
    base_checkpoint_id: selectedCheckpointId(),
    label_id: selectedLabelId(),
    active_batch_id: state.activeBatch?.id || null,
    epochs: Number(el.trainEpochs.value),
    samples_per_epoch: Number(el.trainSamples.value),
    batch_size: Number(el.trainBatch.value),
    learning_rate: Number(el.trainLr.value),
    note: "界面创建",
  });
  await pollTrainingJob(job.id);
};

el.openTraining.onclick = () => {
  window.location.href = contextNavigationUrl("/training");
};

el.openTesting.onclick = () => {
  window.location.href = contextNavigationUrl("/testing");
};

el.labelSelect.onchange = async () => {
  const option = el.labelSelect.selectedOptions?.[0];
  if (option) {
    if (el.labelColorInput) el.labelColorInput.value = option.dataset.color || "#ff9e94";
    document.querySelectorAll(".swatch.spall").forEach((node) => {
      node.style.background = option.dataset.color || "#ff9e94";
    });
  }
  if (state.itemId) await selectItem(state.itemId);
};

if (el.labelColorInput) {
  el.labelColorInput.oninput = () => {
    const option = el.labelSelect?.selectedOptions?.[0];
    if (option) option.dataset.color = el.labelColorInput.value;
    document.querySelectorAll(".swatch.spall").forEach((node) => {
      node.style.background = el.labelColorInput.value;
    });
    if (state.itemId) selectItem(state.itemId).catch(console.error);
  };
}

el.maskCanvas.addEventListener("pointerdown", (evt) => {
  state.drawing = true;
  el.maskCanvas.setPointerCapture(evt.pointerId);
  drawAt(evt);
});
el.maskCanvas.addEventListener("pointermove", drawAt);
el.maskCanvas.addEventListener("pointerup", () => {
  state.drawing = false;
});
el.maskCanvas.addEventListener("pointercancel", () => {
  state.drawing = false;
});

refresh().catch((err) => {
  el.health.textContent = "异常";
  console.error(err);
});
