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
};

const state = {
  datasets: [],
  checkpoints: [],
  selectedRun: null,
  inference: null,
};

const el = {
  testHealth: document.getElementById("testHealth"),
  backHome: document.getElementById("backHome"),
  refreshTests: document.getElementById("refreshTests"),
  datasetSelect: document.getElementById("datasetSelect"),
  checkpointSelect: document.getElementById("checkpointSelect"),
  sampleCount: document.getElementById("sampleCount"),
  threshold: document.getElementById("threshold"),
  runTest: document.getElementById("runTest"),
  testStatus: document.getElementById("testStatus"),
  testList: document.getElementById("testList"),
  testSummary: document.getElementById("testSummary"),
  testRows: document.getElementById("testRows"),
};

const DEFAULT_INFERENCE = {
  threshold: 0.5,
  tile: 512,
  stride: 384,
};

const pageParams = new URLSearchParams(window.location.search);

function returnUrl() {
  return pageParams.get("return") || "/";
}

function contextLabelId() {
  return pageParams.get("label_id") || selectedDataset()?.labels?.[0]?.id || "spall";
}

function selectedDataset() {
  return state.datasets.find((ds) => ds.id === selectedDatasetId()) || null;
}

function selectedDatasetId() {
  return el.datasetSelect.value;
}

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

function fmt(value) {
  return Number.isFinite(value) ? value.toFixed(4) : "-";
}

function checkpointLabelId(ckpt) {
  return ckpt?.label_id || (ckpt?.id === "baseline-spall-unet-recall-v1" ? "spall" : null);
}

function checkpointRoleText(ckpt) {
  if (ckpt.model_stage === "generated_baseline") return "基线模型";
  if (ckpt.role === "baseline") return "基线模型";
  return "训练模型";
}

function datasetMatchesContext(ds) {
  const requestedDataset = pageParams.get("dataset_id");
  const workspace = pageParams.get("workspace");
  const labelId = pageParams.get("label_id");
  if (requestedDataset && ds.id !== requestedDataset) return false;
  if (workspace && ds.project_type !== workspace) return false;
  if (labelId && !(ds.labels || []).some((label) => label.id === labelId)) return false;
  return true;
}

function checkpointMatchesContext(ckpt) {
  const requestedCheckpoint = pageParams.get("checkpoint_id");
  const workspace = pageParams.get("workspace");
  const labelId = pageParams.get("label_id") || selectedDataset()?.labels?.[0]?.id;
  if (requestedCheckpoint && ckpt.id === requestedCheckpoint) return true;
  if (ckpt.role === "foundation") return false;
  if (workspace === "generation") {
    return ckpt.project_type === "generation" && checkpointLabelId(ckpt) === labelId;
  }
  if (workspace === "optimization") {
    if (ckpt.project_type === "generation") return ckpt.model_stage === "generated_baseline" && checkpointLabelId(ckpt) === labelId;
    return checkpointLabelId(ckpt) === labelId;
  }
  return !labelId || checkpointLabelId(ckpt) === labelId;
}

function renderDatasets() {
  el.datasetSelect.innerHTML = "";
  for (const ds of state.datasets) {
    const opt = document.createElement("option");
    opt.value = ds.id;
    opt.textContent = `${ds.name} (${ds.n_items})`;
    el.datasetSelect.appendChild(opt);
  }
  if (pageParams.get("dataset_id") && state.datasets.some((ds) => ds.id === pageParams.get("dataset_id"))) {
    el.datasetSelect.value = pageParams.get("dataset_id");
  }
}

function renderCheckpoints() {
  el.checkpointSelect.innerHTML = "";
  const rows = state.checkpoints
    .filter(checkpointMatchesContext)
    .sort((a, b) => {
      if (a.role !== b.role) return a.role === "training_run" ? -1 : 1;
      return String(b.created_at || "").localeCompare(String(a.created_at || ""));
    });
  for (const ckpt of rows) {
    const opt = document.createElement("option");
    opt.value = ckpt.id;
    opt.textContent = `${checkpointRoleText(ckpt)}：${ckpt.name}`;
    el.checkpointSelect.appendChild(opt);
  }
  if (pageParams.get("checkpoint_id") && rows.some((ckpt) => ckpt.id === pageParams.get("checkpoint_id"))) {
    el.checkpointSelect.value = pageParams.get("checkpoint_id");
  }
}

async function refresh() {
  const [datasets, checkpoints, settings] = await Promise.all([
    api.get("/api/datasets"),
    api.get("/api/checkpoints"),
    api.get("/api/settings"),
  ]);
  applyInferenceSettings(settings.inference);
  const contextualDatasets = datasets.filter(datasetMatchesContext);
  state.datasets = contextualDatasets.length ? contextualDatasets : datasets;
  state.checkpoints = checkpoints;
  renderDatasets();
  renderCheckpoints();
  el.testHealth.textContent = `已同步 · 数据集 ${state.datasets.length} · 模型 ${el.checkpointSelect.options.length}`;
  if (selectedDatasetId()) await loadTests();
}

async function loadTests() {
  if (!selectedDatasetId()) {
    el.testList.innerHTML = '<div class="muted">暂无数据集。</div>';
    return;
  }
  const tests = await api.get(`/api/datasets/${selectedDatasetId()}/model-tests`);
  el.testList.innerHTML = "";
  const labelId = contextLabelId();
  const rows = tests.filter((run) => !labelId || run.label_id === labelId || !run.label_id);
  if (!rows.length) {
    el.testList.innerHTML = '<div class="muted">暂无测试记录。</div>';
    return;
  }
  for (const run of rows) {
    const row = document.createElement("div");
    row.className = `job-row ${run.id === state.selectedRun?.id ? "active" : ""}`;
    row.onclick = () => renderRun(run);
    row.innerHTML = `
      <strong>${run.id}</strong>
      <div class="muted">${run.checkpoint_id} · ${run.sample_count} 张</div>
    `;
    el.testList.appendChild(row);
  }
  if (!state.selectedRun || state.selectedRun.dataset_id !== selectedDatasetId()) renderRun(rows[0]);
}

function renderRun(run) {
  state.selectedRun = run;
  const m = run.metrics || {};
  el.testSummary.textContent = `${run.id} · ${run.checkpoint_id} · Dice ${fmt(m.dice)} · Recall ${fmt(m.recall)} · Precision ${fmt(m.precision)} · FN ${m.fn ?? "-"} px · FP ${m.fp ?? "-"} px`;
  el.testRows.innerHTML = "";
  for (const row of run.rows || []) {
    const wrap = document.createElement("div");
    wrap.className = "test-row";
    wrap.innerHTML = `
      <div class="muted">${row.original_name} · D ${fmt(row.dice)} · R ${fmt(row.recall)} · P ${fmt(row.precision)} · FN ${row.fn} · FP ${row.fp}</div>
      <img src="/api/datasets/${run.dataset_id}/model-tests/${run.id}/files/${row.row}?t=${Date.now()}" />
    `;
    el.testRows.appendChild(wrap);
  }
}

el.runTest.onclick = async () => {
  if (!selectedDatasetId() || !el.checkpointSelect.value) return;
  el.testStatus.textContent = "测试运行中";
  const run = await api.post(`/api/datasets/${selectedDatasetId()}/model-tests`, {
    checkpoint_id: el.checkpointSelect.value,
    label_id: contextLabelId(),
    ...inferenceParams(),
    sample_count: Number(el.sampleCount.value),
    seed: null,
  });
  el.testStatus.textContent = `测试完成：${run.sample_count} 张`;
  renderRun(run);
};

el.datasetSelect.onchange = () => {
  state.selectedRun = null;
  renderCheckpoints();
  loadTests().catch((err) => {
    el.testStatus.textContent = `加载失败：${err.message}`;
  });
};

el.backHome.onclick = () => {
  window.location.href = returnUrl();
};

el.refreshTests.onclick = () => refresh().catch(console.error);

refresh().catch((err) => {
  el.testHealth.textContent = "同步失败";
  console.error(err);
});
