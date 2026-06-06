const api = {
  async get(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
  async post(url, body = {}) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
  async patch(url, body = {}) {
    const res = await fetch(url, {
      method: "PATCH",
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

const el = {
  health: document.getElementById("datasetHealth"),
  table: document.getElementById("datasetTable"),
  archives: document.getElementById("datasetArchives"),
  events: document.getElementById("datasetEvents"),
  filter: document.getElementById("projectTypeFilter"),
  backHome: document.getElementById("backHome"),
  refresh: document.getElementById("refreshDatasets"),
  name: document.getElementById("newDatasetName"),
  defectClass: document.getElementById("newDefectClass"),
  labelColor: document.getElementById("newLabelColor"),
  projectType: document.getElementById("newProjectType"),
  sourceDir: document.getElementById("newSourceDir"),
  newMode: document.getElementById("newDatasetMode"),
  saveSettings: document.getElementById("saveDatasetSettings"),
  createImport: document.getElementById("createImportDataset"),
  importSelected: document.getElementById("importIntoSelected"),
  formStatus: document.getElementById("datasetFormStatus"),
  browserSummary: document.getElementById("datasetBrowserSummary"),
  itemFilter: document.getElementById("itemStatusFilter"),
  itemGrid: document.getElementById("datasetItemGrid"),
  itemPreview: document.getElementById("datasetItemPreview"),
};

const state = {
  rows: [],
  items: [],
  selectedDatasetId: null,
  selectedItemId: null,
};

const initialParams = new URLSearchParams(window.location.search);

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function returnUrl() {
  return initialParams.get("return") || "/";
}

function selectedReturnUrl() {
  const target = new URL(returnUrl(), window.location.origin);
  const row = selectedRow();
  if (row) {
    target.searchParams.set("dataset_id", row.dataset_id);
    target.searchParams.set("label_id", row.label_id || "");
    target.searchParams.set("workspace", row.project_type || el.projectType.value || "");
  }
  return `${target.pathname}${target.search}${target.hash}`;
}

function slugLabel(text) {
  return String(text || "defect")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "") || "defect";
}

function statusText(status) {
  const map = {
    empty: "空项目",
    imported: "已导入",
    predicted: "已预测",
    reviewing: "Review 中",
    baseline_ready: "基线完成",
    review_complete: "Review 完成",
    reviewed: "已 Review",
    unreviewed: "未 Review",
  };
  return map[status] || status || "-";
}

function projectTypeText(value) {
  if (value === "generation") return "模型生成";
  if (value === "optimization") return "模型升级";
  return value || "-";
}

function eventTypeText(type) {
  const map = {
    catalog_baseline: "目录基线",
    create: "创建",
    update_dataset: "更新设置",
    import: "导入",
    prediction_mask_sync: "预测掩码同步",
    review_mask_sync: "人工 Review 同步",
    generated_baseline_promoted: "生成基线转化",
    archive: "归档",
    restore: "恢复归档",
    delete: "删除",
  };
  return map[type] || type || "-";
}

function basename(path) {
  if (!path) return "-";
  return String(path).split(/[\\/]/).pop();
}

function shortPayload(payload) {
  const value = payload || {};
  const parts = [];
  if (value.project_name) parts.push(value.project_name);
  if (value.defect_class) parts.push(value.defect_class);
  if (Number.isFinite(value.count)) parts.push(`样本 ${value.count}`);
  if (Number.isFinite(value.reviewed)) parts.push(`Review ${value.reviewed}`);
  if (Number.isFinite(value.predicted)) parts.push(`预测 ${value.predicted}`);
  if (Number.isFinite(value.imported)) parts.push(`导入 ${value.imported}`);
  if (value.item_id) parts.push(`图像 ${value.item_id}`);
  if (value.checkpoint_id) parts.push(value.checkpoint_id);
  if (value.archive_id) parts.push(value.archive_id);
  if (Number.isFinite(value.mask_px)) parts.push(`mask ${value.mask_px}px`);
  if (Number.isFinite(value.pred_px)) parts.push(`pred ${value.pred_px}px`);
  return parts.join(" | ") || "-";
}

function selectedRow() {
  return state.rows.find((item) => item.dataset_id === state.selectedDatasetId) || null;
}

function selectedLabelId() {
  const row = selectedRow();
  return row?.label_id || slugLabel(el.defectClass.value.trim() || initialParams.get("label_name") || "spall");
}

function itemAnnotation(item, labelId = selectedLabelId()) {
  const annotation = item?.annotations?.[labelId];
  if (annotation?.path) return annotation;
  if (labelId === "spall" && item?.annotation_path) {
    return { path: item.annotation_path };
  }
  return null;
}

function itemHasPrediction(item) {
  return Boolean(item?.latest_prediction || (item?.predictions && Object.keys(item.predictions).length));
}

function itemStatus(item) {
  if (itemAnnotation(item)) return "reviewed";
  if (itemHasPrediction(item)) return "predicted";
  return "unreviewed";
}

function baseQueryParams() {
  const params = new URLSearchParams();
  if (el.filter.value) params.set("project_type", el.filter.value);
  if (initialParams.get("label_id")) params.set("label_id", initialParams.get("label_id"));
  return params;
}

function eventQueryParams() {
  const params = baseQueryParams();
  if (state.selectedDatasetId || initialParams.get("dataset_id")) {
    params.set("dataset_id", state.selectedDatasetId || initialParams.get("dataset_id"));
  }
  return params;
}

function derivedLabelId(defectClass) {
  return slugLabel(defectClass || "defect");
}

function datasetPayload() {
  const defectClass = el.defectClass.value.trim();
  return {
    name: el.name.value.trim(),
    defect_class: defectClass,
    label_id: derivedLabelId(defectClass),
    label_name: defectClass,
    label_color: el.labelColor.value || "#ff9e94",
    project_type: el.projectType.value,
  };
}

function defaultProjectName(defectClass, projectType) {
  return defectClass ? `${defectClass} ${projectTypeText(projectType)}数据集` : "";
}

function applyContextDefaults() {
  const projectType = initialParams.get("project_type") || initialParams.get("workspace") || el.filter.value || "generation";
  const defectClass = initialParams.get("defect_class") || initialParams.get("label_name") || initialParams.get("label_id") || "";
  const projectName = initialParams.get("project_name") || "";
  el.filter.value = initialParams.get("project_type") || "";
  el.projectType.value = projectType === "optimization" ? "optimization" : "generation";
  el.defectClass.value = defectClass;
  el.name.value = projectName || defaultProjectName(defectClass, el.projectType.value);
  el.labelColor.value = "#ff9e94";
  el.sourceDir.value = "";
  el.formStatus.textContent = "未选中数据集：填写项目、缺陷类别、颜色和图像目录后创建数据集。";
}

function setSelected(row) {
  if (!row) {
    state.selectedDatasetId = null;
    state.selectedItemId = null;
    state.items = [];
    el.formStatus.textContent = "未选中数据集：填写项目、缺陷类别、颜色和图像目录后创建数据集。";
    renderTable(state.rows);
    renderItems();
    return;
  }
  state.selectedDatasetId = row.dataset_id;
  el.name.value = row.project_name || "";
  el.defectClass.value = row.defect_class || row.label_name || "";
  el.labelColor.value = row.label_color || "#ff9e94";
  el.projectType.value = row.project_type || "generation";
  el.formStatus.textContent = `已选中：${row.project_name || row.dataset_id}`;
  renderTable(state.rows);
  loadItems(row.dataset_id).catch((err) => {
    el.browserSummary.textContent = `样本加载失败：${err.message}`;
  });
}

function renderTable(rows) {
  if (!rows.length) {
    el.table.innerHTML = '<div class="muted">暂无数据集</div>';
    return;
  }
  el.table.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>项目名称</th>
          <th>缺陷类别</th>
          <th>颜色</th>
          <th>阶段</th>
          <th>数量</th>
          <th>状态</th>
          <th>格式</th>
          <th>最好 PT</th>
          <th>操作</th>
        </tr>
      </thead>
      <tbody>
        ${rows.map((row) => `
          <tr class="${row.dataset_id === state.selectedDatasetId ? "selected-row" : ""}">
            <td>
              <strong>${escapeHtml(row.project_name || row.dataset_id)}</strong>
              <span>${escapeHtml(row.dataset_id)}</span>
            </td>
            <td>${escapeHtml(row.defect_class || row.label_name || "-")}</td>
            <td><span class="catalog-label"><i style="background:${escapeHtml(row.label_color || "#b7c6d5")}"></i>${escapeHtml(row.label_color || "-")}</span></td>
            <td>${projectTypeText(row.project_type)}</td>
            <td>${Number(row.reviewed || 0)}/${Number(row.count || 0)}</td>
            <td><span class="badge ${escapeHtml(row.status)}">${statusText(row.status)}</span></td>
            <td>${escapeHtml(row.format || "-")}</td>
            <td title="${escapeHtml(row.best_pt || "")}">${escapeHtml(basename(row.best_pt))}</td>
            <td>
              <button data-action="open" data-id="${escapeHtml(row.dataset_id)}">查看</button>
              <button data-action="archive" data-id="${escapeHtml(row.dataset_id)}">归档</button>
              <button data-action="delete" data-id="${escapeHtml(row.dataset_id)}">删除</button>
            </td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function renderArchives(rows) {
  if (!rows.length) {
    el.archives.innerHTML = '<div class="muted">暂无归档项目</div>';
    return;
  }
  el.archives.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>归档时间</th>
          <th>项目</th>
          <th>缺陷类别</th>
          <th>阶段</th>
          <th>归档 ID</th>
          <th>操作</th>
        </tr>
      </thead>
      <tbody>
        ${rows.map((row) => `
          <tr>
            <td>${new Date(row.created_at).toLocaleString()}</td>
            <td>${escapeHtml(row.name || row.dataset_id)}</td>
            <td>${escapeHtml(row.defect_class || (row.labels || [])[0]?.name || "-")}</td>
            <td>${projectTypeText(row.project_type)}</td>
            <td>${escapeHtml(row.id)}</td>
            <td><button data-action="restore" data-id="${escapeHtml(row.id)}">恢复</button></td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function renderEvents(rows) {
  if (!rows.length) {
    el.events.innerHTML = '<div class="muted">暂无事件</div>';
    return;
  }
  el.events.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>时间</th>
          <th>事件</th>
          <th>数据集</th>
          <th>阶段</th>
          <th>摘要</th>
        </tr>
      </thead>
      <tbody>
        ${rows.map((row) => `
          <tr>
            <td>${new Date(row.created_at).toLocaleString()}</td>
            <td>${eventTypeText(row.event_type)}</td>
            <td>${escapeHtml(row.dataset_id)}</td>
            <td>${projectTypeText(row.project_type)}</td>
            <td>${escapeHtml(shortPayload(row.payload))}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function filteredItems() {
  const filter = el.itemFilter.value;
  if (!filter) return state.items;
  return state.items.filter((item) => {
    if (filter === "reviewed") return Boolean(itemAnnotation(item));
    if (filter === "unreviewed") return !itemAnnotation(item);
    if (filter === "predicted") return itemHasPrediction(item);
    return true;
  });
}

function renderItems() {
  const row = selectedRow();
  if (!row) {
    el.browserSummary.textContent = "选择一个数据集后查看图像、掩码和 Review 状态。";
    el.itemGrid.innerHTML = '<div class="muted">暂无选中数据集</div>';
    el.itemPreview.innerHTML = "";
    return;
  }
  const reviewed = state.items.filter((item) => itemAnnotation(item)).length;
  const predicted = state.items.filter((item) => itemHasPrediction(item)).length;
  el.browserSummary.textContent = `${row.project_name || row.dataset_id} | ${row.defect_class || row.label_name || "-"} | ${state.items.length} 张 | Review ${reviewed} | 预测 ${predicted}`;
  const items = filteredItems();
  if (!items.length) {
    el.itemGrid.innerHTML = '<div class="muted">当前筛选下没有样本</div>';
    renderPreview(null);
    return;
  }
  if (!state.selectedItemId || !items.some((item) => item.id === state.selectedItemId)) {
    state.selectedItemId = items[0].id;
  }
  el.itemGrid.innerHTML = items.map((item) => {
    const status = itemStatus(item);
    const annotation = itemAnnotation(item);
    return `
      <button class="dataset-item-card ${item.id === state.selectedItemId ? "selected" : ""}" data-item-id="${escapeHtml(item.id)}">
        <img src="/api/datasets/${encodeURIComponent(row.dataset_id)}/items/${encodeURIComponent(item.id)}/image" alt="${escapeHtml(item.original_name || item.id)}" loading="lazy" />
        <span class="badge ${status}">${statusText(status)}</span>
        <strong>${escapeHtml(item.original_name || item.id)}</strong>
        <small>${escapeHtml(item.source_format || "-")} · ${item.width || "-"}x${item.height || "-"}</small>
        <small>${annotation ? "最新 GT 已同步" : "无 GT 掩码"}</small>
      </button>
    `;
  }).join("");
  renderPreview(items.find((item) => item.id === state.selectedItemId) || items[0]);
}

function renderPreview(item) {
  const row = selectedRow();
  if (!row || !item) {
    el.itemPreview.innerHTML = '<div class="muted">选择样本后查看原图和最新 GT。</div>';
    return;
  }
  const labelId = selectedLabelId();
  const annotation = itemAnnotation(item, labelId);
  const predictionKeys = Object.keys(item.predictions || {});
  const latestPrediction = item.latest_prediction || predictionKeys[predictionKeys.length - 1] || "";
  el.itemPreview.innerHTML = `
    <div class="item-preview-title">
      <strong>${escapeHtml(item.original_name || item.id)}</strong>
      <span class="badge ${itemStatus(item)}">${statusText(itemStatus(item))}</span>
    </div>
    <dl>
      <div><dt>样本 ID</dt><dd>${escapeHtml(item.id)}</dd></div>
      <div><dt>缺陷类别</dt><dd>${escapeHtml(row.defect_class || row.label_name || "-")}</dd></div>
      <div><dt>格式</dt><dd>${escapeHtml(item.source_format || "-")}</dd></div>
      <div><dt>尺寸</dt><dd>${item.width || "-"} x ${item.height || "-"}</dd></div>
      <div><dt>最新预测</dt><dd>${escapeHtml(latestPrediction || "-")}</dd></div>
    </dl>
    <div class="dataset-preview-images">
      <figure>
        <img src="/api/datasets/${encodeURIComponent(row.dataset_id)}/items/${encodeURIComponent(item.id)}/image" alt="原图" />
        <figcaption>原图</figcaption>
      </figure>
      <figure>
        ${annotation
          ? `<img src="/api/datasets/${encodeURIComponent(row.dataset_id)}/items/${encodeURIComponent(item.id)}/annotation?label_id=${encodeURIComponent(labelId)}" alt="最新 GT 掩码" />`
          : '<div class="mask-empty">未 Review<br />暂无 GT</div>'}
        <figcaption>最新 GT 掩码</figcaption>
      </figure>
    </div>
  `;
}

async function loadItems(datasetId) {
  el.browserSummary.textContent = "样本同步中";
  state.items = await api.get(`/api/datasets/${encodeURIComponent(datasetId)}/items`);
  if (!state.items.some((item) => item.id === state.selectedItemId)) {
    state.selectedItemId = state.items[0]?.id || null;
  }
  renderItems();
}

async function refresh() {
  const catalogParams = baseQueryParams();
  const eventParams = eventQueryParams();
  const catalogQuery = catalogParams.toString() ? `?${catalogParams}` : "";
  const eventQuery = eventParams.toString() ? `?${eventParams}` : "";
  const [rows, archives, events] = await Promise.all([
    api.get(`/api/datasets/catalog${catalogQuery}`),
    api.get(`/api/dataset-archives${catalogQuery}`),
    api.get(`/api/datasets/events${eventQuery}`),
  ]);
  state.rows = rows;
  if (state.selectedDatasetId && !rows.some((row) => row.dataset_id === state.selectedDatasetId)) {
    state.selectedDatasetId = null;
    state.items = [];
    state.selectedItemId = null;
  }
  renderTable(rows);
  renderArchives(archives);
  renderEvents(events);
  if (state.selectedDatasetId) {
    await loadItems(state.selectedDatasetId);
  } else {
    renderItems();
  }
  el.health.textContent = `已同步 | ${rows.length} 项 | 归档 ${archives.length} | 事件 ${events.length}`;
}

async function createDataset() {
  const payload = datasetPayload();
  if (!payload.name || !payload.defect_class) {
    el.formStatus.textContent = "请先填写项目名称和缺陷类别";
    return;
  }
  if (!el.sourceDir.value.trim()) {
    el.formStatus.textContent = "请填写图像文件夹；如只需空目录，请先创建后直接保存选中数据集。";
    return;
  }
  el.formStatus.textContent = "正在创建并导入";
  const result = await api.post("/api/datasets/create-and-import", {
    ...payload,
    source_dir: el.sourceDir.value.trim(),
  });
  state.selectedDatasetId = result.dataset.id;
  el.filter.value = payload.project_type;
  el.formStatus.textContent = `已创建 ${result.dataset.name}，导入 ${result.import?.imported || 0} 张`;
  await refresh();
}

async function saveSettings() {
  if (!state.selectedDatasetId) {
    const payload = datasetPayload();
    if (!payload.name || !payload.defect_class) {
      el.formStatus.textContent = "请先填写项目名称和缺陷类别";
      return;
    }
    const created = await api.post("/api/datasets", payload);
    state.selectedDatasetId = created.id;
    el.filter.value = payload.project_type;
    el.formStatus.textContent = `已创建空目录：${created.name}`;
    await refresh();
    return;
  }
  const payload = datasetPayload();
  delete payload.project_type;
  const updated = await api.patch(`/api/datasets/${encodeURIComponent(state.selectedDatasetId)}`, payload);
  el.formStatus.textContent = `已保存：${updated.name}`;
  await refresh();
}

async function importIntoSelected() {
  if (!state.selectedDatasetId) {
    el.formStatus.textContent = "请先选择要导入的数据集";
    return;
  }
  if (!el.sourceDir.value.trim()) {
    el.formStatus.textContent = "请填写图像文件夹";
    return;
  }
  el.formStatus.textContent = "正在导入到选中数据集";
  const result = await api.post(`/api/datasets/${encodeURIComponent(state.selectedDatasetId)}/import`, {
    source_dir: el.sourceDir.value.trim(),
  });
  el.formStatus.textContent = `已导入 ${result.imported} 张，跳过 ${result.skipped || 0} 张`;
  await refresh();
}

el.defectClass.addEventListener("input", () => {
  const label = el.defectClass.value.trim();
  if (label && !state.selectedDatasetId && !el.name.value.trim()) {
    el.name.value = defaultProjectName(label, el.projectType.value);
  }
});

el.projectType.addEventListener("change", () => {
  if (!state.selectedDatasetId && el.defectClass.value.trim()) {
    el.name.value = defaultProjectName(el.defectClass.value.trim(), el.projectType.value);
  }
});

el.table.onclick = async (evt) => {
  const button = evt.target.closest("button[data-action]");
  const rowEl = evt.target.closest("tr");
  if (!button && rowEl) {
    const index = [...rowEl.parentElement.children].indexOf(rowEl);
    const row = state.rows[index];
    if (row) setSelected(row);
    return;
  }
  if (!button) return;
  const datasetId = button.dataset.id;
  const row = state.rows.find((item) => item.dataset_id === datasetId);
  if (button.dataset.action === "open") {
    setSelected(row);
    return;
  }
  if (button.dataset.action === "archive") {
    el.health.textContent = "归档中";
    await api.post(`/api/datasets/${encodeURIComponent(datasetId)}/archive`, {});
    await refresh();
    return;
  }
  if (button.dataset.action === "delete") {
    const ok = window.confirm(`永久删除平台数据集「${datasetId}」？\n\n删除会移除平台内图像副本、训练任务和模型测试，不影响外部原始图像文件夹。`);
    if (!ok) return;
    el.health.textContent = "删除中";
    await api.delete(`/api/datasets/${encodeURIComponent(datasetId)}`);
    await refresh();
  }
};

el.itemGrid.onclick = (evt) => {
  const card = evt.target.closest("button[data-item-id]");
  if (!card) return;
  state.selectedItemId = card.dataset.itemId;
  renderItems();
};

el.itemFilter.onchange = renderItems;

el.archives.onclick = async (evt) => {
  const button = evt.target.closest("button[data-action='restore']");
  if (!button) return;
  el.health.textContent = "恢复归档中";
  const projectType = encodeURIComponent(el.filter.value || el.projectType.value);
  const restored = await api.post(`/api/dataset-archives/${encodeURIComponent(button.dataset.id)}/restore?project_type=${projectType}`, {});
  state.selectedDatasetId = restored.id;
  await refresh();
};

el.filter.onchange = () => refresh().catch((err) => {
  el.health.textContent = `同步失败：${err.message}`;
});

el.refresh.onclick = () => refresh().catch((err) => {
  el.health.textContent = `同步失败：${err.message}`;
});

el.newMode.onclick = () => {
  state.selectedDatasetId = null;
  state.selectedItemId = null;
  state.items = [];
  applyContextDefaults();
  setSelected(null);
};

el.saveSettings.onclick = () => saveSettings().catch((err) => {
  el.formStatus.textContent = `保存失败：${err.message}`;
});

el.createImport.onclick = () => createDataset().catch((err) => {
  el.formStatus.textContent = `导入失败：${err.message}`;
});

el.importSelected.onclick = () => importIntoSelected().catch((err) => {
  el.formStatus.textContent = `导入失败：${err.message}`;
});

el.backHome.onclick = () => {
  window.location.href = selectedReturnUrl();
};

applyContextDefaults();
refresh().then(() => {
  const initialDataset = initialParams.get("dataset_id");
  if (initialDataset) {
    setSelected(state.rows.find((row) => row.dataset_id === initialDataset));
  }
}).catch((err) => {
  el.health.textContent = `同步失败：${err.message}`;
  console.error(err);
});
