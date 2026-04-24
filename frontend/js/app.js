const imageFileInput = document.getElementById("imageFile");
const uploadArea = document.getElementById("uploadArea");
const modeImageInput = document.getElementById("modeImage");
const modeUrlInput = document.getElementById("modeUrl");
const imageInputCopy = document.getElementById("imageInputCopy");
const imagePickerBtn = document.getElementById("imagePickerBtn");
const urlInputWrap = document.getElementById("urlInputWrap");
const imageUrlInput = document.getElementById("imageUrlInput");
const previewBox = document.getElementById("previewBox");
const previewImg = document.getElementById("previewImg");
const fileName = document.getElementById("fileName");
const fileInfo = document.getElementById("fileInfo");
const analyzeBtn = document.getElementById("analyzeBtn");
const resetBtn = document.getElementById("resetBtn");
const resultBox = document.getElementById("resultBox");
const resultTitle = document.getElementById("resultTitle");
const resultText = document.getElementById("resultText");
const resultsPanel = document.getElementById("resultsPanel");

const authenticityLabel = document.getElementById("authenticityLabel");
const headlineText = document.getElementById("headlineText");
const riskScorePercent = document.getElementById("riskScorePercent");
const riskLevelText = document.getElementById("riskLevelText");
const visionCounts = document.getElementById("visionCounts");
const reportSource = document.getElementById("reportSource");
const modulePipelineName = document.getElementById("modulePipelineName");

const moduleAStatus = document.getElementById("moduleAStatus");
const moduleASummary = document.getElementById("moduleASummary");
const moduleAChips = document.getElementById("moduleAChips");
const moduleBStatus = document.getElementById("moduleBStatus");
const moduleBSummary = document.getElementById("moduleBSummary");
const moduleBChips = document.getElementById("moduleBChips");
const moduleCStatus = document.getElementById("moduleCStatus");
const moduleCSummary = document.getElementById("moduleCSummary");
const moduleCChips = document.getElementById("moduleCChips");

const originalResultImg = document.getElementById("originalResultImg");
const annotatedResultImg = document.getElementById("annotatedResultImg");
const reportText = document.getElementById("reportText");
const ocrText = document.getElementById("ocrText");
const evidenceList = document.getElementById("evidenceList");
const detectionTableBody = document.getElementById("detectionTableBody");
const faceTableBody = document.getElementById("faceTableBody");
const visionScoreTableBody = document.getElementById("visionScoreTableBody");

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function formatScore(score) {
  return `${(Number(score) * 100).toFixed(2)}%`;
}

function showResult(title, text, isError = false) {
  resultTitle.textContent = title;
  resultText.textContent = text;
  resultBox.classList.add("show");
  resultBox.classList.toggle("error", isError);
}

function hideResult() {
  resultBox.classList.remove("show", "error");
}

function clearImage(imgElement) {
  imgElement.removeAttribute("src");
}

function resetChipContainer(element, fallback) {
  element.innerHTML = `<span class="chip muted">${fallback}</span>`;
}

function hideAnalysisResults() {
  resultsPanel.classList.add("hidden");
  authenticityLabel.textContent = "-";
  headlineText.textContent = "Awaiting analysis.";
  riskScorePercent.textContent = "0";
  riskLevelText.textContent = "Risk level: -";
  visionCounts.textContent = "0 / 0";
  reportSource.textContent = "-";
  modulePipelineName.textContent = "Router + Module A + Module C + Aggregator";
  moduleAStatus.textContent = "-";
  moduleASummary.textContent = "Awaiting output.";
  moduleBStatus.textContent = "-";
  moduleBSummary.textContent = "Awaiting output.";
  moduleCStatus.textContent = "-";
  moduleCSummary.textContent = "Awaiting output.";
  reportText.textContent = "Awaiting report.";
  ocrText.textContent = "-";
  evidenceList.innerHTML = "<li>No evidence yet.</li>";
  clearImage(originalResultImg);
  clearImage(annotatedResultImg);
  resetChipContainer(moduleAChips, "No labels yet");
  resetChipContainer(moduleBChips, "No face metrics yet");
  resetChipContainer(moduleCChips, "No fusion signals yet");

  detectionTableBody.innerHTML = `
    <tr>
      <td colspan="4">No detections yet.</td>
    </tr>
  `;
  faceTableBody.innerHTML = `
    <tr>
      <td colspan="4">No faces yet.</td>
    </tr>
  `;
  visionScoreTableBody.innerHTML = `
    <tr>
      <td colspan="2">No OpenCLIP semantic result yet.</td>
    </tr>
  `;
}

function loadFilePreview(file) {
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (event) => {
    previewImg.src = event.target.result;
    fileName.textContent = file.name;
    fileInfo.textContent = `${file.type || "unknown"} | ${formatSize(file.size)}`;
    previewBox.classList.add("show");
  };
  reader.readAsDataURL(file);
}

function renderDetectionTable(detections) {
  if (!detections.length) {
    detectionTableBody.innerHTML = `
      <tr>
        <td colspan="4">No objects detected.</td>
      </tr>
    `;
    return;
  }

  detectionTableBody.innerHTML = detections
    .map((detection, index) => {
      const bbox = Array.isArray(detection.bbox) ? detection.bbox.join(", ") : "-";
      const className = detection.class_name || `class_${detection.class_id ?? "?"}`;
      return `
        <tr>
          <td>${index + 1}</td>
          <td>${className}</td>
          <td>${formatScore(detection.score)}</td>
          <td>[${bbox}]</td>
        </tr>
      `;
    })
    .join("");
}

function renderFaceTable(faceDetections) {
  if (!faceDetections.length) {
    faceTableBody.innerHTML = `
      <tr>
        <td colspan="4">No faces detected.</td>
      </tr>
    `;
    return;
  }

  faceTableBody.innerHTML = faceDetections
    .map((detection, index) => {
      const bbox = Array.isArray(detection.bbox) ? detection.bbox.join(", ") : "-";
      const landmarks = Array.isArray(detection.landmarks)
        ? detection.landmarks.map((point) => `(${point[0]}, ${point[1]})`).join(" ")
        : "-";

      return `
        <tr>
          <td>${index + 1}</td>
          <td>${formatScore(detection.score)}</td>
          <td>[${bbox}]</td>
          <td>${landmarks}</td>
        </tr>
      `;
    })
    .join("");
}

function renderVisionScores(scores) {
  const entries = Object.entries(scores || {});
  if (!entries.length) {
    visionScoreTableBody.innerHTML = `
      <tr>
        <td colspan="2">Module A did not produce semantic scores.</td>
      </tr>
    `;
    return;
  }

  visionScoreTableBody.innerHTML = entries
    .map(([label, score]) => {
      return `
        <tr>
          <td>${label}</td>
          <td>${formatScore(score)}</td>
        </tr>
      `;
    })
    .join("");
}

function renderChips(container, values, formatter) {
  if (!values || !values.length) {
    container.innerHTML = `<span class="chip muted">No signal</span>`;
    return;
  }

  container.innerHTML = values
    .map((value) => `<span class="chip">${formatter(value)}</span>`)
    .join("");
}

function renderAnalysisResult(result) {
  const modules = result.modules || {};
  const router = modules.router || {};
  const moduleA = modules.a || {};
  const moduleC = modules.c || {};
  const aggregator = modules.aggregator || {};

  authenticityLabel.textContent = result.authenticity_label || "Unknown";
  headlineText.textContent = result.summary?.headline || "Analysis completed.";
  riskScorePercent.textContent = `${Math.round(result.risk_score_percent ?? 0)}`;
  riskLevelText.textContent = `Risk level: ${(result.risk_level || "-").toUpperCase()}`;
  visionCounts.textContent = `${result.num_detections ?? 0} / ${result.num_faces ?? 0}`;
  reportSource.textContent = result.report_source || "rule_based";
  modulePipelineName.textContent = result.pipeline_name || "Router + Module A + Module C + Aggregator";

  moduleAStatus.textContent = router.status || "-";
  moduleASummary.textContent = router.summary || "No router summary.";
  moduleBStatus.textContent = moduleA.status || "-";
  moduleBSummary.textContent = moduleA.summary || "No Module A summary.";
  moduleCStatus.textContent = moduleC.status || "-";
  moduleCSummary.textContent = moduleC.summary || "No Module C summary.";

  renderChips(
    moduleAChips,
    [
      `Route ${router.route_label || "-"}`,
      `Objects ${router.num_detections ?? 0}`,
      `Faces ${router.num_faces ?? 0}`,
      ...(router.routing_flags
        ? Object.entries(router.routing_flags).map(([key, value]) => `${key}:${value ? "on" : "off"}`)
        : []),
    ],
    (item) => item
  );
  renderChips(
    moduleBChips,
    [
      ...(moduleA.signals || []),
      ...(moduleA.serpapi_summary?.top_sources || []).slice(0, 3),
    ],
    (item) => item
  );
  renderChips(
    moduleCChips,
    [
      ...(moduleC.suspicious_keywords || []),
      ...(moduleC.roi_items || []).slice(0, 3).map((item) => `${item.label}@${formatScore(item.score)}`),
    ],
    (item) => item
  );

  originalResultImg.src = result.original_image_url;
  annotatedResultImg.src = result.annotated_image_url;
  reportText.textContent = result.report || aggregator.report || "No report generated.";
  ocrText.textContent = moduleC.ocr_text || "No OCR text extracted.";

  const evidence = aggregator.evidence || moduleC.evidence || [];
  evidenceList.innerHTML = evidence.length
    ? evidence.map((item) => `<li>${item}</li>`).join("")
    : "<li>No evidence provided.</li>";

  renderDetectionTable(result.detections || []);
  renderFaceTable(result.face_detections || []);
  renderVisionScores(moduleA.semantic_scores || {});
  resultsPanel.classList.remove("hidden");
}

function resetAll() {
  imageFileInput.value = "";
  imageUrlInput.value = "";
  clearImage(previewImg);
  previewBox.classList.remove("show");
  fileName.textContent = "No file selected";
  fileInfo.textContent = "Please upload a test image for the project demo.";
  hideResult();
  hideAnalysisResults();
  analyzeBtn.disabled = false;
  analyzeBtn.textContent = "Run Full Analysis";
}

function getCurrentMode() {
  return modeUrlInput.checked ? "url" : "image";
}

function applyInputMode() {
  const mode = getCurrentMode();
  const isImageMode = mode === "image";

  imageInputCopy.classList.toggle("hidden", !isImageMode);
  imagePickerBtn.classList.toggle("hidden", !isImageMode);
  urlInputWrap.classList.toggle("hidden", isImageMode);

  uploadArea.classList.toggle("drag-enabled", isImageMode);
  if (!isImageMode) {
    imageFileInput.value = "";
  } else {
    imageUrlInput.value = "";
  }

  clearImage(previewImg);
  previewBox.classList.remove("show");
  fileName.textContent = isImageMode ? "No file selected" : "No URL selected";
  fileInfo.textContent = isImageMode
    ? "Please upload a test image for the project demo."
    : "Paste a URL and run analysis.";
  hideResult();
  hideAnalysisResults();
}

function isValidHttpUrl(url) {
  try {
    const parsed = new URL(url);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    return false;
  }
}

function loadUrlPreview(rawUrl) {
  const url = (rawUrl || "").trim();
  if (!url) return;
  if (!isValidHttpUrl(url)) return;

  previewImg.src = url;
  fileName.textContent = "URL input";
  fileInfo.textContent = url;
  previewBox.classList.add("show");
}

async function analyzeImage() {
  const file = imageFileInput.files[0];
  if (!file) {
    showResult("No Image", "Please select an image before running analysis.", true);
    return;
  }

  const formData = new FormData();
  formData.append("image", file);

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing...";
  hideAnalysisResults();
  showResult("Processing", "Running router, Module A similarity analysis, Module C OCR/ROI analysis, and the final aggregator.");

  try {
    const response = await fetch("/api/analyze/full", {
      method: "POST",
      body: formData,
    });

    const rawText = await response.text();
    let payload = null;

    try {
      payload = rawText ? JSON.parse(rawText) : null;
    } catch {
      throw new Error(rawText || `Backend returned invalid JSON. HTTP status: ${response.status}`);
    }

    if (!response.ok || payload.status !== "success") {
      throw new Error(payload?.message || "Analysis failed.");
    }

    renderAnalysisResult(payload);
    showResult("Completed", payload.summary?.headline || "Full pipeline completed.");
  } catch (error) {
    showResult("Analysis Failed", error.message || "Unable to connect to the backend API.", true);
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Run Full Analysis";
  }
}

async function analyzeUrl() {
  const url = imageUrlInput.value.trim();
  if (!url) {
    showResult("No URL", "Please enter an URL before running analysis.", true);
    return;
  }
  if (!isValidHttpUrl(url)) {
    showResult("Invalid URL", "URL must start with http:// or https://", true);
    return;
  }

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing...";
  hideAnalysisResults();
  showResult("Processing", "Fetching image from URL, then running router, Module A, Module C, and aggregator.");

  try {
    const response = await fetch("/api/analyze/full-url", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ url }),
    });

    const rawText = await response.text();
    let payload = null;

    try {
      payload = rawText ? JSON.parse(rawText) : null;
    } catch {
      throw new Error(rawText || `Backend returned invalid JSON. HTTP status: ${response.status}`);
    }

    if (!response.ok || payload.status !== "success") {
      throw new Error(payload?.message || "URL analysis failed.");
    }

    renderAnalysisResult(payload);
    showResult("Completed", payload.summary?.headline || "Full pipeline completed from URL.");
  } catch (error) {
    showResult("Analysis Failed", error.message || "Unable to connect to the backend API.", true);
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Run Full Analysis";
  }
}

async function analyzeByCurrentMode() {
  if (getCurrentMode() === "url") {
    await analyzeUrl();
    return;
  }
  await analyzeImage();
}

imageFileInput.addEventListener("change", (event) => {
  if (getCurrentMode() !== "image") return;
  const file = event.target.files[0];
  loadFilePreview(file);
  hideResult();
  hideAnalysisResults();
});

imageUrlInput.addEventListener("input", () => {
  if (getCurrentMode() !== "url") return;
  const url = imageUrlInput.value.trim();
  if (!url) {
    clearImage(previewImg);
    previewBox.classList.remove("show");
    fileName.textContent = "No URL selected";
    fileInfo.textContent = "Paste a URL and run analysis.";
    hideResult();
    hideAnalysisResults();
    return;
  }
  loadUrlPreview(url);
  hideResult();
  hideAnalysisResults();
});

["dragenter", "dragover"].forEach((eventName) => {
  uploadArea.addEventListener(eventName, (event) => {
    if (getCurrentMode() !== "image") return;
    event.preventDefault();
    event.stopPropagation();
    uploadArea.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  uploadArea.addEventListener(eventName, (event) => {
    if (getCurrentMode() !== "image") return;
    event.preventDefault();
    event.stopPropagation();
    uploadArea.classList.remove("dragover");
  });
});

uploadArea.addEventListener("drop", (event) => {
  if (getCurrentMode() !== "image") return;
  const files = event.dataTransfer.files;
  const file = files && files[0];

  if (file && file.type.startsWith("image/")) {
    imageFileInput.files = files;
    loadFilePreview(file);
    hideResult();
    hideAnalysisResults();
    return;
  }

  showResult("Unsupported File", "Please drop a JPG, PNG, or WEBP image.", true);
});

modeImageInput.addEventListener("change", applyInputMode);
modeUrlInput.addEventListener("change", applyInputMode);
analyzeBtn.addEventListener("click", analyzeByCurrentMode);
resetBtn.addEventListener("click", resetAll);

hideAnalysisResults();
applyInputMode();
