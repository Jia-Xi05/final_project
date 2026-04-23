const imageFileInput = document.getElementById("imageFile");
const uploadArea = document.getElementById("uploadArea");
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
const moduleName = document.getElementById("moduleName");
const detectionCount = document.getElementById("detectionCount");
const faceCount = document.getElementById("faceCount");
const originalResultImg = document.getElementById("originalResultImg");
const annotatedResultImg = document.getElementById("annotatedResultImg");
const detectionTableBody = document.getElementById("detectionTableBody");
const faceTableBody = document.getElementById("faceTableBody");

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

function hideAnalysisResults() {
  resultsPanel.classList.add("hidden");
  moduleName.textContent = "-";
  detectionCount.textContent = "0";
  faceCount.textContent = "0";
  clearImage(originalResultImg);
  clearImage(annotatedResultImg);
  detectionTableBody.innerHTML = `
    <tr>
      <td colspan="4">No object detections yet.</td>
    </tr>
  `;
  faceTableBody.innerHTML = `
    <tr>
      <td colspan="4">No face detections yet.</td>
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

function renderAnalysisResult(result) {
  moduleName.textContent = result.module || "yolo";
  detectionCount.textContent = String(result.num_detections ?? 0);
  faceCount.textContent = String(result.num_faces ?? 0);
  originalResultImg.src = result.original_image_url;
  annotatedResultImg.src = result.annotated_image_url;
  renderDetectionTable(result.detections || []);
  renderFaceTable(result.face_detections || []);
  resultsPanel.classList.remove("hidden");
}

function resetAll() {
  imageFileInput.value = "";
  clearImage(previewImg);
  previewBox.classList.remove("show");
  fileName.textContent = "No file selected";
  fileInfo.textContent = "Please upload a test image.";
  hideResult();
  hideAnalysisResults();
  analyzeBtn.disabled = false;
  analyzeBtn.textContent = "Run Analysis";
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
  showResult("Processing", "Running YOLO object detection and SCRFD face detection.");

  try {
    const response = await fetch("/api/analyze/yolo", {
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
    showResult(
      "Completed",
      `Detected ${payload.num_detections ?? 0} objects and ${payload.num_faces ?? 0} faces.`
    );
  } catch (error) {
    showResult("Analysis Failed", error.message || "Unable to connect to the backend API.", true);
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Run Analysis";
  }
}

imageFileInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  loadFilePreview(file);
  hideResult();
  hideAnalysisResults();
});

["dragenter", "dragover"].forEach((eventName) => {
  uploadArea.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    uploadArea.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  uploadArea.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    uploadArea.classList.remove("dragover");
  });
});

uploadArea.addEventListener("drop", (event) => {
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

analyzeBtn.addEventListener("click", analyzeImage);
resetBtn.addEventListener("click", resetAll);

hideAnalysisResults();
