const tabUrl = document.getElementById('tab-url');
const tabImage = document.getElementById('tab-image');
const urlSection = document.getElementById('url-section');
const imageSection = document.getElementById('image-section');
const imageUrlInput = document.getElementById('imageUrl');
const urlNoteInput = document.getElementById('urlNote');
const imageFileInput = document.getElementById('imageFile');
const uploadArea = document.getElementById('uploadArea');
const previewBox = document.getElementById('previewBox');
const previewImg = document.getElementById('previewImg');
const fileName = document.getElementById('fileName');
const fileInfo = document.getElementById('fileInfo');
const analyzeBtn = document.getElementById('analyzeBtn');
const resetBtn = document.getElementById('resetBtn');
const resultBox = document.getElementById('resultBox');
const resultTitle = document.getElementById('resultTitle');
const resultText = document.getElementById('resultText');

let activeMode = 'url';

function switchTab(mode) {
  activeMode = mode;
  const isUrl = mode === 'url';
  tabUrl.classList.toggle('active', isUrl);
  tabImage.classList.toggle('active', !isUrl);
  urlSection.classList.toggle('hidden', !isUrl);
  imageSection.classList.toggle('hidden', isUrl);
  hideResult();
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function showResult(title, text, isError = false) {
  resultTitle.textContent = title;
  resultText.textContent = text;
  resultBox.classList.add('show');
  resultBox.classList.toggle('error', isError);
}

function hideResult() {
  resultBox.classList.remove('show', 'error');
}

function loadFilePreview(file) {
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    fileName.textContent = file.name;
    fileInfo.textContent = `格式：${file.type || '未知'} ｜ 大小：${formatSize(file.size)}`;
    previewBox.classList.add('show');
  };
  reader.readAsDataURL(file);
}

function resetAll() {
  imageUrlInput.value = '';
  urlNoteInput.value = '';
  imageFileInput.value = '';
  previewImg.src = '';
  previewBox.classList.remove('show');
  fileName.textContent = '未選擇檔案';
  fileInfo.textContent = '檔案資訊將顯示於此。';
  hideResult();
  switchTab('url');
}

tabUrl.addEventListener('click', () => switchTab('url'));
tabImage.addEventListener('click', () => switchTab('image'));

imageFileInput.addEventListener('change', (event) => {
  const file = event.target.files[0];
  loadFilePreview(file);
  if (file) hideResult();
});

['dragenter', 'dragover'].forEach((eventName) => {
  uploadArea.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.add('dragover');
  });
});

['dragleave', 'drop'].forEach((eventName) => {
  uploadArea.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('dragover');
  });
});

uploadArea.addEventListener('drop', (e) => {
  const files = e.dataTransfer.files;
  if (files && files[0] && files[0].type.startsWith('image/')) {
    imageFileInput.files = files;
    loadFilePreview(files[0]);
    hideResult();
  } else {
    showResult('格式不支援', '請拖曳圖片檔案，例如 JPG、PNG 或 WEBP。', true);
  }
});

analyzeBtn.addEventListener('click', () => {
  if (activeMode === 'url') {
    const imageUrl = imageUrlInput.value.trim();
    const note = urlNoteInput.value.trim();

    if (!imageUrl) {
      showResult('缺少圖片網址', '請先輸入要分析的圖片 URL。', true);
      return;
    }

    try {
      new URL(imageUrl);
    } catch {
      showResult('網址格式錯誤', '請輸入有效的圖片連結。', true);
      return;
    }

    showResult(
      '已建立分析請求',
      `目前已接收圖片網址。${note ? ' 備註內容也已一併記錄。' : ''}`
    );
  } else {
    const file = imageFileInput.files[0];

    if (!file) {
      showResult('尚未上傳圖片', '請先選擇或拖曳一張圖片。', true);
      return;
    }

    showResult(
      '圖片已準備完成',
      `已載入「${file.name}」，檔案大小為 ${formatSize(file.size)}。`
    );
  }
});

resetBtn.addEventListener('click', resetAll);
