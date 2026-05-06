# TruFor Setup (Local Project Path)

This project expects TruFor files under `models/trufor/` so cloned repos can run with relative paths.

## 1) Clone TruFor repository

```bash
cd models/trufor
git clone https://github.com/grip-unina/TruFor.git
```

Expected code path after clone:

`models/trufor/TruFor/TruFor_train_test`

## 2) Install TruFor dependencies

Follow the official README inference section:

https://github.com/grip-unina/TruFor/blob/main/TruFor_train_test/README.md#inference

Install in your local Python environment (not Colab).

## 3) Download weights

Module B will try to auto-download weights from:

`https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip`

You can also download manually (recommended fallback), then place:

`trufor.pth.tar` at  
`models/trufor/TruFor/TruFor_train_test/pretrained_models/trufor.pth.tar`

## 4) Optional `.env` overrides

```env
TRUFOR_ENABLE=1
TRUFOR_MODEL_DIR=models/trufor
TRUFOR_REPO_DIR=models/trufor/TruFor
TRUFOR_WORK_DIR=models/trufor/TruFor/TruFor_train_test
TRUFOR_MODEL_FILE=models/trufor/TruFor/TruFor_train_test/pretrained_models/trufor.pth.tar
TRUFOR_AUTO_DOWNLOAD_WEIGHTS=1
TRUFOR_GPU_ID=-1
```
