# Colab Setup Guide

## 1) Clone branch in Colab
Public repo:
```bash
!git clone -b colab-train https://github.com/<your-username>/npt-model.git
```

Private repo (PAT):
```bash
!git clone -b colab-train https://<GITHUB_USERNAME>:<GITHUB_PAT>@github.com/<your-username>/npt-model.git
```

## 2) Install dependencies
```bash
%cd npt-model
!pip install -r requirements_colab.txt
```

## 3) Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

## 4) Sync to Drive folder
```bash
!mkdir -p /content/drive/MyDrive/npt-model
!rsync -a --delete /content/npt-model/ /content/drive/MyDrive/npt-model/
%cd /content/drive/MyDrive/npt-model
```

## 5) Download data
```bash
!python3 download_data.py
!ls -lh data/raw/train_data.txt
```

## 6) Train
```bash
!python3 scripts/colab_train.py --drive-root /content/drive/MyDrive/npt-model --tokenizer-name deepseek-ai/DeepSeek-V4-Pro --epochs 1 --dataset-size 5000 --seq-length 64 --batch-size 2
```

## 7) Start API
```bash
!TOKENIZER_NAME=deepseek-ai/DeepSeek-V4-Pro python3 inference/server.py
```
