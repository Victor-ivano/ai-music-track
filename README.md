# AI Music Track — GuitarAI

## 專案目標
開發自製深度學習音樂源分離模型，將歌曲分離為七個音軌。

## 目標輸出
- vocals.wav：主唱
- lead_guitar.wav：主奏吉他
- rhythm_guitar.wav：節奏吉他
- bass.wav：貝斯
- drums.wav：鼓組
- keyboard.wav：鍵盤
- other.wav：其他樂器

## 開發環境
- Windows 11
- Python 3.11
- PyTorch 2.5.0
- CUDA 12.4
- NVIDIA GeForce RTX 3050 Laptop GPU

## 開發狀態
目前處於專案初始化與模型設計階段。

## 訓練資料
每首歌曲需包含混音檔與七個對應音軌，且音檔必須時間對齊。