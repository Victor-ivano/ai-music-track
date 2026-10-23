# AI Music Track — GuitarAI

目標是把一首混音歌曲分離成七個可獨立編輯的音軌：人聲、主奏吉他、節奏吉他、Bass、鼓組、鍵盤與其他樂器。

## 目前進度（2026-10-23）

目前已成功跑通第一個預訓練模型基線：使用 [Music-Source-Separation-Training（MSST）](https://github.com/ZFTurbo/Music-Source-Separation-Training) 專案中的 MVSEP Mega 53-stem BS-RoFormer。這是**外部預訓練模型的推論測試**，不是已經訓練完成的 GuitarAI 七軌模型。

- 在 Kaggle GPU 環境中，成功處理一段 20.59 秒、44.1 kHz 的測試音訊。
- 一次推論產生 53 個樂器／聲部分類的 WAV 檔。
- 已記錄一些輸出品質問題：單看 `drums.wav` 與 `snare.wav` 時能量偏低，部分鼓件分散在 `kick.wav`、`hh.wav`、`toms.wav` 等分類。
- 現有模型沒有獨立的「主奏吉他」與「節奏吉他」類別，因此目前還不能可靠地輸出真正獨立的兩條吉他音軌。

重現此實驗的指令與測試觀察紀錄見 [實驗文件](experiments/mvsep_53_stems/README.md)。

## 目標七軌

- `vocals.wav` — 人聲
- `lead_guitar.wav` — 主奏吉他
- `rhythm_guitar.wav` — 節奏吉他
- `bass.wav` — 貝斯
- `drums.wav` — 鼓組
- `keyboard.wav` — 鍵盤
- `other.wav` — 其他樂器

以上是專案目標，不代表目前 53-stem 基線已能穩定做到七軌分離。後處理時不能單憑檔名就假設輸出乾淨，也不應把相同的吉他 stem 複製成兩軌後宣稱已完成主奏／節奏分離。

## 推論環境

目前實測環境為 Kaggle Notebook、Python 3.13.15、PyTorch 2.11.0+cu128、NVIDIA T4 GPU。測試使用一張 GPU（`--device_ids 0`）；多張 GPU 不代表 VRAM 會自動合併。

本專案沒有複製 MSST 的整個原始碼庫，也沒有把預訓練權重或音訊樣本提交到 GitHub。模型權重約 1.275 GiB，應依上游專案提供的方式取得，而不是直接放進一般 Git 儲存庫。

## 專案檔案

```text
.
├── data/
│   └── README.md
├── docs/
│   └── progress.md
├── experiments/
│   └── mvsep_53_stems/
│       ├── README.md
│       └── run_kaggle_inference.py
├── .gitignore
├── README.md
└── requirements.txt
```

## 原先的自訓練方向

長期目標仍是建立可自行訓練、針對樂團編制的七軌模型，並要求混音與各個 stem 精確對齊。每首歌曲的資料夾格式應為：

```text
data/track001/
├── mix.wav
├── vocals.wav
├── lead_guitar.wav
├── rhythm_guitar.wav
├── bass.wav
├── drums.wav
├── keyboard.wav
└── other.wav
```

所有音檔必須涵蓋相同時間範圍、速度與取樣率，並精確對齊。沒有該樂器的 stem 可以是靜音。測試資料、音訊、模型權重與生成輸出不應提交到 GitHub。
