# MVSEP 53-stem 預訓練模型實驗

## 目的

用公開的預訓練模型建立 GuitarAI 的第一個推論基線，驗證音訊讀取、模型推論及 stem 輸出流程。模型與設定檔來自上游專案，不是本專案自行訓練的權重。

上游專案：[ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)

## 模型檔案

請先依照上游專案的授權與下載指示取得以下檔案，**不要把權重提交到本儲存庫**：

- 設定檔：`pretrained/mvsep_53_stems/mvsep_mega_model_bs_roformer_53_stems.yaml`
- 權重：`pretrained/mvsep_53_stems/mvsep_mega_model_bs_roformer_53_stems_v1.ckpt`

預期的 MSST 目錄大致如下：

```text
Music-Source-Separation-Training/
├── inference.py
└── pretrained/
    └── mvsep_53_stems/
        ├── mvsep_mega_model_bs_roformer_53_stems.yaml
        └── mvsep_mega_model_bs_roformer_53_stems_v1.ckpt
```

權重約 1.275 GiB；請從上游正式來源下載，並確認檔案完整。

## Kaggle 實測環境

- Python 3.13.15
- PyTorch 2.11.0+cu128
- NVIDIA T4 GPU（本次推論使用裝置 0）
- 測試音訊：20.59 秒、44,100 Hz
- 本次推論完成時間：約 35.49 秒
- 輸出：53 個 WAV stem

MSST 的完整 requirements 在此 Python 3.13 環境曾因 `pedalboard~=0.8.1` 找不到相容套件而安裝失敗。不要為了這個實驗直接重新安裝或降級 Kaggle 內建的 PyTorch。若遇到缺少模組，可先依實際錯誤補裝推論所需套件；此環境曾使用 `beartype`、`einops`、`packaging`、`rotary-embedding-torch` 與 `loralib`。

## 執行方式

1. 在 Kaggle clone 上游 MSST repository，並取得以上設定檔與權重。
2. 將待分離的音訊 WAV 放入 MSST 目錄下的 `input_test/`。若使用其他資料夾，執行時指定 `--input-folder`。
3. 從本專案根目錄執行：

```bash
python experiments/mvsep_53_stems/run_kaggle_inference.py \
  --msst-dir /kaggle/working/Music-Source-Separation-Training \
  --input-folder /kaggle/working/Music-Source-Separation-Training/input_test \
  --output-dir /kaggle/working/Music-Source-Separation-Training/separation_results \
  --device-ids 0
```

腳本會檢查上游程式、設定檔、權重與輸入資料夾是否存在，之後呼叫上游的 `inference.py`。真正的 stem 會由上游模型輸出到指定資料夾。

## 目前觀察與限制

本次樣本中的部分輸出能量：

| Stem | RMS |
|---|---:|
| `bass.wav` | 0.031577 |
| `drums.wav` | 0.002575 |
| `kick.wav` | 0.021041 |
| `hh.wav` | 0.006063 |
| `snare.wav` | 0.000184 |
| `toms.wav` | 0.002514 |
| `guitar.wav` | 0.048149 |
| `acoustic-guitar.wav` | 0.046782 |
| `electric-guitar.wav` | 0.001351 |

RMS 是整段訊號的能量指標，不等於人耳主觀音量，也不能證明該 stem 已正確分離。這個測試顯示鼓組可能分散在多個分類中；需用耳聽與更多樣本驗證，不能只靠 RMS 判定品質。

同一樣本中，`guitar.wav` 和 `acoustic-guitar.wav` 的波形相似度約 0.9986；`keys.wav` 和 `piano.wav` 約 0.9994。這顯示部分模型類別可能高度重疊。因此不能把相近類別不加判斷地相加，否則可能重複同一段聲音。

### 七軌目標尚未完成

53-stem 模型不會直接輸出獨立的主奏吉他和節奏吉他。這兩個聲部需使用具備合適標籤的訓練資料再做專門分離或微調。目前結果僅供原型驗證，不應標示為已完成的七軌模型。

## 不提交的內容

- 預訓練模型權重（大型 `.ckpt`）
- 使用者測試音訊與所有輸出 WAV
- 由推論生成的 ZIP 檔
