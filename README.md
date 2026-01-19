# AI Story/Knowledge Video Generator（本地版本）

一套完全本地運行的 AI 影片自動化生成系統，不使用 OpenAI 或任何香港受限 API。

## 🎯 功能特點

- ✅ **完全本地運行** - 不依賴外部 API
- ✅ **自動生成劇本** - 使用 Ollama + Qwen 2.5 7B
- ✅ **本地圖片生成** - Stable Diffusion (SD 1.5 / SDXL)
- ✅ **本地語音合成** - Coqui TTS / Piper TTS
- ✅ **自動字幕** - 內建字幕生成
- ✅ **影片合成** - FFmpeg 自動剪輯
- ✅ **一鍵執行** - Python 腳本自動化整個流程

## 📋 系統需求

### 必需軟件

1. **Python 3.8+**
2. **Ollama** - [下載安裝](https://ollama.ai/)
3. **FFmpeg** - [下載安裝](https://ffmpeg.org/download.html)
4. **CUDA** (可選，但強烈建議用於 GPU 加速)

### 硬體建議

- **GPU**: NVIDIA GPU with 6GB+ VRAM (推薦 8GB+)
- **RAM**: 16GB+ (32GB 推薦)
- **存儲**: 至少 20GB 可用空間（用於模型下載）

## 🚀 安裝步驟

### 1. 克隆或下載項目

```bash
cd AIStoryFarm
```

### 2. 安裝 Python 依賴

```bash
pip install -r requirements.txt
```

### 3. 安裝並配置 Ollama

#### Windows:
1. 下載 [Ollama Windows 版本](https://ollama.ai/download/windows)
2. 安裝後，在命令行運行：

```bash
ollama pull qwen2.5:7b
```

#### Linux/Mac:
```bash
curl https://ollama.ai/install.sh | sh
ollama pull qwen2.5:7b
```

### 4. 安裝 FFmpeg

#### Windows:
1. 下載 [FFmpeg Windows 版本](https://www.gyan.dev/ffmpeg/builds/)
2. 解壓並將 `bin` 目錄添加到系統 PATH

#### Linux:
```bash
sudo apt-get install ffmpeg
```

#### Mac:
```bash
brew install ffmpeg
```

### 5. 配置 TTS（選擇一種）

#### 選項 A: Coqui TTS（推薦，自動安裝）

Coqui TTS 會通過 `pip install TTS` 自動安裝。首次運行時會自動下載中文模型。

#### 選項 B: Piper TTS

1. 下載 [Piper TTS](https://github.com/rhasspy/piper/releases)
2. 下載中文模型：
   ```bash
   # 創建模型目錄
   mkdir -p models/piper/zh_CN
   
   # 下載中文模型（從 Piper 官方）
   # 將模型文件放在 models/piper/zh_CN/ 目錄
   ```

### 6. 下載 Stable Diffusion 模型（首次運行時自動下載）

首次運行時，程序會自動下載模型：
- **SD 1.5**: ~4GB（較輕量，推薦）
- **SDXL**: ~7GB（高質量，需要更多 VRAM）

## 📖 使用方法

### 基本用法

```bash
python main.py "成語故事"
```

### 進階選項

```bash
# 指定圖片風格
python main.py "歷史典故" --style chinese_ink

# 指定 TTS 引擎
python main.py "冷知識" --tts piper

# 使用 SDXL 模型（需要更多 VRAM）
python main.py "都市傳說" --image-model sdxl

# 自定義輸出文件名
python main.py "成語故事" --output my_story
```

### 批次生成

一次生成多個影片：

```bash
# 使用預定義列表
python batch_generate.py

# 自定義關鍵字列表
python batch_generate.py --keywords "成語故事：守株待兔" "歷史典故：三顧茅廬" "冷知識：為什麼天空是藍色的"

# 指定統一樣式
python batch_generate.py --keywords "關鍵字1" "關鍵字2" --style cinematic
```

### 參數說明

- `keyword`: 題材關鍵字（必需）
- `--style`: 圖片風格
  - `cinematic` (默認) - 電影風格
  - `chinese_ink` - 中國水墨
  - `ancient` - 古代場景
  - `fantasy` - 奇幻風格
  - `horror` - 恐怖風格
  - `hand_drawn` - 手繪風格
- `--tts`: TTS 引擎 (`coqui` 或 `piper`)
- `--image-model`: 圖片模型 (`sd15` 或 `sdxl`)
- `--output`: 輸出文件名（不含擴展名）

## 📁 項目結構

```
AIStoryFarm/
├── main.py                 # 主程序入口
├── requirements.txt        # Python 依賴
├── README.md              # 本文件
├── scripts/                # 各功能模組
│   ├── generate_script.py  # 劇本生成
│   ├── generate_images.py  # 圖片生成
│   ├── generate_audio.py   # 語音生成
│   └── generate_video.py   # 影片生成
├── models/                 # 模型文件（自動下載）
├── output/                 # 輸出目錄
│   └── {keyword}/
│       ├── script/         # 生成的劇本
│       ├── images/         # 生成的圖片
│       ├── audio/          # 生成的音頻
│       └── video/          # 最終影片
├── images/                 # 臨時圖片（可選）
├── audio/                  # 臨時音頻（可選）
└── video/                  # 臨時影片（可選）
```

## 🔧 故障排除

### 問題 1: Ollama 連接失敗

**錯誤**: `無法連接到 Ollama`

**解決方案**:
1. 確認 Ollama 正在運行：
   ```bash
   ollama list
   ```
2. 確認模型已下載：
   ```bash
   ollama pull qwen2.5:7b
   ```
3. 檢查 Ollama 服務是否在 `http://localhost:11434` 運行

### 問題 2: FFmpeg 未找到

**錯誤**: `FFmpeg 不可用`

**解決方案**:
1. 確認 FFmpeg 已安裝：
   ```bash
   ffmpeg -version
   ```
2. 確認 FFmpeg 在系統 PATH 中

### 問題 3: GPU 記憶體不足

**錯誤**: `CUDA out of memory`

**解決方案**:
1. 使用較輕量的模型：
   ```bash
   python main.py "關鍵字" --image-model sd15
   ```
2. 減少批次大小（修改 `generate_images.py`）
3. 使用 CPU 模式（較慢）

### 問題 4: TTS 生成失敗

**錯誤**: `Coqui TTS 不可用` 或 `Piper TTS 不可用`

**解決方案**:
1. **Coqui TTS**: 確認已安裝：
   ```bash
   pip install TTS
   ```
2. **Piper TTS**: 確認已安裝並配置模型路徑

### 問題 5: 模型下載緩慢

**解決方案**:
1. 使用國內鏡像（如果可用）
2. 手動下載模型到 `~/.cache/huggingface/` 目錄
3. 使用 VPN 或代理

## 🎨 自定義配置

### 修改劇本風格

編輯 `scripts/generate_script.py` 中的 prompt 模板。

### 修改圖片風格

編輯 `scripts/generate_images.py` 中的 `style_prompts` 字典。

### 修改影片效果

編輯 `scripts/generate_video.py` 中的效果參數。

## 📝 輸出說明

生成的影片將保存在：
```
output/{keyword}/video/{keyword}_with_subtitles.mp4
```

影片規格：
- 解析度: 1080x1920 (Shorts 格式)
- 幀率: 30 FPS
- 格式: MP4 (H.264 + AAC)

## 🔄 工作流程

1. **輸入關鍵字** → 用戶提供題材
2. **生成劇本** → Ollama + Qwen 生成故事段落
3. **生成圖片** → Stable Diffusion 為每段生成背景圖
4. **生成語音** → TTS 合成語音
5. **合成影片** → FFmpeg 組合所有元素
6. **輸出影片** → 最終 MP4 文件

## 💡 使用建議

1. **首次運行**: 建議使用 `--image-model sd15`（較輕量）
2. **GPU 加速**: 確保 CUDA 正確安裝以獲得最佳性能
3. **批量生成**: 可以編寫腳本循環調用 `main.py` 進行批量生成
4. **自定義**: 根據需要修改各模組的參數和提示詞

## 📄 許可證

本項目僅供學習和研究使用。

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📧 聯繫

如有問題，請在 GitHub 上提交 Issue。

---

**注意**: 本系統完全本地運行，不依賴任何外部 API，適合香港地區使用。

