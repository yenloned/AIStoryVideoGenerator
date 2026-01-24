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

### 2.5. 安裝 PyTorch with CUDA（重要！用於 GPU 加速）

**⚠️ 重要**: 如果您的系統有 NVIDIA GPU，必須安裝 PyTorch CUDA 版本才能使用 GPU 加速。否則會使用 CPU，速度極慢（每張圖片可能需要 40+ 分鐘）。

#### 檢查當前 PyTorch 版本：
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

如果顯示 `CUDA available: False`，請安裝 CUDA 版本：

#### Windows (CUDA 12.1):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Windows (CUDA 11.8):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 驗證安裝：
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

應該顯示 `CUDA available: True` 和您的 GPU 名稱。

**注意**: 
- 確保您的 NVIDIA 驅動程序已更新到最新版本
- CUDA 12.1 需要 NVIDIA 驅動 525.60.13 或更高版本
- CUDA 11.8 需要 NVIDIA 驅動 450.80.02 或更高版本

### 3. 安裝並配置 Ollama

#### Windows:
1. 下載 [Ollama Windows 版本](https://ollama.ai/download/windows)
2. 安裝 Ollama（GUI 應用程序）
3. **確保 Ollama 服務正在運行**（檢查系統托盤是否有 Ollama 圖標）
4. 下載模型（選擇以下方法之一）：

**方法 A: 使用命令行（推薦）**
   
   如果 `ollama` 命令不可用，需要將 Ollama 添加到系統 PATH：
   
   a. 找到 Ollama 安裝目錄（通常在 `C:\Users\<用戶名>\AppData\Local\Programs\Ollama`）
   
   b. 將 `ollama.exe` 所在目錄添加到系統 PATH：
      - 按 `Win + R`，輸入 `sysdm.cpl`，按 Enter
      - 點擊「高級」標籤 → 「環境變數」
      - 在「系統變數」中找到 `Path`，點擊「編輯」
      - 點擊「新增」，添加 Ollama 安裝目錄（例如：`C:\Users\<用戶名>\AppData\Local\Programs\Ollama`）
      - 點擊「確定」保存
      - **重新開啟命令提示符**（重要！）
   
   c. 驗證安裝：
      ```bash
      ollama --version
      ```
   
   d. 下載模型：
      ```bash
      ollama pull qwen2.5:7b
      ```

**方法 B: 使用 Ollama GUI**
   
   1. 打開 Ollama GUI 應用程序
   2. 在界面中搜索並下載 `qwen2.5:7b` 模型
   3. 等待下載完成

**方法 C: 使用完整路徑（CMD 或 PowerShell）**
   
   **在 CMD 中**：
   ```cmd
   "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" pull qwen2.5:7b
   ```
   
   **在 PowerShell 中**：
   ```powershell
   & "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull qwen2.5:7b
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

Coqui TTS 會通過 `pip install TTS` 自動安裝。系統會自動嘗試使用最佳模型：
- **XTTS v2** (優先) - 最高質量，自然語音，多語言支持
- **Tacotron2** (備用) - 標準中文模型
- **FastSpeech2** (備用) - 快速生成

首次運行時會自動下載模型（XTTS v2 約 1.7GB，Tacotron2 約 500MB）。

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
- **SD 1.5 (DreamShaper)**: ~4GB（較輕量，推薦，平衡模型）
  - 使用 `Lykon/DreamShaper`
  - 適合多樣化主題：人物、動物、物體、場景
  - 對故事插圖有良好表現，已優化提示詞和風格限制
  - 備用模型：Realistic Vision V5.1 或原始 SD 1.5
- **SDXL**: ~7GB（高質量，需要更多 VRAM）

## 📖 使用方法

### 基本用法

```bash
python main.py "成語故事"
```

### 測試圖片生成（自定義提示詞）

如果你想測試不同的提示詞來生成圖片：

```bash
# 使用中文提示詞
python test_image_generation.py "一位古代中國老翁坐在傳統木屋內，牆上掛著精美的壁畫"

# 使用英文提示詞（推薦，模型理解更好）
python test_image_generation.py "an old Chinese man sitting in a traditional wooden room with beautiful wall paintings, bronze wine cups on the table, sunset light through window"

# 自定義參數
python test_image_generation.py "your prompt" --steps 40 --guidance 10 --style ancient

# 查看所有選項
python test_image_generation.py
```

**提示詞技巧：**
- 使用英文提示詞通常效果更好
- 描述要具體：包含人物、動作、環境、光線等
- 使用 `--guidance` 調整嚴格度（7-12，默認 9.0）
- 使用 `--steps` 調整質量（20-50，默認 30）

詳見 `PROMPT_EXPLANATION.md` 了解提示詞系統的工作原理。

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

### 問題 1: Ollama 命令未找到（Windows）

**錯誤**: `'ollama' is not recognized as an internal or external command`

**解決方案**:
1. **確認 Ollama 已安裝並運行**：
   - 檢查系統托盤是否有 Ollama 圖標
   - 如果沒有，從開始菜單啟動 Ollama

2. **將 Ollama 添加到 PATH**：
   - 找到 Ollama 安裝目錄：`C:\Users\<你的用戶名>\AppData\Local\Programs\Ollama`
   - 將此目錄添加到系統 PATH（見上方安裝步驟）
   - **重新開啟命令提示符**

3. **使用完整路徑**（臨時解決方案）：
   
   **在 CMD 中**：
   ```cmd
   "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" pull qwen2.5:7b
   ```
   
   **在 PowerShell 中**：
   ```powershell
   & "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull qwen2.5:7b
   ```

4. **使用 GUI 下載模型**：
   - 打開 Ollama GUI，在界面中直接下載模型

### 問題 2: Ollama 連接失敗

**錯誤**: `無法連接到 Ollama`

**解決方案**:
1. 確認 Ollama 正在運行（檢查系統托盤）
2. 確認模型已下載：
   ```bash
   ollama list
   ```
   或使用完整路徑：
   ```powershell
   & "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" list
   ```
3. 檢查 Ollama 服務是否在 `http://localhost:11434` 運行
4. 如果 Ollama 未運行，從開始菜單啟動 Ollama

### 問題 3: FFmpeg 未找到

**錯誤**: `FFmpeg 不可用`

**解決方案**:
1. 確認 FFmpeg 已安裝：
   ```bash
   ffmpeg -version
   ```
2. 確認 FFmpeg 在系統 PATH 中

### 問題 4: 使用 CPU 而不是 GPU（圖片生成極慢）

**症狀**: 圖片生成顯示 `設備: cpu`，每張圖片需要 40+ 分鐘

**解決方案**:
1. **檢查 PyTorch CUDA 支持**：
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```
   
2. **如果顯示 `False`，安裝 PyTorch CUDA 版本**：
   ```bash
   # CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # 或 CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   
3. **確認 NVIDIA 驅動已安裝並更新**
4. **重新運行程序**，應該會顯示 `設備: cuda`

### 問題 5: GPU 記憶體不足

**錯誤**: `CUDA out of memory`

**解決方案**:
1. 使用較輕量的模型：
   ```bash
   python main.py "關鍵字" --image-model sd15
   ```
2. 減少批次大小（修改 `generate_images.py`）
3. 關閉其他使用 GPU 的程序
4. 使用 CPU 模式（較慢，不推薦）

### 問題 6: TTS 生成失敗

**錯誤**: `Coqui TTS 不可用` 或 `Piper TTS 不可用`

**解決方案**:
1. **Coqui TTS**: 確認已安裝：
   ```bash
   pip install TTS
   ```
2. **Piper TTS**: 確認已安裝並配置模型路徑

### 問題 7: 生成的圖片與主題無關

**症狀**: 生成的圖片不符合中文故事內容

**解決方案**:
1. 檢查劇本中的 `scene` 描述是否準確
2. 嘗試不同的風格選項（`--style`）
3. 如果問題持續，可以手動編輯 `scripts/generate_images.py` 中的提示詞模板

### 問題 8: 模型下載緩慢

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

