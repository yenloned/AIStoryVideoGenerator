# 快速開始指南

## 🚀 5 分鐘快速開始

### 步驟 1: 安裝必需軟件

#### 安裝 Ollama
1. 訪問 https://ollama.ai/
2. 下載並安裝 Ollama
3. 打開命令行，運行：
   ```bash
   ollama pull qwen2.5:7b
   ```

#### 安裝 FFmpeg
- **Windows**: 下載 https://www.gyan.dev/ffmpeg/builds/ 並添加到 PATH
- **Linux**: `sudo apt-get install ffmpeg`
- **Mac**: `brew install ffmpeg`

### 步驟 2: 安裝 Python 依賴

```bash
pip install -r requirements.txt
```

### 步驟 3: 運行第一個影片

```bash
python main.py "成語故事"
```

就是這麼簡單！🎉

## 📝 完整範例

### 生成成語故事影片（中國水墨風格）

```bash
python main.py "守株待兔" --style chinese_ink
```

### 生成歷史典故影片（古代場景風格）

```bash
python main.py "三顧茅廬" --style ancient
```

### 生成冷知識影片（電影風格）

```bash
python main.py "為什麼天空是藍色的" --style cinematic
```

## ⚠️ 常見問題

### Q: 提示 "無法連接到 Ollama"
**A**: 確認 Ollama 正在運行：
```bash
ollama list
```

### Q: 提示 "FFmpeg 不可用"
**A**: 確認 FFmpeg 已安裝並在 PATH 中：
```bash
ffmpeg -version
```

### Q: GPU 記憶體不足
**A**: 使用較輕量的模型：
```bash
python main.py "關鍵字" --image-model sd15
```

### Q: 首次運行很慢
**A**: 正常！首次運行需要下載模型（約 4-7GB），之後會快很多。

## 🎯 下一步

- 查看 [README.md](README.md) 了解詳細配置
- 修改 `scripts/` 中的代碼自定義風格
- 批量生成多個影片

---

**提示**: 如果遇到問題，請查看 README.md 的「故障排除」部分。

