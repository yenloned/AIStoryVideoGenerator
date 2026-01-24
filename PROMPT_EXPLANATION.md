# 圖片生成提示詞系統說明

## 如何告訴模型生成圖片

我們使用 **Stable Diffusion** 模型生成圖片，它通過 **提示詞（Prompt）** 來理解你想要什麼圖片。

## 提示詞的組成部分

### 1. 主提示詞（Positive Prompt）

這是告訴模型**要生成什麼**的提示詞，由以下部分組成：

```
[場景描述] + [強制要求] + [風格提示] + [質量要求]
```

#### 示例（實際生成的提示詞）：

```
輸入場景描述: "屋內壁畫與酒杯"

生成的完整提示詞:
"屋內壁畫與酒杯, must show exactly what is described in the scene, traditional Chinese setting, period-appropriate details, accurate representation, cinematic composition, dramatic lighting, historical Chinese setting, traditional Chinese architecture visible, period-appropriate costumes, authentic Chinese cultural elements, detailed background, professional story illustration, highly detailed scene matching the description exactly, accurate visual representation, clear and readable composition, professional story illustration, 4k quality"
```

### 2. 負面提示詞（Negative Prompt）

這是告訴模型**不要生成什麼**的提示詞，用來排除不想要的元素：

```
"blurry, low quality, distorted, watermark, text overlay, ugly, bad anatomy, 
deformed, disfigured, poorly drawn, bad proportions, extra limbs, duplicate, 
cropped, out of frame, worst quality, low quality, jpeg artifacts, signature, 
username, error, Western style, modern setting, modern clothing, modern architecture, 
unrelated to Chinese culture, incorrect period details, anachronistic elements, 
unrelated objects, random elements not in description, scene does not match description, 
incorrect setting, wrong time period, non-Chinese elements, abstract art, 
unclear scene, confusing composition, elements not matching story context"
```

## 提示詞構建流程

### 步驟 1: 獲取場景描述
- 從劇本的 `scene` 字段獲取（例如："屋內壁畫與酒杯"）

### 步驟 2: 添加故事上下文
- 如果有故事標題，添加：`story theme: {標題}`
- 如果有故事文本，添加：`story context: {文本前80字}`

### 步驟 3: 選擇風格
根據 `--style` 參數選擇風格提示詞：
- `cinematic`: 電影風格
- `chinese_ink`: 中國水墨畫風格
- `ancient`: 古代場景風格
- 等等...

### 步驟 4: 組合完整提示詞
將所有部分組合：
```
{場景描述} + {強制要求} + {風格} + {質量要求}
```

### 步驟 5: 生成圖片
使用以下參數：
- `prompt`: 完整提示詞
- `negative_prompt`: 負面提示詞
- `guidance_scale`: 9.0（越高越嚴格遵循提示詞）
- `num_inference_steps`: 30（步數越多質量越好）

## 為什麼圖片可能不符合描述？

### 問題 1: 場景描述太簡短或模糊
**示例：**
- ❌ 不好："屋內壁畫與酒杯"（太簡短，缺少細節）
- ✅ 更好："一位古代中國老翁坐在傳統木屋內，牆上掛著精美的壁畫，桌上擺放著青銅酒杯，夕陽從窗戶照進來"

### 問題 2: 模型不理解中文
- Stable Diffusion 主要訓練於英文數據
- 中文描述可能被誤解
- **解決方案**: 使用英文描述，或添加英文翻譯

### 問題 3: 提示詞衝突
- 如果提示詞中有矛盾的元素，模型可能混淆
- **解決方案**: 確保提示詞一致

### 問題 4: Guidance Scale 不夠高
- 當前設置：9.0（已經較高）
- 可以嘗試提高到 10-12（但可能過度飽和）

## 改進建議

### 1. 改進場景描述生成
在 `scripts/generate_script.py` 中，讓 AI 生成更詳細的場景描述：
- 包含人物動作
- 包含環境細節
- 包含視覺元素（顏色、光線等）

### 2. 添加英文翻譯
將中文場景描述翻譯成英文，因為模型對英文理解更好：
```python
# 偽代碼示例
chinese_scene = "屋內壁畫與酒杯"
english_scene = translate_to_english(chinese_scene)
# "an old man in a traditional Chinese room with wall paintings and bronze wine cups"
```

### 3. 使用更詳細的提示詞模板
```python
template = """
A detailed scene showing: {scene_description}
Setting: Traditional Chinese {period} period
Characters: {characters}
Objects: {objects}
Lighting: {lighting}
Composition: {composition}
Style: {style}
"""
```

## 調試提示詞

運行時會看到：
```
📝 提示詞預覽: 屋內壁畫與酒杯, must show exactly what is described...
```

**檢查點：**
1. 提示詞是否包含所有必要元素？
2. 是否有衝突的描述？
3. 是否太簡短？
4. 是否需要添加更多細節？

## 測試自己的提示詞

使用 `test_image_generation.py` 來測試不同的提示詞，看看哪個效果最好！

