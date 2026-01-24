# Fine-Tuning Guide for Image Generation

## 關於 Fine-Tuning

Fine-tuning Stable Diffusion 模型是可能的，但需要大量資源和時間。以下是幾種方法：

## 方法 1: LoRA (Low-Rank Adaptation) - 推薦 ⭐

**優點：**
- 只需要少量圖片（10-50張）
- 訓練速度快（幾小時）
- 模型文件小（幾MB到幾十MB）
- 可以疊加多個 LoRA

**缺點：**
- 需要準備訓練數據集
- 需要一些技術知識

**步驟：**
1. 準備 10-50 張高質量圖片，與你的故事主題相關
2. 使用工具如 [Kohya_ss](https://github.com/bmaltais/kohya_ss) 或 [EveryDream2](https://github.com/victorchall/EveryDream2trainer)
3. 訓練 LoRA（通常需要 4-8 小時）
4. 在代碼中加載 LoRA

**示例代碼（加載 LoRA）：**
```python
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipeline.load_lora_weights("./lora_weights/chinese_story.safetensors")
```

## 方法 2: 使用更好的中文模型

**推薦模型：**
- **Taiyi-Stable-Diffusion-1B-Chinese-v0.1** - 專門為中文內容訓練
- **IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1** - 中英文混合

**優點：**
- 已經針對中文內容優化
- 無需訓練
- 直接使用

**修改代碼使用中文模型：**
```python
# 在 generate_images.py 中修改
model_id = "IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1"
```

## 方法 3: 使用 ControlNet（精確控制）

**優點：**
- 可以精確控制圖片構圖
- 不需要訓練模型
- 可以基於草圖、深度圖等生成

**缺點：**
- 需要額外的控制圖像
- 稍微複雜一些

## 方法 4: 完整 Fine-Tuning（不推薦）

**缺點：**
- 需要數百到數千張圖片
- 訓練時間長（數天到數週）
- 需要大量 VRAM（至少 24GB）
- 模型文件大（幾GB）

## 推薦方案

**短期解決方案（已實現）：**
1. ✅ 改進提示詞（已完成）
2. ✅ 添加故事上下文到圖片生成（已完成）
3. ✅ 使用更詳細的場景描述（已完成）

**中期解決方案：**
1. 切換到中文專用模型（Taiyi）
2. 使用 LoRA 微調（如果有多個相關故事）

**長期解決方案：**
1. 收集大量高質量故事圖片
2. 訓練專用 LoRA
3. 整合到系統中

## 快速測試：使用中文模型

如果你想立即嘗試更好的中文支持，可以修改 `scripts/generate_images.py`：

```python
# 將第 72 行改為：
model_id = "IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1"
```

這個模型對中文提示詞的理解更好。

## 資源

- [Kohya_ss LoRA 訓練工具](https://github.com/bmaltais/kohya_ss)
- [Taiyi 中文模型](https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1)
- [Stable Diffusion Fine-Tuning 教程](https://huggingface.co/docs/diffusers/training/overview)


