"""
圖片生成模組 - 使用本地 Stable Diffusion
為每個段落生成對應的背景圖片
"""

import os
import sys
from typing import List, Dict
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import json

# 設置 CUDA 記憶體管理環境變數（減少記憶體碎片）
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')


class ImageGenerator:
    def __init__(
        self,
        model_type: str = "sdxl",  # "sdxl" or "sd15"
        device: str = None,
        output_dir: str = "images"
    ):
        """
        初始化圖片生成器
        
        Args:
            model_type: 模型類型 "sdxl" 或 "sd15"
            device: 設備 ("cuda", "cpu", "mps")
            output_dir: 輸出目錄
        """
        self.model_type = model_type
        self.output_dir = output_dir
        self.pipeline = None
        
        # 設備檢測和診斷
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"✅ 檢測到 GPU: {gpu_name} ({gpu_memory:.1f} GB VRAM)")
            else:
                self.device = "cpu"
                print("⚠️  警告: 未檢測到 CUDA，將使用 CPU（速度極慢）")
                print("   提示: 請安裝 PyTorch CUDA 版本以使用 GPU 加速")
                print("   安裝命令: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        
        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🖼️  圖片生成器初始化，設備: {self.device}")
    
    def load_model(self):
        """載入 Stable Diffusion 模型"""
        try:
            print(f"📦 正在載入 {self.model_type} 模型...")
            
            if self.model_type == "sdxl":
                # SDXL 模型（需要更多 VRAM）
                model_id = "stabilityai/stable-diffusion-xl-base-1.0"
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    use_safetensors=True
                )
            else:
                # SD 1.5 模型（較輕量）
                # 使用原始模型（支援 safetensors，兼容所有 PyTorch 版本）
                # 已優化提示詞以更好地處理中文內容
                model_id = "runwayml/stable-diffusion-v1-5"
                print(f"📦 使用模型: {model_id}")
                print(f"💡 提示：已優化中文提示詞以獲得更好的中國傳統場景效果")
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    use_safetensors=True
                )
            
            # 優化記憶體使用（針對 RTX 3070 等 8GB VRAM GPU）
            if self.device == "cuda":
                # 使用 sequential CPU offload（最節省 VRAM 的方法）
                try:
                    # 使用 enable_sequential_cpu_offload 而不是 enable_model_cpu_offload
                    # 這會將模型的不同部分按順序移到 CPU，只在需要時移到 GPU
                    self.pipeline.enable_sequential_cpu_offload()
                    # 同時啟用 attention slicing 以進一步節省記憶體
                    self.pipeline.enable_attention_slicing(1)  # 切片大小 1（最節省記憶體）
                    print("💾 已啟用 sequential CPU offload + attention slicing（節省 VRAM）")
                except Exception as e:
                    print(f"⚠️  Sequential CPU offload 失敗，使用標準模式: {e}")
                    # 回退到標準優化
                    self.pipeline = self.pipeline.to(self.device)
                    self.pipeline.enable_attention_slicing(1)  # 使用切片大小 1（最節省記憶體）
                    # 使用 VAE tiling 而不是 slicing（更節省記憶體）
                    if hasattr(self.pipeline, 'vae'):
                        if hasattr(self.pipeline.vae, 'enable_tiling'):
                            self.pipeline.vae.enable_tiling()
                            print("💾 已啟用 VAE tiling")
            else:
                self.pipeline = self.pipeline.to(self.device)
            
            print("✅ 模型載入完成")
            
        except Exception as e:
            print(f"❌ 模型載入失敗: {e}")
            raise
    
    def generate_image(
        self,
        scene_description: str,
        style: str = "cinematic",
        output_path: str = None,
        width: int = None,
        height: int = None,
        story_title: str = None,
        story_text: str = None
    ) -> str:
        """
        生成單張圖片
        
        Args:
            scene_description: 場景描述
            style: 風格選項 (cinematic, chinese_ink, ancient, fantasy, horror, hand_drawn)
            output_path: 輸出路徑
            height: 圖片高度（Shorts 格式）
            width: 圖片寬度
            
        Returns:
            生成的圖片路徑
        """
        if self.pipeline is None:
            self.load_model()
        
        # 根據 GPU VRAM 自動調整解析度（8GB VRAM 使用較小解析度）
        if width is None or height is None:
            if self.device == "cuda" and torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if gpu_memory < 10:  # 8GB VRAM
                    # 使用較小的解析度以節省 VRAM
                    width = 768
                    height = 1344  # 保持 9:16 比例（Shorts 格式）
                    print(f"💾 檢測到 {gpu_memory:.1f}GB VRAM，使用較小解析度: {width}x{height}")
                else:
                    width = 1024
                    height = 1920
            else:
                width = 1024
                height = 1920
        
        # 風格提示詞（增強版，更適合中文故事）
        style_prompts = {
            "cinematic": "cinematic lighting, dramatic shadows, 4k, highly detailed, professional illustration, vibrant colors, sharp focus",
            "chinese_ink": "Chinese ink painting style, traditional Chinese art, elegant brush strokes, monochrome, classical Chinese aesthetics, traditional Chinese scene",
            "ancient": "ancient Chinese scene, historical setting, traditional Chinese architecture, period costume, Chinese historical drama style, authentic Chinese culture",
            "fantasy": "fantasy art style, magical atmosphere, vibrant colors, ethereal, Chinese fantasy elements, mystical",
            "horror": "dark atmosphere, eerie lighting, gothic style, mysterious shadows, Chinese horror aesthetic",
            "hand_drawn": "hand-drawn illustration, sketch style, artistic drawing, detailed linework, Chinese illustration style"
        }
        
        style_prompt = style_prompts.get(style, style_prompts["cinematic"])
        
        # 增強場景描述，結合故事上下文
        enhanced_scene = scene_description
        
        # 如果有故事標題和文本，添加到提示詞中以提供更多上下文
        context_parts = []
        if story_title:
            context_parts.append(f"Story: {story_title}")
        if story_text:
            # 提取關鍵信息（前50字）
            key_info = story_text[:50] if len(story_text) > 50 else story_text
            context_parts.append(f"Context: {key_info}")
        
        # 構建增強場景描述
        if any('\u4e00' <= char <= '\u9fff' for char in scene_description):
            # 包含中文字符，確保是中國傳統場景
            scene_with_context = scene_description
            if context_parts:
                scene_with_context = f"{scene_description}. Story context: {' '.join(context_parts)}"
            enhanced_scene = f"{scene_with_context}, ancient Chinese setting, traditional Chinese culture, historical Chinese scene, authentic Chinese period drama style"
        else:
            # 英文場景描述，也添加中國文化上下文
            if context_parts:
                enhanced_scene = f"{scene_description}. {' '.join(context_parts)}. Ancient Chinese setting, traditional Chinese culture"
            else:
                enhanced_scene = f"{scene_description}, ancient Chinese setting, traditional Chinese culture"
        
        # 組合完整提示詞（更詳細的提示，強調視覺細節）
        full_prompt = f"{enhanced_scene}, {style_prompt}, highly detailed, vivid colors, clear composition, accurate historical details, high quality, professional illustration, masterpiece, 4k"
        negative_prompt = "blurry, low quality, distorted, watermark, text, ugly, bad anatomy, deformed, disfigured, poorly drawn, bad proportions, extra limbs, duplicate, cropped, out of frame, worst quality, low quality, jpeg artifacts, signature, username, error, Western style, modern setting, unrelated to Chinese culture"
        
        try:
            print(f"🎨 正在生成圖片: {scene_description[:30]}...")
            
            # 清除 CUDA 快取（釋放記憶體）
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # 優化生成參數（減少步數以節省記憶體和時間）
            num_steps = 20 if self.device == "cuda" else 15  # 減少步數以節省 VRAM
            
            # 設置生成器
            generator = None
            if self.device == "cuda":
                generator = torch.Generator(device="cuda")
                generator.manual_seed(42)
            
            # 生成圖片
            image = self.pipeline(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_steps,
                guidance_scale=7.5,
                generator=generator,
            ).images[0]
            
            # 生成後清除快取
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # 保存圖片
            if output_path is None:
                import time
                timestamp = int(time.time())
                output_path = os.path.join(self.output_dir, f"image_{timestamp}.png")
            
            image.save(output_path)
            print(f"✅ 圖片已保存: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ 圖片生成失敗: {e}")
            raise
    
    def generate_images_for_script(self, script_data: Dict, style: str = "cinematic") -> List[str]:
        """
        為整個劇本生成所有圖片
        
        Args:
            script_data: 劇本數據（包含 paragraphs 和 title）
            style: 圖片風格
            
        Returns:
            圖片路徑列表
        """
        paragraphs = script_data.get("paragraphs", [])
        story_title = script_data.get("title", "")
        image_paths = []
        
        print(f"🖼️  開始為 {len(paragraphs)} 個段落生成圖片...")
        print(f"📖 故事標題: {story_title}")
        
        for i, paragraph in enumerate(paragraphs):
            scene = paragraph.get("scene", paragraph.get("text", ""))
            text = paragraph.get("text", "")
            output_path = os.path.join(self.output_dir, f"scene_{i+1:02d}.png")
            
            try:
                # 在每次生成前清除快取
                if self.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 傳遞故事標題和文本以提供上下文
                img_path = self.generate_image(
                    scene_description=scene,
                    style=style,
                    output_path=output_path,
                    story_title=story_title,
                    story_text=text
                )
                image_paths.append(img_path)
                
                # 生成後清除快取
                if self.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except torch.cuda.OutOfMemoryError as e:
                print(f"⚠️  段落 {i+1} 圖片生成失敗: VRAM 不足")
                print(f"   嘗試清理記憶體並重試...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                # 跳過這張圖片
                continue
            except Exception as e:
                print(f"⚠️  段落 {i+1} 圖片生成失敗: {e}")
                # 清除快取後繼續
                if self.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        
        print(f"✅ 共生成 {len(image_paths)} 張圖片")
        return image_paths


def main():
    """測試用主函數"""
    if len(sys.argv) < 2:
        print("用法: python generate_images.py <script.json> [style]")
        sys.exit(1)
    
    script_file = sys.argv[1]
    style = sys.argv[2] if len(sys.argv) > 2 else "cinematic"
    
    with open(script_file, "r", encoding="utf-8") as f:
        script_data = json.load(f)
    
    generator = ImageGenerator()
    
    try:
        image_paths = generator.generate_images_for_script(script_data, style)
        print(f"\n生成的圖片: {image_paths}")
    except Exception as e:
        print(f"錯誤: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()





