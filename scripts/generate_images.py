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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.pipeline = None
        
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
                model_id = "runwayml/stable-diffusion-v1-5"
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    use_safetensors=True
                )
            
            self.pipeline = self.pipeline.to(self.device)
            
            # 優化記憶體使用
            if self.device == "cuda":
                self.pipeline.enable_attention_slicing()
                self.pipeline.enable_vae_slicing()
            
            print("✅ 模型載入完成")
            
        except Exception as e:
            print(f"❌ 模型載入失敗: {e}")
            raise
    
    def generate_image(
        self,
        scene_description: str,
        style: str = "cinematic",
        output_path: str = None,
        width: int = 1024,
        height: int = 1920
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
        
        # 風格提示詞
        style_prompts = {
            "cinematic": "cinematic lighting, dramatic shadows, 4k, detailed, illustration",
            "chinese_ink": "Chinese ink painting style, traditional Chinese art, elegant brush strokes, monochrome",
            "ancient": "ancient Chinese scene, historical setting, traditional architecture, period costume",
            "fantasy": "fantasy art style, magical atmosphere, vibrant colors, ethereal",
            "horror": "dark atmosphere, eerie lighting, gothic style, mysterious shadows",
            "hand_drawn": "hand-drawn illustration, sketch style, artistic drawing, detailed linework"
        }
        
        style_prompt = style_prompts.get(style, style_prompts["cinematic"])
        
        # 組合完整提示詞
        full_prompt = f"{scene_description}, {style_prompt}"
        negative_prompt = "blurry, low quality, distorted, watermark, text, ugly, bad anatomy"
        
        try:
            print(f"🎨 正在生成圖片: {scene_description[:30]}...")
            
            # 生成圖片
            image = self.pipeline(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=30,
                guidance_scale=7.5,
            ).images[0]
            
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
            script_data: 劇本數據（包含 paragraphs）
            style: 圖片風格
            
        Returns:
            圖片路徑列表
        """
        paragraphs = script_data.get("paragraphs", [])
        image_paths = []
        
        print(f"🖼️  開始為 {len(paragraphs)} 個段落生成圖片...")
        
        for i, paragraph in enumerate(paragraphs):
            scene = paragraph.get("scene", paragraph.get("text", ""))
            output_path = os.path.join(self.output_dir, f"scene_{i+1:02d}.png")
            
            try:
                img_path = self.generate_image(
                    scene_description=scene,
                    style=style,
                    output_path=output_path
                )
                image_paths.append(img_path)
            except Exception as e:
                print(f"⚠️  段落 {i+1} 圖片生成失敗: {e}")
                # 使用預設圖片或跳過
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

