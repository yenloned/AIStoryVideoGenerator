"""
圖片生成測試工具
允許你輸入自定義提示詞來測試圖片生成
"""

import os
import sys
import argparse
from pathlib import Path
from scripts.generate_images import ImageGenerator


def test_custom_prompt(
    prompt: str,
    negative_prompt: str = None,
    style: str = "cinematic",
    width: int = 768,
    height: int = 1344,
    output_dir: str = "test_images",
    num_steps: int = 30,
    guidance_scale: float = 9.0
):
    """
    使用自定義提示詞生成圖片
    
    Args:
        prompt: 主提示詞（描述你想要生成的圖片）
        negative_prompt: 負面提示詞（描述不想要的元素，可選）
        style: 風格（cinematic, chinese_ink, ancient, fantasy, horror, hand_drawn）
        width: 圖片寬度
        height: 圖片高度
        output_dir: 輸出目錄
        num_steps: 推理步數（越多質量越好，但越慢）
        guidance_scale: 引導強度（越高越嚴格遵循提示詞，建議 7-12）
    """
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("圖片生成測試工具")
    print("=" * 60)
    print(f"\n📝 主提示詞:")
    print(f"   {prompt}")
    print(f"\n🚫 負面提示詞:")
    if negative_prompt:
        print(f"   {negative_prompt}")
    else:
        print("   (使用默認負面提示詞)")
    print(f"\n🎨 風格: {style}")
    print(f"📐 尺寸: {width}x{height}")
    print(f"⚙️  推理步數: {num_steps}")
    print(f"📊 引導強度: {guidance_scale}")
    print("\n" + "=" * 60)
    
    # 初始化圖片生成器
    generator = ImageGenerator(
        model_type="sd15",
        output_dir=output_dir
    )
    
    # 載入模型
    print("\n📦 正在載入模型...")
    generator.load_model()
    
    # 生成圖片
    print("\n🎨 開始生成圖片...")
    try:
        import torch
        
        # 確保模型已載入
        if generator.pipeline is None:
            generator.load_model()
            
        # 自動翻譯提示詞
        if any('\u4e00' <= char <= '\u9fff' for char in prompt):
            print(f"\n🔤 檢測到中文，正在翻譯...")
            prompt = generator.translate_to_english(prompt)
            print(f"   翻譯結果: {prompt}")
        
        # 清除快取
        if generator.device == "cuda":
            torch.cuda.empty_cache()
        
        # 設置生成器
        gen = None
        if generator.device == "cuda":
            gen = torch.Generator(device="cuda")
            gen.manual_seed(42)
        
        # 默認負面提示詞
        if negative_prompt is None:
            negative_prompt = (
                "blurry, low quality, distorted, watermark, text overlay, ugly, bad anatomy, "
                "deformed, disfigured, poorly drawn, bad proportions, extra limbs, duplicate, "
                "cropped, out of frame, worst quality, low quality, jpeg artifacts, signature, "
                "username, error"
            )
        
        # 生成圖片
        print(f"\n⏳ 正在生成（這可能需要 1-3 分鐘）...")
        print(f"📝 完整提示詞: {prompt}")
        print(f"🚫 負面提示詞: {negative_prompt[:100]}...")
        
        image = generator.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=gen,
        ).images[0]
        
        # 保存圖片
        import time
        timestamp = int(time.time())
        output_path = os.path.join(output_dir, f"test_{timestamp}.png")
        image.save(output_path)
        
        print(f"\n✅ 圖片生成成功！")
        print(f"📁 保存位置: {os.path.abspath(output_path)}")
        
        # 清除快取
        if generator.device == "cuda":
            torch.cuda.empty_cache()
        
        return output_path
        
    except Exception as e:
        print(f"\n❌ 生成失敗: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="圖片生成測試工具 - 使用自定義提示詞生成圖片"
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="主提示詞（描述你想要生成的圖片）"
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="負面提示詞（描述不想要的元素）"
    )
    parser.add_argument(
        "--style",
        type=str,
        default="cinematic",
        choices=["cinematic", "chinese_ink", "ancient", "fantasy", "horror", "hand_drawn"],
        help="圖片風格"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="圖片寬度（默認 768）"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1344,
        help="圖片高度（默認 1344，9:16 比例）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_images",
        help="輸出目錄（默認 test_images）"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="推理步數（默認 30，越多質量越好但越慢）"
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=9.0,
        help="引導強度（默認 9.0，越高越嚴格遵循提示詞，建議 7-12）"
    )
    
    args = parser.parse_args()
    
    # 如果提示詞包含中文，給出建議
    if any('\u4e00' <= char <= '\u9fff' for char in args.prompt):
        print("\n💡 提示：你使用了中文提示詞")
        print("   建議：Stable Diffusion 對英文理解更好，可以嘗試使用英文提示詞")
        print("   例如：'an old man in a traditional Chinese room with wall paintings and bronze wine cups'\n")
    
    test_custom_prompt(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        style=args.style,
        width=args.width,
        height=args.height,
        output_dir=args.output_dir,
        num_steps=args.steps,
        guidance_scale=args.guidance
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 60)
        print("圖片生成測試工具")
        print("=" * 60)
        print("\n用法示例：")
        print("\n1. 基本用法（中文提示詞）：")
        print('   python test_image_generation.py "一位古代中國老翁坐在傳統木屋內，牆上掛著精美的壁畫"')
        print("\n2. 使用英文提示詞（推薦）：")
        print('   python test_image_generation.py "an old Chinese man sitting in a traditional wooden room with beautiful wall paintings, bronze wine cups on the table, sunset light through window"')
        print("\n3. 自定義參數：")
        print('   python test_image_generation.py "your prompt" --steps 40 --guidance 10 --style ancient')
        print("\n4. 指定負面提示詞：")
        print('   python test_image_generation.py "your prompt" --negative-prompt "modern, Western style, abstract"')
        print("\n參數說明：")
        print("  --style: cinematic, chinese_ink, ancient, fantasy, horror, hand_drawn")
        print("  --steps: 推理步數（20-50，默認 30）")
        print("  --guidance: 引導強度（7-12，默認 9.0）")
        print("  --width, --height: 圖片尺寸（默認 768x1344）")
        print("\n" + "=" * 60)
        sys.exit(0)
    
    main()

