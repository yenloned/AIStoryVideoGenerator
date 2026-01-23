"""
主程式 - AI 故事/知識影片生成器
一鍵執行完整 pipeline
"""

import os
import sys
import json
import argparse
from pathlib import Path

# 導入各模組
from scripts.generate_script import ScriptGenerator
from scripts.generate_images import ImageGenerator
from scripts.generate_audio import AudioGenerator
from scripts.generate_video import VideoGenerator


class VideoPipeline:
    """完整的影片生成流程"""
    
    def __init__(
        self,
        keyword: str,
        style: str = "cinematic",
        tts_engine: str = "coqui",
        image_model: str = "sd15",
        output_name: str = None
    ):
        """
        初始化 pipeline
        
        Args:
            keyword: 題材關鍵字
            style: 圖片風格
            tts_engine: TTS 引擎
            image_model: 圖片模型類型
            output_name: 輸出文件名（不含擴展名）
        """
        self.keyword = keyword
        self.style = style
        self.tts_engine = tts_engine
        self.image_model = image_model
        self.output_name = output_name or keyword.replace(" ", "_")
        
        # 創建輸出目錄
        self.work_dir = Path("output") / self.output_name
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # 子目錄
        self.script_dir = self.work_dir / "script"
        self.images_dir = self.work_dir / "images"
        self.audio_dir = self.work_dir / "audio"
        self.video_dir = self.work_dir / "video"
        
        for d in [self.script_dir, self.images_dir, self.audio_dir, self.video_dir]:
            d.mkdir(exist_ok=True)
        
        print(f"🎯 初始化 Pipeline，關鍵字: {keyword}")
        print(f"📁 工作目錄: {self.work_dir}")
    
    def run(self):
        """執行完整流程"""
        try:
            # 步驟 1: 生成劇本
            print("\n" + "="*50)
            print("步驟 1/5: 生成劇本")
            print("="*50)
            script_data = self._generate_script()
            
            # 保存劇本
            script_file = self.script_dir / "script.json"
            with open(script_file, "w", encoding="utf-8") as f:
                json.dump(script_data, f, ensure_ascii=False, indent=2)
            print(f"💾 劇本已保存: {script_file}")
            
            # 步驟 2: 生成圖片
            print("\n" + "="*50)
            print("步驟 2/5: 生成圖片")
            print("="*50)
            image_paths = self._generate_images(script_data)
            
            # 步驟 3: 生成語音
            print("\n" + "="*50)
            print("步驟 3/5: 生成語音")
            print("="*50)
            audio_paths = self._generate_audio(script_data)
            
            # 步驟 4: 生成影片
            print("\n" + "="*50)
            print("步驟 4/5: 生成影片")
            print("="*50)
            video_path = self._generate_video(script_data, image_paths, audio_paths)
            
            # 步驟 5: 完成
            print("\n" + "="*50)
            print("步驟 5/5: 完成")
            print("="*50)
            print(f"✅ 影片生成完成！")
            print(f"📹 輸出文件: {video_path}")
            print(f"📁 所有文件保存在: {self.work_dir}")
            
            return video_path
            
        except Exception as e:
            print(f"\n❌ Pipeline 執行失敗: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def _generate_script(self) -> dict:
        """生成劇本"""
        generator = ScriptGenerator()
        return generator.generate_script(self.keyword)
    
    def _generate_images(self, script_data: dict) -> list:
        """生成圖片"""
        generator = ImageGenerator(
            model_type=self.image_model,
            output_dir=str(self.images_dir)
        )
        return generator.generate_images_for_script(script_data, style=self.style)
    
    def _generate_audio(self, script_data: dict) -> list:
        """生成語音"""
        generator = AudioGenerator(
            tts_engine=self.tts_engine,
            output_dir=str(self.audio_dir)
        )
        return generator.generate_audio_for_script(script_data)
    
    def _generate_video(self, script_data: dict, image_paths: list, audio_paths: list) -> str:
        """生成影片"""
        generator = VideoGenerator(output_dir=str(self.video_dir))
        output_path = str(self.video_dir / f"{self.output_name}.mp4")
        return generator.generate_video(
            script_data, image_paths, audio_paths,
            output_path=output_path, style="mixed"
        )


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description="AI 故事/知識影片生成器 - 本地運行版本"
    )
    parser.add_argument(
        "keyword",
        type=str,
        help="題材關鍵字（例如：成語故事、歷史典故）"
    )
    parser.add_argument(
        "--style",
        type=str,
        default="cinematic",
        choices=["cinematic", "chinese_ink", "ancient", "fantasy", "horror", "hand_drawn"],
        help="圖片風格"
    )
    parser.add_argument(
        "--tts",
        type=str,
        default="coqui",
        choices=["coqui", "piper"],
        help="TTS 引擎"
    )
    parser.add_argument(
        "--image-model",
        type=str,
        default="sd15",
        choices=["sd15", "sdxl"],
        help="圖片生成模型（sd15 較輕量，sdxl 較高質量但需要更多 VRAM）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="輸出文件名（不含擴展名）"
    )
    
    args = parser.parse_args()
    
    # 創建並運行 pipeline
    pipeline = VideoPipeline(
        keyword=args.keyword,
        style=args.style,
        tts_engine=args.tts,
        image_model=args.image_model,
        output_name=args.output
    )
    
    pipeline.run()


if __name__ == "__main__":
    main()





