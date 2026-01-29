"""
主程式 - AI 故事/知識影片生成器
執行時以選單輸入（故事來源、圖片模型等），無需指令參數
"""

import os
import sys
import json
import random
from pathlib import Path

# 導入各模組
from scripts.generate_script import ScriptGenerator, script_from_story_text
from scripts.generate_images import ImageGenerator
from scripts.generate_audio import AudioGenerator
from scripts.generate_video import VideoGenerator

# 專案根目錄
PROJECT_ROOT = Path(__file__).resolve().parent
TOPICS_FILE = PROJECT_ROOT / "data" / "topics.json"
MODELS_DIR = PROJECT_ROOT / "models"
DREAMSHAPER_PATH = MODELS_DIR / "dreamshaper_8.safetensors"
TTS_REFERENCE_WAV = PROJECT_ROOT / "data" / "tts_reference.wav"  # 可選，6–10 秒情感參考音（可任一種語言），只取音色與語調


class VideoPipeline:
    """完整的影片生成流程"""
    
    def __init__(
        self,
        keyword: str = None,
        style: str = "cinematic",
        tts_engine: str = "coqui",
        image_model: str = "sd15",
        output_name: str = None,
        lora_path: str = None,
        lora_scale: float = 0.8,
        checkpoint_path: str = None,
        script_path: str = None,
        story_file: str = None,
        story_text: str = None,
    ):
        """
        初始化 pipeline
        
        Args:
            keyword: 題材關鍵字（與 --script / --story-file / --story 二選一）
            style: 圖片風格
            tts_engine: TTS 引擎
            image_model: 圖片模型類型
            output_name: 輸出文件名（不含擴展名）
            lora_path: 可選 LoRA 權重路徑（見 FINE_TUNING_GUIDE.md）
            lora_scale: LoRA 強度 0~1
            checkpoint_path: 可選本地完整模型路徑（CivitAI 等，見 CIVITAI_IMPORT.md）
            script_path: 可選，直接使用此劇本 JSON 檔（跳過劇本生成）
            story_file: 可選，從此文字檔讀取故事（分段後當劇本）
            story_text: 可選，直接傳入故事文字（同 --story）
        """
        self.keyword = keyword
        self.script_path = script_path
        self.story_file = story_file
        self.story_text = story_text
        self.style = style
        self.tts_engine = tts_engine
        self.image_model = image_model
        self.output_name = output_name or (keyword.replace(" ", "_") if keyword else "my_story")
        self.lora_path = lora_path
        self.lora_scale = lora_scale
        self.checkpoint_path = checkpoint_path
        
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
        
        if keyword:
            print(f"🎯 初始化 Pipeline，關鍵字: {keyword}")
        elif script_path:
            print(f"🎯 初始化 Pipeline，使用劇本檔: {script_path}")
        elif story_file:
            print(f"🎯 初始化 Pipeline，使用故事檔: {story_file}")
        elif story_text:
            print(f"🎯 初始化 Pipeline，使用輸入故事文字")
        else:
            print(f"🎯 初始化 Pipeline，輸出: {self.output_name}")
        print(f"📁 工作目錄: {self.work_dir}")
    
    def run(self):
        """執行完整流程"""
        try:
            # 步驟 1: 劇本來源（生成 / 讀取 JSON / 從故事文字轉換）
            print("\n" + "="*50)
            print("步驟 1/5: 劇本")
            print("="*50)
            script_data = self._get_script_data()
            
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
    
    def _get_script_data(self) -> dict:
        """取得劇本：從檔案、故事文字或由關鍵字生成"""
        if self.script_path:
            with open(self.script_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"📂 已載入劇本: {self.script_path}")
            return data
        if self.story_file:
            with open(self.story_file, "r", encoding="utf-8") as f:
                text = f.read()
            data = script_from_story_text(text, title=Path(self.story_file).stem)
            print(f"📂 已從故事檔轉成劇本: {self.story_file}（{len(data.get('paragraphs', []))} 段）")
            return data
        if self.story_text:
            data = script_from_story_text(self.story_text)
            print(f"📂 已從輸入文字轉成劇本（{len(data.get('paragraphs', []))} 段）")
            return data
        print("📝 由關鍵字生成劇本...")
        generator = ScriptGenerator()
        return generator.generate_script(self.keyword)
    
    def _generate_images(self, script_data: dict) -> list:
        """生成圖片"""
        generator = ImageGenerator(
            model_type=self.image_model,
            output_dir=str(self.images_dir),
            lora_path=self.lora_path,
            lora_scale=self.lora_scale,
            checkpoint_path=self.checkpoint_path
        )
        # 如果 LLM 推薦了風格，使用 LLM 推薦的風格，否則使用用戶指定的風格
        llm_recommended_style = script_data.get("style", self.style)
        print(f"🎨 使用圖片風格: {llm_recommended_style} (LLM 推薦: {script_data.get('reason', 'N/A')})")
        return generator.generate_images_for_script(script_data, style=llm_recommended_style)
    
    def _generate_audio(self, script_data: dict) -> list:
        """生成語音（支援情感參考音與輸出正規化，更乾淨、有感情）"""
        reference_wav = str(TTS_REFERENCE_WAV) if TTS_REFERENCE_WAV.exists() else None
        generator = AudioGenerator(
            tts_engine=self.tts_engine,
            output_dir=str(self.audio_dir),
            reference_wav=reference_wav,
            clean_output=True,
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


def load_topics():
    """載入 data/topics.json，若無則回傳空結構"""
    if not TOPICS_FILE.exists():
        return {"categories": []}
    with open(TOPICS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def interactive_run():
    """以選單取得使用者輸入後執行 pipeline"""
    print("\n" + "=" * 50)
    print("  AI 故事 / 知識影片生成器")
    print("=" * 50)

    # --- 1. 故事來源 ---
    print("\n【1】故事來源")
    print("  1. 依主題（從固定主題隨機選一則故事）")
    print("  2. 輸入自己的故事（從檔案或貼上文字）")
    choice_source = input("請選擇 (1 或 2): ").strip() or "1"

    keyword = None
    script_path = None
    story_file = None
    story_text = None
    style = "cinematic"
    output_name = None

    if choice_source == "1":
        topics_data = load_topics()
        categories = topics_data.get("categories", [])
        if not categories:
            print("⚠️ 未找到主題資料（請確認 data/topics.json 存在）。改用預設關鍵字。")
            keyword = "成語故事：守株待兔"
            style = "chinese_ink"
        else:
            print("\n  可選主題：")
            for i, cat in enumerate(categories, 1):
                print(f"    {i}. {cat['name']}")
            idx = input(f"請選擇 (1-{len(categories)}): ").strip()
            try:
                idx = int(idx) if idx else 1
                cat = categories[max(0, min(idx - 1, len(categories) - 1))]
                keywords = cat.get("keywords", [])
                keyword = random.choice(keywords) if keywords else cat["name"]
                style = cat.get("style", "cinematic")
                print(f"  → 已選：{keyword}")
            except (ValueError, IndexError):
                keyword = categories[0]["keywords"][0] if categories[0].get("keywords") else categories[0]["name"]
                style = categories[0].get("style", "cinematic")
        output_name = output_name or (keyword.replace(" ", "_")[:40] if keyword else None)
    else:
        print("\n  2a. 從檔案讀取故事")
        print("  2b. 貼上故事文字（輸入完後空一行結束）")
        sub = input("請選擇 (a 或 b): ").strip().lower()
        if sub == "a":
            path = input("請輸入故事檔路徑: ").strip()
            if path and Path(path).exists():
                story_file = path
                output_name = Path(path).stem
            else:
                print("⚠️ 檔案不存在，改為貼上文字。")
                sub = "b"
        if sub == "b" or not story_file:
            print("請貼上故事文字，輸入完後按 Enter 空一行結束：")
            lines = []
            while True:
                line = input()
                if line == "" and lines:
                    break
                if line == "":
                    continue
                lines.append(line)
            story_text = "\n".join(lines) if lines else None
            if not story_text:
                print("❌ 未輸入文字，結束。")
                sys.exit(1)
            output_name = output_name or "my_story"

    # --- 2. 圖片模型 ---
    print("\n【2】圖片模型")
    print("  1. 預設模型（從網路載入，SD 1.5）")
    print("  2. 本地模型 DreamShaper（需將 dreamshaper_8.safetensors 放在 /models）")
    choice_model = input("請選擇 (1 或 2): ").strip() or "1"

    image_model = "sd15"
    checkpoint_path = None
    if choice_model == "2":
        if DREAMSHAPER_PATH.exists():
            checkpoint_path = str(DREAMSHAPER_PATH)
            print(f"  → 使用本地模型: {checkpoint_path}")
        else:
            print("  ⚠️ 未找到本地模型。")
            print(f"     請將 dreamshaper_8.safetensors 放在：{MODELS_DIR}")
            print("  → 改為使用預設模型。")
    else:
        print("  → 使用預設模型（SD 1.5）。")

    # --- 3. 輸出名稱（可選） ---
    if not output_name:
        output_name = "my_story"
    custom = input(f"\n輸出資料夾名稱（直接 Enter 使用「{output_name}」）: ").strip()
    if custom:
        output_name = custom

    pipeline = VideoPipeline(
        keyword=keyword,
        style=style,
        tts_engine="coqui",
        image_model=image_model,
        output_name=output_name,
        checkpoint_path=checkpoint_path,
        script_path=script_path,
        story_file=story_file,
        story_text=story_text,
    )
    pipeline.run()


def main():
    """主函數：無參數時為選單模式；可傳一個關鍵字快速執行主題模式"""
    if len(sys.argv) <= 1:
        interactive_run()
        return
    keyword = sys.argv[1].strip()
    if keyword in ("-h", "--help", "help"):
        print("用法: python main.py          → 選單模式（故事來源、圖片模型等）")
        print("      python main.py 關鍵字   → 依關鍵字生成（使用預設模型）")
        return
    topics_data = load_topics()
    style = "cinematic"
    for cat in topics_data.get("categories", []):
        if any(kw == keyword for kw in cat.get("keywords", [])):
            style = cat.get("style", "cinematic")
            break
    pipeline = VideoPipeline(
        keyword=keyword,
        style=style,
        tts_engine="coqui",
        image_model="sd15",
        output_name=keyword.replace(" ", "_")[:40],
        checkpoint_path=None,
    )
    pipeline.run()


if __name__ == "__main__":
    main()






