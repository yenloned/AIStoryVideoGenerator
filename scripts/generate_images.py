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
from deep_translator import GoogleTranslator

# 設置 CUDA 記憶體管理環境變數（減少記憶體碎片）
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')


class ImageGenerator:
    def __init__(
        self,
        model_type: str = "sdxl",  # "sdxl" or "sd15"
        device: str = None,
        output_dir: str = "images",
        lora_path: str = None,
        lora_scale: float = 0.8,
        checkpoint_path: str = None
    ):
        """
        初始化圖片生成器
        
        Args:
            model_type: 模型類型 "sdxl" 或 "sd15"
            device: 設備 ("cuda", "cpu", "mps")
            output_dir: 輸出目錄
            lora_path: 可選 LoRA 權重路徑（.safetensors 或目錄），也可用環境變數 LORA_PATH
            lora_scale: LoRA 強度 0~1，預設 0.8；也可用環境變數 LORA_SCALE
            checkpoint_path: 可選本地完整模型路徑（CivitAI 等 .safetensors/.ckpt），也可用環境變數 CHECKPOINT_PATH
        """
        self.model_type = model_type
        self.output_dir = output_dir
        self.pipeline = None
        self.translator = GoogleTranslator(source='auto', target='en')
        self.base_character_prompt = ""  # 用於保持角色一致性
        self.is_turbo = False  # 標記是否為 Turbo 模型
        self.lora_path = lora_path or os.environ.get("LORA_PATH", "").strip() or None
        self.lora_scale = float(os.environ.get("LORA_SCALE", str(lora_scale)))
        self.checkpoint_path = checkpoint_path or os.environ.get("CHECKPOINT_PATH", "").strip() or None
        
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
            
            # 優先：本地完整模型（CivitAI 等下載的 .safetensors / .ckpt）
            if self.checkpoint_path and os.path.exists(self.checkpoint_path):
                print(f"📂 從本地檔案載入: {self.checkpoint_path}")
                dtype = torch.float16 if self.device == "cuda" else torch.float32
                if self.model_type == "sdxl":
                    self.pipeline = StableDiffusionXLPipeline.from_single_file(
                        self.checkpoint_path,
                        torch_dtype=dtype
                    )
                else:
                    self.pipeline = StableDiffusionPipeline.from_single_file(
                        self.checkpoint_path,
                        torch_dtype=dtype
                    )
                print(f"✅ 已載入本地模型（{self.model_type}）")
            elif self.model_type == "sdxl":
                # SDXL 模型（需要更多 VRAM）
                model_id = "stabilityai/stable-diffusion-xl-base-1.0"
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    use_safetensors=True
                )
            else:
                # SD 1.5 模型（較輕量）
                # 使用 Turbo 模型以獲得極速生成（1-4 步即可）
                # 優先順序：SDXL Turbo (最快) -> SD 1.5 Turbo -> DreamShaper 8 -> Realistic Vision
                models_to_try = [
                    ("stabilityai/sdxl-turbo", "SDXL Turbo - 極速模型，1-4 步生成", "sdxl"),
                    ("stabilityai/sd-turbo", "SD 1.5 Turbo - 極速模型，1-4 步生成", "sd15"),
                    ("Lykon/DreamShaper-8", "DreamShaper 8 - 最新版本，平衡模型", "sd15"),
                    ("SG161222/Realistic_Vision_V5.1_noVAE", "Realistic Vision - 寫實風格", "sd15"),
                    ("runwayml/stable-diffusion-v1-5", "原始 SD 1.5 - 穩定版本", "sd15")
                ]
                
                loaded = False
                is_turbo = False
                for model_info in models_to_try:
                    if len(model_info) == 3:
                        model_id, description, model_type = model_info
                    else:
                        model_id, description = model_info
                        model_type = "sd15"
                    
                    try:
                        print(f"📦 嘗試載入模型: {model_id}")
                        print(f"💡 {description}")
                        
                        # 檢查是否為 Turbo 模型
                        if "turbo" in model_id.lower():
                            is_turbo = True
                            print(f"⚡ 這是 Turbo 模型，將使用極少步數（1-4 步）")
                        
                        # 根據模型類型選擇 Pipeline
                        if model_type == "sdxl":
                            # 嘗試 safetensors 優先
                            try:
                                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                                    model_id,
                                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                                    use_safetensors=True
                                )
                            except:
                                print(f"   ⚠️  Safetensors 載入失敗，嘗試標準格式...")
                                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                                    model_id,
                                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                                    use_safetensors=False
                                )
                        else:
                            # SD 1.5 模型
                            try:
                                self.pipeline = StableDiffusionPipeline.from_pretrained(
                                    model_id,
                                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                                    use_safetensors=True
                                )
                            except:
                                print(f"   ⚠️  Safetensors 載入失敗，嘗試標準格式...")
                                self.pipeline = StableDiffusionPipeline.from_pretrained(
                                    model_id,
                                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                                    use_safetensors=False
                                )
                        
                        # 保存是否為 Turbo 模型
                        self.is_turbo = is_turbo
                        print(f"✅ 成功載入: {model_id}")
                        loaded = True
                        break
                    except Exception as e:
                        print(f"   ❌ {model_id} 載入失敗: {str(e)[:100]}...")
                        continue
                
                if not loaded:
                    raise Exception("所有模型載入失敗，請檢查網絡連接或模型可用性")
            
            # 優化記憶體使用（針對 RTX 3070 等 8GB VRAM GPU）
            # 重要：不使用 sequential CPU offload（太慢），使用標準 GPU 模式
            if self.device == "cuda":
                # 直接載入到 GPU（最快）
                self.pipeline = self.pipeline.to(self.device)
                # 輕量優化（不影響速度太多）
                self.pipeline.enable_attention_slicing(4)  # 切片大小 4（較快）
                # 使用 VAE tiling（節省 VRAM，不影響速度）
                if hasattr(self.pipeline, 'vae'):
                    if hasattr(self.pipeline.vae, 'enable_tiling'):
                        self.pipeline.vae.enable_tiling()
                print("💾 已啟用輕量優化（GPU 模式，速度優先）")
            else:
                self.pipeline = self.pipeline.to(self.device)
            
            # 可選：載入 LoRA 權重（見 FINE_TUNING_GUIDE.md）
            if self.lora_path and os.path.exists(self.lora_path):
                try:
                    if os.path.isfile(self.lora_path):
                        lora_dir = os.path.dirname(self.lora_path)
                        weight_name = os.path.basename(self.lora_path)
                        self.pipeline.load_lora_weights(
                            lora_dir,
                            weight_name=weight_name,
                            adapter_name="story_style"
                        )
                    else:
                        self.pipeline.load_lora_weights(
                            self.lora_path,
                            adapter_name="story_style"
                        )
                    self.pipeline.set_adapters(["story_style"], adapter_weights=[self.lora_scale])
                    print(f"✅ LoRA 已載入: {self.lora_path} (scale={self.lora_scale})")
                except Exception as lora_err:
                    print(f"⚠️  LoRA 載入失敗（將不使用 LoRA）: {lora_err}")
            elif self.lora_path:
                print(f"⚠️  LoRA 路徑不存在，跳過: {self.lora_path}")
            
            print("✅ 模型載入完成")
            
        except Exception as e:
            print(f"❌ 模型載入失敗: {e}")
            raise
    
    def translate_to_english(self, text: str) -> str:
        """將中文翻譯為英文（SD 對英文理解更好）"""
        try:
            # 檢測是否包含中文
            if any('\u4e00' <= char <= '\u9fff' for char in text):
                translated = self.translator.translate(text)
                print(f"🔤 翻譯: {text[:20]}... -> {translated[:50]}...")
                return translated
            return text
        except Exception as e:
            print(f"⚠️  翻譯失敗: {e}")
            return text

    def _get_emotion_prompt_from_llm(self, emotion: str) -> str:
        """
        根據 LLM 分析的情感返回相應的提示詞
        
        Args:
            emotion: LLM 分析的情感（positive/negative/neutral）
            
        Returns:
            情感相關的提示詞片段
        """
        emotion_lower = emotion.lower() if emotion else ""
        
        if "positive" in emotion_lower or "happy" in emotion_lower or "joy" in emotion_lower:
            return "joyful atmosphere, warm mood, uplifting emotions, bright lighting, positive energy"
        elif "negative" in emotion_lower or "sad" in emotion_lower or "sorrow" in emotion_lower:
            return "melancholic atmosphere, somber mood, emotional depth, dramatic lighting, expressive emotions"
        elif "neutral" in emotion_lower or "calm" in emotion_lower:
            return "calm atmosphere, peaceful mood, serene emotions, soft lighting, contemplative"
        else:
            # 默認返回空，讓下面的函數處理
            return ""
    
    def _analyze_emotional_context(self, scene_description: str, story_text: str = None) -> str:
        """
        分析場景的情感色彩，添加相應的情感詞彙到提示詞
        
        Args:
            scene_description: 場景描述
            story_text: 故事文本
            
        Returns:
            情感相關的提示詞片段
        """
        # 合併文本進行分析
        combined_text = (scene_description + " " + (story_text or "")).lower()
        
        # 負面情感關鍵詞
        negative_keywords = [
            "悲傷", "難過", "痛苦", "絕望", "失望", "沮喪", "憂愁", "哀傷",
            "害怕", "恐懼", "驚恐", "擔憂", "焦慮", "不安",
            "生氣", "憤怒", "惱怒", "不滿", "怨恨",
            "失敗", "失去", "死亡", "災難", "不幸", "困難", "危險",
            "sad", "sorrow", "pain", "despair", "disappointment", "depression",
            "fear", "afraid", "worried", "anxious", "scared",
            "angry", "rage", "furious", "upset", "hate",
            "failure", "loss", "death", "disaster", "misfortune", "difficulty", "danger"
        ]
        
        # 正面情感關鍵詞
        positive_keywords = [
            "快樂", "開心", "高興", "喜悅", "興奮", "歡樂", "愉快", "滿足",
            "成功", "勝利", "獲得", "成就", "希望", "美好", "幸福", "和平",
            "慶祝", "歡呼", "讚美", "感謝", "愛", "友誼",
            "happy", "joy", "pleasure", "excited", "delighted", "cheerful", "content",
            "success", "victory", "achieve", "accomplishment", "hope", "beautiful", "peace",
            "celebrate", "cheer", "praise", "thank", "love", "friendship"
        ]
        
        # 中性/平靜情感關鍵詞
        neutral_keywords = [
            "平靜", "安寧", "沉思", "思考", "專注", "認真",
            "calm", "peaceful", "serene", "contemplative", "thoughtful", "focused"
        ]
        
        # 檢測情感
        negative_count = sum(1 for keyword in negative_keywords if keyword in combined_text)
        positive_count = sum(1 for keyword in positive_keywords if keyword in combined_text)
        neutral_count = sum(1 for keyword in neutral_keywords if keyword in combined_text)
        
        # 根據情感添加相應的提示詞
        if negative_count > positive_count and negative_count > 0:
            # 負面情感場景
            return "melancholic atmosphere, somber mood, emotional depth, dramatic lighting, expressive emotions"
        elif positive_count > negative_count and positive_count > 0:
            # 正面情感場景
            return "joyful atmosphere, warm mood, uplifting emotions, bright lighting, positive energy"
        elif neutral_count > 0:
            # 中性/平靜場景
            return "calm atmosphere, peaceful mood, serene emotions, soft lighting, contemplative"
        else:
            # 默認：根據場景描述推斷
            if any(word in combined_text for word in ["悲", "傷", "失", "敗", "死", "sad", "loss", "fail"]):
                return "emotional depth, expressive mood"
            elif any(word in combined_text for word in ["喜", "樂", "成", "勝", "歡", "happy", "success", "win"]):
                return "joyful mood, positive atmosphere"
            else:
                return ""  # 無明顯情感，不添加
    
    def check_safety(self, prompt: str) -> bool:
        """簡單的安全檢查"""
        unsafe_words = ["nsfw", "nude", "sex", "naked", "porn", "explicit", "gore", "blood", "violence"]
        return not any(word in prompt.lower() for word in unsafe_words)

    def _build_character_prompt(self, character: Dict) -> str:
        """
        從角色資訊組出精準的英文提示片段：物種、性別、年齡、服裝、民族。
        """
        if not character or not isinstance(character, dict):
            return ""
        parts = []
        breed = (character.get("breed") or "").strip()
        if breed:
            parts.append(breed)
        gender = (character.get("gender") or "").strip().lower()
        if gender in ("male", "female"):
            parts.append(gender)
        age = (character.get("age") or "").strip().lower()
        if age in ("child", "young", "adult", "elder"):
            parts.append(age)
        clothes = (character.get("clothes") or "").strip()
        if clothes:
            parts.append(clothes)
        nation = (character.get("nation") or "").strip()
        if nation:
            parts.append(f"{nation} style")
        if not parts:
            return ""
        raw = ", ".join(parts)
        return self.translate_to_english(raw) if any("\u4e00" <= c <= "\u9fff" for c in raw) else raw

    def generate_image(
        self,
        scene_description: str,
        style: str = "cinematic",
        output_path: str = None,
        width: int = 640,  # 平衡解析度（太高會導致偽影和重複部分）
        height: int = 1152,  # 保持 9:16 比例
        story_title: str = None,
        story_text: str = None,
        paragraph_emotion: str = None,  # LLM 分析的情感
        character: Dict = None,  # main_character: breed, gender, age, clothes, nation
        action: str = None,  # 此段中人物正在做什麼
        image_prompt: str = None  # LLM 輸出的關鍵字串（逗號分隔，tag 風格），優先使用
    ) -> str:
        """
        生成單張圖片
        
        Args:
            scene_description: 場景描述（環境、視覺細節）
            style: 風格選項
            output_path: 輸出路徑
            image_prompt: 若提供則作為主體 positive prompt（關鍵字串，逗號分隔），取代從 scene/character 組成的句子
        """
        if self.pipeline is None:
            self.load_model()
            
        # LLM 提供的關鍵字串（tag 風格）優先作為主 prompt
        # 注意：關鍵字順序很重要，前面的關鍵字權重更高
        keyword_prompt = (image_prompt or "").strip()
        if keyword_prompt and any(c in keyword_prompt for c in "abcdefghijklmnopqrstuvwxyz"):
            # 若有中文則翻譯成英文
            if any("\u4e00" <= c <= "\u9fff" for c in keyword_prompt):
                keyword_prompt = self.translate_to_english(keyword_prompt)
            # 保留原始順序（前面的關鍵字權重更高），只清理空白
            keyword_prompt = ", ".join(t.strip() for t in keyword_prompt.split(",") if t.strip())
            
        # 翻譯場景描述（無 keyword_prompt 時或作為 fallback 用）
        english_description = self.translate_to_english(scene_description)
        
        # 分析場景的情感色彩
        # 優先使用 LLM 分析的情感，如果沒有則從文本分析
        if paragraph_emotion:
            # 使用 LLM 分析的情感
            emotional_context = self._get_emotion_prompt_from_llm(paragraph_emotion)
            print(f"💭 使用 LLM 分析的情感: {paragraph_emotion}")
        else:
            # 從文本分析情感
            emotional_context = self._analyze_emotional_context(scene_description, story_text)
            print(f"💭 從文本分析的情感: {emotional_context[:50] if emotional_context else '中性'}")
        
        # 定義風格提示詞（非寫實風格優先，加強中國風格）
        style_prompts = {
            "anime": "anime style, japanese anime studio style, cel shaded, high quality anime art, vibrant colors, stylized",
            "chinese_ink": "Chinese ink painting style, traditional Chinese shuimo painting, watercolor, monochrome with subtle color accents, artistic brushwork, traditional Chinese art, elegant and refined",
            "cinematic": "cinematic lighting, movie scene, photorealistic, 8k, dramatic atmosphere, depth of field",
            "ancient": "ancient Chinese illustration style, traditional Chinese painting, classical Chinese art, traditional art, stylized, non-photorealistic, historical Chinese aesthetics",
            "fantasy": "fantasy art style, magical atmosphere, vibrant colors, ethereal, stylized illustration",
            "horror": "dark art style, eerie atmosphere, stylized illustration, non-photorealistic",
            "hand_drawn": "hand-drawn illustration, sketch style, artistic drawing, detailed linework, stylized"
        }
        chosen_style = style_prompts.get(style, style_prompts["anime"])  # 默認 anime
        
        # 如果是中國風格，添加更多中國元素提示
        if style in ["chinese_ink", "ancient"]:
            print(f"🇨🇳 使用中國傳統風格: {style}")
        
        # 風格與品質尾綴（兩種路徑共用）
        style_suffix = f"{chosen_style}, detailed, stylized, clear composition, vertical format, simple background, dynamic, highly detailed, sharp focus, high resolution"
        
        if keyword_prompt:
            # 使用 LLM 輸出的關鍵字串作為主體 prompt（tag 風格，多關鍵字）
            # 關鍵字順序已由 LLM 決定（前面的權重更高），直接使用
            # 不強制添加角色（LLM 已決定是否包含角色）
            prompt_parts = ["masterpiece, best quality", keyword_prompt]
            # 只在必要時添加情感上下文（如果 LLM 的 prompt 中沒有明顯情感標記）
            if emotional_context and not any(emotion_word in keyword_prompt.lower() for emotion_word in ["joyful", "melancholic", "calm", "happy", "sad", "focused", "determined", "eager"]):
                # 如果 LLM prompt 中沒有情感相關關鍵字，才添加
                prompt_parts.append(emotional_context)
            prompt_parts.append(style_suffix)
            prompt = ", ".join(prompt_parts)
            tag_count = len([t.strip() for t in keyword_prompt.split(",") if t.strip()])
            print(f"📝 使用 LLM 關鍵字 prompt（{tag_count} tags，順序已保留）")
        else:
            # Fallback：從角色、情感、動作、場景組句（當 LLM 沒有提供 image_prompt 時）
            # 注意：角色不一定需要存在，根據場景描述判斷
            character_subject = ""
            # 只在場景描述或文本中明確提到角色時才添加
            if character:
                # 檢查場景描述或文本中是否提到角色相關內容
                scene_lower = (scene_description + " " + (story_text or "")).lower()
                has_character_mention = any(
                    word in scene_lower for word in 
                    ["人", "角色", "主角", "他", "她", "person", "character", "man", "woman", "boy", "girl", "people"]
                )
                if has_character_mention:
                    character_subject = self._build_character_prompt(character)
                    if character_subject and not self.base_character_prompt and story_title:
                        self.base_character_prompt = character_subject
                    if not self.base_character_prompt and story_title:
                        self.base_character_prompt = "consistent character"
            
            action_english = ""
            if action and str(action).strip():
                action_english = self.translate_to_english(str(action).strip())
            
            # 按重要性排序：品質標籤 → 角色（如果存在）→ 情感 → 動作 → 場景 → 風格
            prompt_parts = ["masterpiece, best quality"]
            if character_subject:
                prompt_parts.append(f"({character_subject}:1.2)")
            elif self.base_character_prompt:
                prompt_parts.append(f"({self.base_character_prompt}:1.2)")
            if emotional_context:
                prompt_parts.append(emotional_context)
            if action_english:
                prompt_parts.append(action_english)
            prompt_parts.append(english_description)
            prompt_parts.append(style_suffix)
            prompt = ", ".join(prompt_parts)
            
            words = prompt.split()
            if len(words) > 75:
                essential_parts = ["masterpiece, best quality"]
                if character_subject:
                    essential_parts.append(f"({character_subject}:1.2)")
                elif self.base_character_prompt:
                    essential_parts.append(f"({self.base_character_prompt}:1.2)")
                if emotional_context:
                    essential_parts.append(emotional_context)
                if action_english:
                    essential_parts.append(action_english)
                essential_parts.append(english_description[:100] if len(english_description) > 100 else english_description)
                essential_parts.append(f"{chosen_style}, vertical format")
                prompt = ", ".join(essential_parts)
        
        print(f"📝 提示詞長度: {len(prompt.split())} 詞（約 {len(prompt.split()) * 1.3:.0f} tokens）")
        
        # Negative Prompt: 強化以減少解剖錯誤和重複部分
        negative_prompt = "nsfw, nude, explicit, sexual, modern clothing, modern architecture, unrelated objects, blurry, low quality, distorted, watermark, bad anatomy, extra limbs, extra fingers, extra arms, extra legs, duplicated body parts, malformed limbs, missing limbs, fused fingers, too many fingers, cropped, out of frame, jpeg artifacts, text, signature, deformed, disfigured, mutation, mutated, ugly, bad proportions, extra digits, fewer digits, missing digits, bad hands, bad feet"
        
        # 安全檢查
        if not self.check_safety(prompt):
            print("⚠️  提示詞包含不安全內容，已跳過")
            return None

        try:
            preview = (keyword_prompt[:60] + "...") if keyword_prompt else (english_description[:50] + "...")
            print(f"🎨 正在生成圖片: {preview}")
            
            # 清除 CUDA 快取
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # 設置生成器
            generator = None
            if self.device == "cuda":
                generator = torch.Generator(device="cuda")
                generator.manual_seed(42)  # 固定種子以提高一致性
            
            # 生成圖片 (高質量優化，優先細節)
            # 使用更多步數和更好的調度器以獲得更好細節
            if self.is_turbo:
                # Turbo 模型：可以用更多步數獲得更好質量
                num_steps = 4  # Turbo 模型用 4 步（比 1 步質量更好）
                guidance_scale = 1.0  # Turbo 模型可以用少量 guidance
                print("⚡ 使用 Turbo 模型（4 步生成，平衡速度與質量）")
            else:
                # 普通模型：使用高質量調度器，更多步數
                try:
                    from diffusers import DPMSolverMultistepScheduler
                    # 使用 DPM++ 2M Karras（高質量）
                    self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                        self.pipeline.scheduler.config,
                        use_karras_sigmas=True
                    )
                    num_steps = 20  # 20 步是質量與速度的最佳平衡（太多步會產生偽影）
                    print("🎨 使用高質量調度器 (DPM++ 2M Karras, 20 steps)")
                except:
                    try:
                        from diffusers import EulerAncestralDiscreteScheduler
                        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                            self.pipeline.scheduler.config
                        )
                        num_steps = 20
                        print("🎨 使用高質量調度器 (Euler Ancestral, 20 steps)")
                    except:
                        num_steps = 20
                        print("⚙️  使用默認調度器 (20 steps)")
                guidance_scale = 7.0  # 7.0 是標準值，太高會過度飽和
            
            print(f"⏳ 開始生成（預計需要 30-50 秒）...")
            import time
            start_time = time.time()
            
            image = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt if not self.is_turbo else None,  # Turbo 模型不需要 negative prompt
                width=width,
                height=height,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
            
            elapsed = time.time() - start_time
            print(f"✅ 生成完成，耗時: {elapsed:.1f} 秒")
            
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
            # 如果失敗，嘗試降低解析度重試
            if "out of memory" in str(e).lower() and width > 512:
                print("⚠️  VRAM 不足，嘗試降低解析度重試...")
                return self.generate_image(
                    scene_description, style, output_path,
                    width=512, height=896,
                    story_title=story_title, story_text=story_text,
                    paragraph_emotion=paragraph_emotion, character=character, action=action,
                    image_prompt=image_prompt,
                )
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
        
        # 重置角色特徵
        self.base_character_prompt = ""
        
        # 獲取整體情感與主角資訊（若 LLM 有提供）
        overall_emotion = script_data.get("emotion", None)
        main_character = script_data.get("main_character", None)
        # 若腳本未提供 main_character，仍可只用 scene / emotion / action
        style_override = script_data.get("style", style)
        
        for i, paragraph in enumerate(paragraphs):
            scene = paragraph.get("scene", paragraph.get("text", ""))
            text = paragraph.get("text", "")
            paragraph_emotion = paragraph.get("emotion", overall_emotion)
            action = paragraph.get("action", "").strip() or None
            image_prompt = paragraph.get("image_prompt", "").strip() or None  # LLM 輸出的關鍵字串（tag 風格）
            output_path = os.path.join(self.output_dir, f"scene_{i+1:02d}.png")
            
            try:
                if self.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                img_path = self.generate_image(
                    scene_description=scene,
                    style=style_override or style,
                    output_path=output_path,
                    story_title=story_title,
                    story_text=text,
                    paragraph_emotion=paragraph_emotion,
                    character=main_character,
                    action=action,
                    image_prompt=image_prompt,
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





