"""
劇本生成模組 - 使用 Ollama + Qwen 2.5 7B
生成結構化的故事段落
"""

import json
import requests
import sys
from typing import List, Dict


class ScriptGenerator:
    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        """
        初始化劇本生成器
        
        Args:
            ollama_base_url: Ollama 服務的 URL
        """
        self.ollama_url = ollama_base_url
        self.model_name = "qwen2.5:7b"
    
    def check_ollama_connection(self) -> bool:
        """檢查 Ollama 是否運行中"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Ollama 連接失敗: {e}")
            return False
    
    def check_model_available(self) -> bool:
        """檢查模型是否可用"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            return any(self.model_name in name for name in model_names)
        except Exception as e:
            print(f"❌ 檢查模型失敗: {e}")
            return False
    
    def generate_script(self, keyword: str) -> Dict:
        """
        生成劇本
        
        Args:
            keyword: 題材關鍵字
            
        Returns:
            包含段落列表的字典
        """
        if not self.check_ollama_connection():
            raise ConnectionError("無法連接到 Ollama，請確認 Ollama 正在運行")
        
        if not self.check_model_available():
            print(f"⚠️  模型 {self.model_name} 未找到，嘗試使用...")
        
        prompt = f"""你是專業編劇和視覺設計師，講述中國成語故事。

題材：{keyword}

要求：
1. 如果是成語，講述真實典故和完整故事
2. 故事必須分為 5-8 段（paragraphs），每段 3-5 句（可以更長，根據需要）。每段文字長度根據段落總數調整，確保總語音時長在 20-60 秒之間（中文 TTS 約 2.5-4 字/秒）：
   - 5段：每段約 30-48 字，總計約 150-240 字（總時長約 37-60 秒）
   - 6段：每段約 25-40 字，總計約 150-240 字（總時長約 37-60 秒）
   - 7段：每段約 21-34 字，總計約 147-238 字（總時長約 37-60 秒）
   - 8段：每段約 18-30 字，總計約 144-240 字（總時長約 36-60 秒）
   - 【重要】所有段落加起來的總語音時長必須在 20-60 秒之間。如果段落較少（5-6段），每段可以更長（4-6句，40-50字）；如果段落較多（7-8段），每段可以稍短（3-4句，20-30字），以確保總時長符合要求。
3. 場景描述要具體、視覺化
4. 分析故事的情感基調和視覺風格

輸出 JSON 格式：
{{
  "title": "故事標題",
  "emotion": "故事整體情感（positive/negative/neutral）",
  "style": "推薦的圖片風格（anime/chinese_ink/ancient/cinematic/fantasy/hand_drawn）",
  "reason": "為什麼選擇這個風格和情感（簡短說明）",
  "main_character": {{
    "breed": "human 或動物種類（如 human, dog, fox, dragon），主角物種",
    "gender": "male 或 female 或 other",
    "age": "child, young, adult 或 elder",
    "clothes": "服裝描述（如 ancient Chinese robe, scholar hat）",
    "nation": "文化/民族（如 Chinese, Japanese）"
  }},
  "paragraphs": [
    {{"text": "第一段文字", "scene": "場景描述", "emotion": "這段的情感", "action": "此段中人物正在做什麼", "image_prompt": "1boy, young, scholar, black hair, short hair, eager expression, focused eyes, traditional Chinese robe, blue robe, long sleeves, sitting pose, cross-legged, hole in wall, cracked wall, stone wall texture, light through hole, beam of light, warm light, flickering candlelight, dim room, dark room, shadows, high contrast, books, ancient books, scroll, ink brush, ink stone, paper, reading pose, leaning forward, hand holding book, concentrated expression, scholarly atmosphere, quiet study, candle flame, wooden desk, traditional furniture, rough wall surface, dust particles, light rays, illumination, darkness, contrast ratio, close-up composition, eye level perspective, warm color palette, orange light, yellow candlelight, detailed textures, intricate patterns, high quality, sharp focus"}},
    {{"text": "第二段文字（例如：軍隊場景）", "scene": "場景描述", "emotion": "這段的情感", "action": "此段中人物正在做什麼", "image_prompt": "army, soldiers, troops, multiple people, crowd, group of soldiers, military formation, army ranks, many people, diverse crowd, various ages, different clothing, armor, weapons, helmets, military uniforms, organized ranks, unified movement, collective action, battlefield, wide shot, medium shot, group composition, crowd density, spatial arrangement, formation type, group dynamics, individual details, mixed expressions, varied poses, different body types, various occupations, determined expressions, resolute stance, military discipline, organized structure, high quality, detailed textures"}},
    {{"text": "第三段文字", "scene": "場景描述", "emotion": "這段的情感", "action": "此段中人物正在做什麼", "image_prompt": "必須包含至少 40-60 個詳細關鍵字，涵蓋角色（單人或多人/群體）、服裝、動作、環境、光線、色彩、構圖、背景、道具、氛圍、細節等所有視覺層面"}},
    {{"text": "第四段文字", "scene": "場景描述", "emotion": "這段的情感", "action": "此段中人物正在做什麼", "image_prompt": "必須包含至少 40-60 個詳細關鍵字，涵蓋角色（單人或多人/群體）、服裝、動作、環境、光線、色彩、構圖、背景、道具、氛圍、細節等所有視覺層面"}},
    {{"text": "第五段文字", "scene": "場景描述", "emotion": "這段的情感", "action": "此段中人物正在做什麼", "image_prompt": "必須包含至少 40-60 個詳細關鍵字，涵蓋角色（單人或多人/群體）、服裝、動作、環境、光線、色彩、構圖、背景、道具、氛圍、細節等所有視覺層面"}},
    {{"text": "第六段文字（可選，根據故事需要）", "scene": "場景描述", "emotion": "這段的情感", "action": "此段中人物正在做什麼", "image_prompt": "必須包含至少 40-60 個詳細關鍵字，涵蓋角色（單人或多人/群體）、服裝、動作、環境、光線、色彩、構圖、背景、道具、氛圍、細節等所有視覺層面"}},
    {{"text": "第七段文字（可選，根據故事需要）", "scene": "場景描述", "emotion": "這段的情感", "action": "此段中人物正在做什麼", "image_prompt": "必須包含至少 40-60 個詳細關鍵字，涵蓋角色（單人或多人/群體）、服裝、動作、環境、光線、色彩、構圖、背景、道具、氛圍、細節等所有視覺層面"}},
    {{"text": "第八段文字（可選，根據故事需要）", "scene": "場景描述", "emotion": "這段的情感", "action": "此段中人物正在做什麼", "image_prompt": "必須包含至少 40-60 個詳細關鍵字，涵蓋角色（單人或多人/群體）、服裝、動作、環境、光線、色彩、構圖、背景、道具、氛圍、細節等所有視覺層面"}}
  ]
  
注意：paragraphs 陣列必須包含 5-8 個段落。LLM 根據故事複雜度和需要決定具體數量，但最少 5 段，最多 8 段。
}}

image_prompt 規則（每段必填，全部英文；由你根據該段故事與場景決定內容，要極度詳細）：
- 【關鍵要求】每段 image_prompt 必須包含至少 40-60 個關鍵字，少於 40 個關鍵字視為不合格。這是強制要求！
- 【角色數量判斷】根據文字和場景描述判斷角色數量：
  * 如果文字提到「士兵們」「眾人」「群眾」「軍隊」「部隊」「人群」等複數概念，必須使用群體關鍵字（army, soldiers, crowd, multiple people, group等）
  * 如果文字只提到單一角色（如「他」「她」「一個人」），使用單人關鍵字（1boy/1girl/1man/1woman）
  * 如果場景中有多個人物但文字未明確說明，根據場景描述判斷（如「圍了過來」「聚集」「列隊」等表示多人）
- 必須「具體反映該段文字與 scene」的視覺細節，不要漏掉關鍵元素。例如：文中若寫「鄰居的燈光透過小洞照進來」→ 要出現 hole in wall, light through hole, dim room, beam of light, shadow, contrast, warm light, flickering, stone wall texture, rough surface, dust particles, light rays, illumination, darkness, contrast ratio；若寫「鑿壁」→ wall, hole, cracked wall, stone wall, tool marks, chisel marks, rough texture, ancient masonry, weathered surface, broken fragments, dust, debris；若寫「如饑似渴地閱讀」→ reading, focused, books, scroll, ink brush, candlelight, concentration, intense gaze, leaning forward, hand holding book, page turning, text visible, scholarly atmosphere, quiet study, dedication。每段都先讀懂再寫 image_prompt。
- 關鍵字順序非常重要：越前面的關鍵字權重越高（更聚焦）。必須按照重要性排序。
- 【必須包含】以下所有相關方面的關鍵字（如果該段有相關內容，每類至少 3-5 個關鍵字）：
  * 情感/情緒/表情：emotion, expression, mood, atmosphere, facial expression, eye expression, body language, emotional state（如 eager, focused, determined, sad, joyful, contemplative, melancholic, desperate, heroic, tragic, intense, calm, anxious, resolute）
  * 角色（如果存在）：可以是單人或多人群體：
    - 單人：1boy/1girl/1man/1woman, age (child/young/adult/elder), occupation, hair color, hair style, hair length, eye color, eye shape, facial features, body type, physique, skin tone, facial hair, accessories
    - 多人/群體：multiple people, crowd, group, army, soldiers, troops, warriors, crowd of people, many people, group of soldiers, army formation, military ranks, crowd scene, gathering, assembly, multitude, throng, horde, battalion, regiment, platoon, squad, team, group of men, group of women, group of children, mixed group, diverse crowd
    - 群體特徵：group size (few/many/crowd), group composition, group arrangement, formation type, crowd density, people distribution, spatial arrangement, group dynamics, collective action, unified movement, scattered individuals, organized ranks, chaotic crowd, orderly formation
    - 個體特徵（當描述群體中的個體時）：individual details, various ages, different clothing, diverse appearances, mixed expressions, varied poses, different body types, various occupations
  * 服裝/配飾：clothing details, clothing style, fabric type, clothing colors, accessories, jewelry, headwear, footwear, traditional/modern style, textures, patterns, decorations, armor pieces, weapon holsters
  * 動作/姿勢：pose, action, body position, gesture, movement, stance, posture, hand position, arm position, leg position, head angle, body orientation, dynamic/static pose
  * 環境細節：location type, architecture style, building materials, furniture type, furniture style, objects placement, materials, textures, surface details, structural elements, decorative elements
  * 光線/照明：lighting type, light source, brightness level, shadows, shadow direction, contrast, time of day, light color, light intensity, light direction, ambient light, key light, rim light, backlight, candlelight, firelight, moonlight, sunlight, torchlight
  * 色彩/色調：color palette, warm/cool tones, saturation level, mood colors, dominant colors, accent colors, color harmony, color contrast, monochrome elements, color temperature
  * 構圖/視角：composition style, camera angle, perspective type, framing style, depth of field, focal point, rule of thirds, leading lines, symmetry, asymmetry, close-up, medium shot, wide shot, bird's eye view, worm's eye view, eye level
  * 背景元素：background details, scenery type, landscape features, sky appearance, ground texture, distant objects, horizon line, atmospheric perspective, depth layers, foreground, midground, background separation
  * 道具/物品：objects in scene, tools, books, furniture, decorations, symbolic items, weapons, containers, utensils, scrolls, documents, artifacts, personal belongings, environmental objects
  * 天氣/氛圍：weather conditions, atmosphere type, mist, fog, dust, particles, smoke, wind effects, precipitation, cloud formations, air quality, visibility, environmental effects
  * 細節品質：detailed, intricate, fine details, textures, patterns, craftsmanship, high quality, sharp focus, clear details, realistic rendering, artistic style, brush strokes, line work
- 注意：角色不一定每段都出現（例如純風景、環境描述）。角色可以是單人（1boy/1girl/1man/1woman）或多人群體（army, soldiers, crowd, multiple people, group等）。如果該段沒有角色，就用更多環境、背景、氛圍關鍵字來補足，確保達到 40-60 個關鍵字。如果場景中有軍隊、群眾、多人，必須使用群體相關關鍵字（army, soldiers, crowd, multiple people, group等），並描述群體特徵和個體細節。
- 只輸出「逗號分隔的關鍵字/標籤」，不寫完整句子。格式：keyword1, keyword2, keyword3, ...
- 重要元素可加權重，如 (light through hole:1.3), (focused expression:1.2), (dramatic lighting:1.4)
- 【再次強調】每段必須包含 40-60 個關鍵字，涵蓋所有視覺層面。不要重複同義標籤，但要包含相關的細節變體和不同角度的描述。
- 由你根據故事理解決定要強調的視覺，但必須全面覆蓋所有相關方面，並且達到關鍵字數量要求。

風格選擇指南：
- chinese_ink: 中國傳統故事、古典文學、水墨畫風格
- ancient: 古代歷史故事、傳統文化
- anime: 現代化故事、動畫風格
- cinematic: 電影感、寫實風格
- fantasy: 神話、奇幻故事
- hand_drawn: 手繪插圖風格

如果是中國傳統故事（成語、歷史典故），優先使用 chinese_ink 或 ancient 風格。

【重要】paragraphs 陣列必須包含 5-8 個段落（最少 5 段，最多 8 段）。LLM 根據故事複雜度和完整度決定具體數量，但必須在此範圍內。

【視頻時長要求】所有段落加起來的總語音時長必須在 20-60 秒之間（中文 TTS 約 2.5-4 字/秒）。每段文字長度根據段落總數調整：如果段落較少（5-6段），每段可以更長（4-6句，30-50字）；如果段落較多（7-8段），每段可以稍短（3-4句，18-35字），以確保總時長在 20-60 秒範圍內。每段可以比基本要求更長一些，但必須確保總時長不超過 60 秒。

只輸出 JSON，確保完整閉合所有括號。故事必須與「{keyword}」相關。"""

        try:
            print(f"📝 正在生成劇本，關鍵字: {keyword}")
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,  # 降低溫度以獲得更準確的故事
                        "top_p": 0.9,
                        "num_predict": 6000,  # 增加生成長度以獲得完整故事（5-8 段需要更多 tokens）
                    }
                },
                timeout=180  # 增加超時時間
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API 錯誤: {response.status_code}")
            
            result = response.json()
            response_text = result.get("response", "")
            
            # 嘗試提取 JSON
            json_text = self._extract_json(response_text)
            script_data = json.loads(json_text)
            
            # 驗證數據結構
            if "paragraphs" not in script_data or not isinstance(script_data["paragraphs"], list):
                raise ValueError("生成的劇本格式不正確")
            
            # 驗證段落數量（必須 5-8 段）
            paragraph_count = len(script_data["paragraphs"])
            if paragraph_count < 5:
                print(f"⚠️  段落數量不足：{paragraph_count} 段（需要至少 5 段）")
                raise ValueError(f"劇本段落數量不足：{paragraph_count} 段，需要至少 5 段")
            elif paragraph_count > 8:
                print(f"⚠️  段落數量過多：{paragraph_count} 段（最多 8 段）")
                raise ValueError(f"劇本段落數量過多：{paragraph_count} 段，最多 8 段")
            
            # 估算總文字長度和視頻時長（中文 TTS 約 2.5-4 字/秒）
            total_chars = sum(len(p.get("text", "")) for p in script_data["paragraphs"])
            estimated_duration_min = total_chars / 4.0  # 較快語速
            estimated_duration_max = total_chars / 2.5  # 較慢語速
            avg_chars_per_para = total_chars / paragraph_count if paragraph_count > 0 else 0
            
            print(f"📊 文字統計：總字數 {total_chars} 字，平均每段 {avg_chars_per_para:.1f} 字")
            print(f"⏱️  預估視頻時長：{estimated_duration_min:.1f}-{estimated_duration_max:.1f} 秒")
            
            if estimated_duration_min < 18:
                print(f"⚠️  警告：預估時長可能低於 20 秒（{estimated_duration_min:.1f} 秒），建議增加段落長度")
            elif estimated_duration_max > 65:
                print(f"⚠️  警告：預估時長可能超過 60 秒（{estimated_duration_max:.1f} 秒），建議縮短段落長度")
            
            # 顯示 LLM 的分析結果
            if "emotion" in script_data:
                print(f"💭 LLM 分析的情感: {script_data['emotion']}")
            if "style" in script_data:
                print(f"🎨 LLM 推薦的風格: {script_data['style']}")
                if "reason" in script_data:
                    print(f"📝 推薦理由: {script_data['reason']}")
            
            print(f"✅ 劇本生成成功，共 {paragraph_count} 段（符合 5-8 段要求）")
            return script_data
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON 解析失敗: {e}")
            print(f"原始回應長度: {len(response_text)} 字符")
            print(f"原始回應前 500 字符: {response_text[:500]}...")
            if len(response_text) > 500:
                print(f"原始回應後 200 字符: ...{response_text[-200:]}")
            
            # 嘗試再次提取和修復
            try:
                print("🔄 嘗試修復 JSON...")
                json_text = self._extract_json(response_text)
                script_data = json.loads(json_text)
                print(f"✅ JSON 修復成功，共 {len(script_data.get('paragraphs', []))} 段")
                return script_data
            except Exception as e2:
                print(f"❌ JSON 修復也失敗: {e2}")
                raise
        except Exception as e:
            print(f"❌ 生成劇本失敗: {e}")
            raise
    
    def _extract_json(self, text: str) -> str:
        """從文本中提取 JSON，嘗試修復不完整的 JSON"""
        # 尋找 JSON 開始
        start_idx = text.find("{")
        if start_idx == -1:
            raise ValueError("無法在回應中找到 JSON 開始標記")
        
        # 尋找 JSON 結束（從後往前找最後一個 }）
        end_idx = text.rfind("}")
        if end_idx == -1 or end_idx <= start_idx:
            # JSON 可能不完整，嘗試修復
            print("⚠️  檢測到不完整的 JSON，嘗試修復...")
            # 計算開括號和閉括號的數量
            open_braces = text[start_idx:].count("{")
            close_braces = text[start_idx:].count("}")
            
            if close_braces < open_braces:
                # 缺少閉括號，添加它們
                missing = open_braces - close_braces
                text = text + "}" * missing
                print(f"   添加了 {missing} 個閉括號")
            
            end_idx = text.rfind("}")
        
        json_text = text[start_idx:end_idx + 1]
        
        # 嘗試修復常見的 JSON 問題
        # 1. 移除尾隨的逗號
        import re
        json_text = re.sub(r',\s*}', '}', json_text)
        json_text = re.sub(r',\s*]', ']', json_text)
        
        # 2. 如果最後的段落不完整，嘗試修復
        if '"scene":' in json_text and json_text.count('"scene":') > json_text.count('"scene": "') + json_text.count('"scene":'):
            # 可能有未完成的 scene 字段
            last_scene_idx = json_text.rfind('"scene":')
            if last_scene_idx != -1:
                # 檢查是否有閉引號
                after_scene = json_text[last_scene_idx + 8:]
                if '"' not in after_scene[:50] or after_scene.strip().startswith('"') and '"' not in after_scene[1:100]:
                    # scene 字段可能不完整，嘗試補全
                    # 找到下一個可能的結束位置
                    next_comma = after_scene.find(',')
                    next_brace = after_scene.find('}')
                    if next_comma != -1 and (next_brace == -1 or next_comma < next_brace):
                        # 在逗號前添加閉引號
                        json_text = json_text[:last_scene_idx + 8] + ' "' + after_scene[:next_comma] + '",' + after_scene[next_comma + 1:]
        
        return json_text


def script_from_story_text(story_text: str, title: str = None) -> Dict:
    """
    將使用者輸入的純文字故事轉成劇本格式（用於 --story / --story-file）。
    依段落分割（雙換行或單換行），每段作為一個 paragraph，scene 與 text 相同。
    """
    text = (story_text or "").strip()
    if not text:
        return {"title": title or "My Story", "paragraphs": []}
    # 先以雙換行分大段，再以單換行分（避免一大塊沒分段）
    raw_paragraphs = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]
    if not raw_paragraphs:
        raw_paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    paragraphs = []
    for p in raw_paragraphs:
        paragraphs.append({
            "text": p,
            "scene": p[:300] if len(p) > 300 else p,  # 場景描述可略短
            "emotion": "neutral",
        })
    return {
        "title": title or raw_paragraphs[0][:50] if raw_paragraphs else "My Story",
        "emotion": "neutral",
        "style": "cinematic",
        "paragraphs": paragraphs,
    }


def main():
    """測試用主函數"""
    if len(sys.argv) < 2:
        print("用法: python generate_script.py <關鍵字>")
        sys.exit(1)
    
    keyword = sys.argv[1]
    generator = ScriptGenerator()
    
    try:
        script = generator.generate_script(keyword)
        print("\n生成的劇本:")
        print(json.dumps(script, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"錯誤: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()





