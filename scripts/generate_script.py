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
2. 故事 150-250 字，分 4-5 段，每段 2-3 句
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
    {{"text": "第一段文字", "scene": "場景描述", "emotion": "這段的情感", "action": "此段中人物正在做什麼", "image_prompt": "1boy, young, scholar, black hair, eager expression, traditional robe, hole in wall, light through hole, dim room, books, ink and brush, sitting, reading"}},
    {{"text": "第二段文字", "scene": "場景描述", "emotion": "這段的情感", "action": "此段中人物正在做什麼", "image_prompt": "依該段文字與 scene 寫出詳細關鍵字，含年齡/職業/姿勢與本段特有視覺"}},
    {{"text": "第三段文字", "scene": "場景描述", "emotion": "這段的情感", "action": "此段中人物正在做什麼", "image_prompt": "同上"}},
    {{"text": "第四段文字", "scene": "場景描述", "emotion": "這段的情感", "action": "此段中人物正在做什麼", "image_prompt": "同上"}}
  ]
}}

image_prompt 規則（每段必填，全部英文；由你根據該段故事與場景決定內容）：
- 必須「具體反映該段文字與 scene」的視覺細節，不要漏掉關鍵元素。例如：文中若寫「鄰居的燈光透過小洞照進來」→ 要出現 hole in wall, light through hole, dim room, beam of light；若寫「鑿壁」→ wall, hole；若寫「如饑似渴地閱讀」→ reading, focused, books。每段都先讀懂再寫 image_prompt。
- 必含：角色年齡（依故事判斷）child / young / adult / elder、職業 career、姿勢 pose。例如鑿壁偷光的匡衡可為 young boy 或 child。
- 只輸出「逗號分隔的關鍵字/標籤」，不寫完整句子。順序建議：情緒/表情 → 角色（1boy/1girl, 年齡, 職業, 髮色, 眼色, 服裝）→ 環境與本段特有細節（光線、洞、牆、書、燭台等）→ 背景 → 手中/身旁物品 → 姿勢。重要可加權如 (light through hole:1.2)。
- 要詳細、多關鍵字，但不重複同義標籤。由你根據故事理解決定要強調的視覺。

風格選擇指南：
- chinese_ink: 中國傳統故事、古典文學、水墨畫風格
- ancient: 古代歷史故事、傳統文化
- anime: 現代化故事、動畫風格
- cinematic: 電影感、寫實風格
- fantasy: 神話、奇幻故事
- hand_drawn: 手繪插圖風格

如果是中國傳統故事（成語、歷史典故），優先使用 chinese_ink 或 ancient 風格。

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
                        "num_predict": 2000,  # 增加生成長度以獲得完整故事（從 1000 增加到 2000）
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
            
            # 顯示 LLM 的分析結果
            if "emotion" in script_data:
                print(f"💭 LLM 分析的情感: {script_data['emotion']}")
            if "style" in script_data:
                print(f"🎨 LLM 推薦的風格: {script_data['style']}")
                if "reason" in script_data:
                    print(f"📝 推薦理由: {script_data['reason']}")
            
            print(f"✅ 劇本生成成功，共 {len(script_data['paragraphs'])} 段")
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





