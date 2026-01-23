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
        
        prompt = f"""你是一名專業的短影片編劇，專門講述中國成語故事、歷史典故和傳統文化故事。

題材：{keyword}

**重要要求：**
1. 如果關鍵字是成語（如「塞翁失馬」「守株待兔」等），你必須講述該成語的**真實典故和完整故事**，不要編造不相關的內容。
2. 如果關鍵字是歷史典故，必須基於真實歷史事件或傳說。
3. 故事要完整、生動、有教育意義，總字數 150-250 字。
4. 每段文字要具體描述場景和情節，場景描述要詳細、視覺化。

請寫一個完整的故事，分 4-6 段，每段 2-3 句。場景描述要具體，包含：
- 人物動作和表情
- 環境細節（地點、時間、天氣等）
- 視覺元素（顏色、物品、建築等）

請以 JSON 格式輸出，格式如下：
{{
  "title": "故事標題（必須與關鍵字相關）",
  "paragraphs": [
    {{"text": "第一段文字（2-3句，描述具體情節）", "scene": "詳細的場景描述，包含人物、動作、環境、視覺細節"}},
    {{"text": "第二段文字（2-3句）", "scene": "詳細的場景描述"}},
    ...
  ]
}}

**示例（塞翁失馬）：**
{{
  "title": "塞翁失馬",
  "paragraphs": [
    {{"text": "邊境有一位老翁，他養了一匹好馬。一天，這匹馬突然跑丟了，鄰居們都來安慰他。", "scene": "古代邊境小村莊，一位白髮蒼蒼的老翁站在簡陋的農舍前，周圍是黃土和低矮的籬笆，遠處可見邊境山巒，幾位鄰居圍著老翁，表情關切"}},
    {{"text": "老翁卻說：『這未必是壞事。』果然，幾個月後，那匹馬帶著一匹駿馬回來了。", "scene": "幾個月後，夕陽西下，老翁的馬帶著一匹更健壯的駿馬回到農舍，老翁站在門口微笑，鄰居們驚訝地看著這一幕，背景是金色的夕陽和遠山"}},
    ...
  ]
}}

只輸出 JSON，不要其他文字。確保故事內容與關鍵字「{keyword}」完全相關。"""

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
                        "num_predict": 1000,  # 增加生成長度以獲得完整故事
                    }
                },
                timeout=120
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
            
            print(f"✅ 劇本生成成功，共 {len(script_data['paragraphs'])} 段")
            return script_data
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON 解析失敗: {e}")
            print(f"原始回應: {response_text[:200]}...")
            raise
        except Exception as e:
            print(f"❌ 生成劇本失敗: {e}")
            raise
    
    def _extract_json(self, text: str) -> str:
        """從文本中提取 JSON"""
        # 尋找 JSON 開始和結束
        start_idx = text.find("{")
        end_idx = text.rfind("}") + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("無法在回應中找到 JSON")
        
        return text[start_idx:end_idx]


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





