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
        
        prompt = f"""你是一名短影片編劇，請用 80–120 字寫一段有懸念、有資訊、有學習性的故事。題材：{keyword}。請分 3–5 段，每段 1–2 句，節奏快。

請以 JSON 格式輸出，格式如下：
{{
  "title": "故事標題",
  "paragraphs": [
    {{"text": "第一段文字", "scene": "場景描述"}},
    {{"text": "第二段文字", "scene": "場景描述"}},
    ...
  ]
}}

只輸出 JSON，不要其他文字。"""

        try:
            print(f"📝 正在生成劇本，關鍵字: {keyword}")
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.8,
                        "top_p": 0.9,
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

