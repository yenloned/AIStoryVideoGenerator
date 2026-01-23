"""
語音生成模組 - 使用本地 Coqui TTS 或 Piper TTS
生成語音音頻文件
"""

import os
import sys
import json
from typing import Dict, List
import subprocess


class AudioGenerator:
    def __init__(
        self,
        tts_engine: str = "coqui",  # "coqui" or "piper"
        output_dir: str = "audio",
        language: str = "zh"
    ):
        """
        初始化語音生成器
        
        Args:
            tts_engine: TTS 引擎類型
            output_dir: 輸出目錄
            language: 語言代碼
        """
        self.tts_engine = tts_engine
        self.output_dir = output_dir
        self.language = language
        
        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🔊 語音生成器初始化，引擎: {tts_engine}")
    
    def check_coqui_available(self) -> bool:
        """檢查 Coqui TTS 是否可用"""
        try:
            import TTS
            return True
        except ImportError:
            return False
    
    def check_piper_available(self) -> bool:
        """檢查 Piper TTS 是否可用"""
        try:
            result = subprocess.run(
                ["piper", "--version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def generate_audio_coqui(self, text: str, output_path: str, speaker_id: str = None) -> str:
        """
        使用 Coqui TTS 生成語音
        
        Args:
            text: 要合成的文字
            output_path: 輸出路徑
            speaker_id: 說話者 ID
            
        Returns:
            生成的音頻文件路徑
        """
        try:
            from TTS.api import TTS
            
            # 初始化 TTS
            # 使用中文模型
            model_name = "tts_models/zh-CN/baker/tacotron2-DDC-GST"
            
            try:
                tts = TTS(model_name=model_name, gpu=True)
            except:
                tts = TTS(model_name=model_name, gpu=False)
            
            # 生成語音
            print(f"🔊 正在生成語音: {text[:30]}...")
            tts.tts_to_file(
                text=text,
                file_path=output_path,
                speaker=speaker_id
            )
            
            print(f"✅ 語音已保存: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Coqui TTS 生成失敗: {e}")
            raise
    
    def generate_audio_piper(self, text: str, output_path: str, model_path: str = None) -> str:
        """
        使用 Piper TTS 生成語音
        
        Args:
            text: 要合成的文字
            output_path: 輸出路徑
            model_path: Piper 模型路徑
            
        Returns:
            生成的音頻文件路徑
        """
        try:
            # 預設中文模型（需要下載）
            if model_path is None:
                model_path = "models/piper/zh_CN/zh_CN-lessac-medium.onnx"
            
            # 使用 Piper 命令行工具
            print(f"🔊 正在生成語音: {text[:30]}...")
            
            # 將文字寫入臨時文件
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                f.write(text)
                temp_text_file = f.name
            
            try:
                # 執行 Piper
                result = subprocess.run(
                    [
                        "piper",
                        "--model", model_path,
                        "--output_file", output_path,
                        temp_text_file
                    ],
                    capture_output=True,
                    check=True,
                    timeout=30
                )
                
                print(f"✅ 語音已保存: {output_path}")
                return output_path
                
            finally:
                # 清理臨時文件
                if os.path.exists(temp_text_file):
                    os.unlink(temp_text_file)
                    
        except subprocess.CalledProcessError as e:
            print(f"❌ Piper TTS 生成失敗: {e.stderr.decode()}")
            raise
        except Exception as e:
            print(f"❌ Piper TTS 錯誤: {e}")
            raise
    
    def generate_audio(self, text: str, output_path: str = None) -> str:
        """
        生成語音（自動選擇可用引擎）
        
        Args:
            text: 要合成的文字
            output_path: 輸出路徑
            
        Returns:
            生成的音頻文件路徑
        """
        if output_path is None:
            import time
            timestamp = int(time.time())
            output_path = os.path.join(self.output_dir, f"audio_{timestamp}.wav")
        
        # 根據引擎類型生成
        if self.tts_engine == "coqui":
            if not self.check_coqui_available():
                print("⚠️  Coqui TTS 不可用，嘗試使用 Piper...")
                self.tts_engine = "piper"
            
            if self.tts_engine == "coqui":
                return self.generate_audio_coqui(text, output_path)
        
        if self.tts_engine == "piper":
            if not self.check_piper_available():
                raise RuntimeError("Piper TTS 不可用，請安裝或配置")
            
            return self.generate_audio_piper(text, output_path)
        
        raise ValueError(f"不支援的 TTS 引擎: {self.tts_engine}")
    
    def generate_audio_for_script(self, script_data: Dict) -> List[str]:
        """
        為整個劇本生成所有語音
        
        Args:
            script_data: 劇本數據
            
        Returns:
            音頻文件路徑列表
        """
        paragraphs = script_data.get("paragraphs", [])
        audio_paths = []
        
        print(f"🔊 開始為 {len(paragraphs)} 個段落生成語音...")
        
        for i, paragraph in enumerate(paragraphs):
            text = paragraph.get("text", "")
            output_path = os.path.join(self.output_dir, f"audio_{i+1:02d}.wav")
            
            try:
                audio_path = self.generate_audio(text, output_path)
                audio_paths.append(audio_path)
            except Exception as e:
                print(f"⚠️  段落 {i+1} 語音生成失敗: {e}")
                continue
        
        print(f"✅ 共生成 {len(audio_paths)} 個音頻文件")
        return audio_paths
    
    def merge_audio_files(self, audio_paths: List[str], output_path: str) -> str:
        """
        合併多個音頻文件
        
        Args:
            audio_paths: 音頻文件路徑列表
            output_path: 輸出路徑
            
        Returns:
            合併後的音頻文件路徑
        """
        try:
            import subprocess
            
            # 使用 FFmpeg 合併音頻
            # 創建文件列表
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                for path in audio_paths:
                    f.write(f"file '{os.path.abspath(path)}'\n")
                list_file = f.name
            
            try:
                # 執行 FFmpeg
                subprocess.run(
                    [
                        "ffmpeg",
                        "-f", "concat",
                        "-safe", "0",
                        "-i", list_file,
                        "-c", "copy",
                        output_path
                    ],
                    check=True,
                    capture_output=True
                )
                
                print(f"✅ 音頻合併完成: {output_path}")
                return output_path
                
            finally:
                if os.path.exists(list_file):
                    os.unlink(list_file)
                    
        except subprocess.CalledProcessError as e:
            print(f"❌ 音頻合併失敗: {e}")
            raise
        except Exception as e:
            print(f"❌ 合併錯誤: {e}")
            raise


def main():
    """測試用主函數"""
    if len(sys.argv) < 2:
        print("用法: python generate_audio.py <script.json> [engine]")
        sys.exit(1)
    
    script_file = sys.argv[1]
    engine = sys.argv[2] if len(sys.argv) > 2 else "coqui"
    
    with open(script_file, "r", encoding="utf-8") as f:
        script_data = json.load(f)
    
    generator = AudioGenerator(tts_engine=engine)
    
    try:
        audio_paths = generator.generate_audio_for_script(script_data)
        print(f"\n生成的音頻: {audio_paths}")
    except Exception as e:
        print(f"錯誤: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()





