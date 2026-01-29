"""
語音生成模組 - 使用本地 Coqui TTS 或 Piper TTS
支援情感參考音（更自然、有感情）與輸出正規化（更乾淨）
"""

import os
import sys
import json
from typing import Dict, List, Optional
import subprocess


class AudioGenerator:
    def __init__(
        self,
        tts_engine: str = "coqui",  # "coqui" or "piper"
        output_dir: str = "audio",
        language: str = "zh",
        reference_wav: Optional[str] = None,
        clean_output: bool = True,
    ):
        """
        初始化語音生成器
        
        Args:
            tts_engine: TTS 引擎類型
            output_dir: 輸出目錄
            language: 語言代碼
            reference_wav: 可選，6–10 秒情感參考音檔路徑（XTTS 用於更自然、有感情的語調）。參考音可以是任一種語言，XTTS 只取音色與語調，合成語言由 language 決定。
            clean_output: 是否對輸出做正規化與輕量壓縮，使音量一致、更乾淨
        """
        self.tts_engine = tts_engine
        self.output_dir = output_dir
        self.language = language
        self.reference_wav = reference_wav
        self.clean_output = clean_output
        
        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🔊 語音生成器初始化，引擎: {tts_engine}")
        if reference_wav and os.path.exists(reference_wav):
            print(f"   情感參考音: {reference_wav}")
        if clean_output:
            print("   輸出: 正規化音量、輕量壓縮（更乾淨）")
    
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
    
    def _normalize_and_clean_audio(self, wav_path: str) -> None:
        """音量正規化 + 目標響度（更乾淨、一致）"""
        if not self.clean_output or not os.path.exists(wav_path):
            return
        try:
            from pydub import AudioSegment, effects
            seg = AudioSegment.from_wav(wav_path)
            seg = effects.normalize(seg)
            target_dBFS = -20.0
            diff = target_dBFS - seg.dBFS
            if abs(diff) > 0.5:
                seg = seg.apply_gain(diff)
            seg.export(wav_path, format="wav")
        except ImportError:
            pass
        except Exception as e:
            print(f"⚠️ 音頻正規化跳過: {e}")
    
    def generate_audio_coqui(self, text: str, output_path: str, speaker_id: str = None, emotion: str = None) -> str:
        """
        使用 Coqui TTS 生成語音（優先 XTTS v2，支援情感參考音）
        
        Args:
            text: 要合成的文字
            output_path: 輸出路徑
            speaker_id: 說話者 ID（非 XTTS 時使用）
            emotion: 段落情感（可作為參考，目前用於參考音風格）
            
        Returns:
            生成的音頻文件路徑
        """
        try:
            from TTS.api import TTS
            
            # 優先 XTTS v2：更自然、可接情感參考音（speaker_wav）
            models_to_try = [
                ("tts_models/multilingual/multi-dataset/xtts_v2", "XTTS v2 - 自然、可情感參考"),
                ("tts_models/zh-CN/baker/tacotron2-DDC-GST", "Tacotron2 - 標準中文"),
                ("tts_models/zh-CN/baker/fastspeech2", "FastSpeech2 - 快速生成"),
            ]
            
            tts = None
            used_model = None
            
            ref_wav = self.reference_wav if (self.reference_wav and os.path.exists(self.reference_wav)) else None
            
            for model_name, description in models_to_try:
                try:
                    print(f"🎤 嘗試載入模型: {description}")
                    try:
                        tts = TTS(model_name=model_name, gpu=True)
                    except Exception:
                        tts = TTS(model_name=model_name, gpu=False)
                    try:
                        import torch
                        tts = tts.to("cuda" if torch.cuda.is_available() else "cpu")
                    except Exception:
                        pass
                    used_model = description
                    print(f"✅ 成功載入: {description}")
                    break
                except Exception as e:
                    err = str(e)
                    print(f"⚠️  {description} 載入失敗: {err[:120]}...")
                    if "BeamSearchScorer" in err or "transformers" in err.lower():
                        print("   💡 情感參考音需 XTTS v2。若要用參考音，請執行: pip install \"transformers>=4.30,<4.37\"")
                    continue
            
            if tts is None:
                if ref_wav:
                    print("   💡 情感參考音僅 XTTS v2 支援；目前使用備用模型，音色不會複製參考音。")
                raise RuntimeError("所有 TTS 模型載入失敗")
            
            print(f"🔊 正在生成語音 ({used_model}): {text[:30]}...")
            
            is_xtts = "xtts" in (used_model or "").lower()
            
            if is_xtts:
                # XTTS v2：合成語言由 self.language 決定；參考音可為任一種語言，只取音色與語調
                kwargs = {"text": text, "file_path": output_path, "language": self.language}
                if ref_wav:
                    kwargs["speaker_wav"] = ref_wav
                tts.tts_to_file(**kwargs)
            else:
                tts.tts_to_file(text=text, file_path=output_path, speaker=speaker_id)
            
            self._normalize_and_clean_audio(output_path)
            print(f"✅ 語音已保存: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Coqui TTS 生成失敗: {e}")
            import traceback
            traceback.print_exc()
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
                
                self._normalize_and_clean_audio(output_path)
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
    
    def generate_audio(self, text: str, output_path: str = None, emotion: Optional[str] = None) -> str:
        """
        生成語音（自動選擇可用引擎）
        
        Args:
            text: 要合成的文字
            output_path: 輸出路徑
            emotion: 可選，段落情感（供 XTTS 等未來擴展）
            
        Returns:
            生成的音頻文件路徑
        """
        if output_path is None:
            import time
            timestamp = int(time.time())
            output_path = os.path.join(self.output_dir, f"audio_{timestamp}.wav")
        
        if self.tts_engine == "coqui":
            if not self.check_coqui_available():
                print("⚠️  Coqui TTS 不可用，嘗試使用 Piper...")
                self.tts_engine = "piper"
            
            if self.tts_engine == "coqui":
                return self.generate_audio_coqui(text, output_path, emotion=emotion)
        
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
            emotion = paragraph.get("emotion")
            output_path = os.path.join(self.output_dir, f"audio_{i+1:02d}.wav")
            
            try:
                audio_path = self.generate_audio(text, output_path, emotion=emotion)
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






