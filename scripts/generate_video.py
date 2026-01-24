"""
影片生成模組 - 使用 FFmpeg
合成圖片、語音、字幕為完整影片
"""

import os
import sys
import json
import subprocess
from typing import List, Dict
import tempfile


class VideoGenerator:
    def __init__(
        self,
        output_dir: str = "video",
        width: int = 1080,
        height: int = 1920,
        fps: int = 30
    ):
        """
        初始化影片生成器
        
        Args:
            output_dir: 輸出目錄
            width: 影片寬度
            height: 影片高度
            fps: 幀率
        """
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.fps = fps
        
        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🎬 影片生成器初始化，解析度: {width}x{height}")
    
    def check_ffmpeg(self) -> bool:
        """檢查 FFmpeg 是否可用"""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def get_audio_duration(self, audio_path: str) -> float:
        """獲取音頻時長（秒）"""
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    audio_path
                ],
                capture_output=True,
                check=True,
                timeout=10
            )
            return float(result.stdout.decode().strip())
        except Exception as e:
            print(f"⚠️  無法獲取音頻時長，使用預設值: {e}")
            return 3.0  # 預設 3 秒
    
    def create_subtitle_file(self, script_data: Dict, audio_paths: List[str], output_path: str) -> str:
        """
        創建字幕文件（SRT 格式），使用實際音頻時長
        
        Args:
            script_data: 劇本數據
            audio_paths: 音頻文件路徑列表（用於獲取實際時長）
            output_path: 輸出路徑
            
        Returns:
            字幕文件路徑
        """
        paragraphs = script_data.get("paragraphs", [])
        
        # 使用實際音頻時長
        current_time = 0.0
        subtitle_lines = []
        
        # 格式化時間
        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 60 - secs) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
        
        for i, paragraph in enumerate(paragraphs):
            text = paragraph.get("text", "")
            
            # 使用實際音頻時長（如果可用）
            if i < len(audio_paths) and os.path.exists(audio_paths[i]):
                duration = self.get_audio_duration(audio_paths[i])
            else:
                # 備用：根據文字長度估算
                duration = max(2.0, min(len(text) * 0.1, 6.0))
            
            start_time = current_time
            end_time = current_time + duration
            
            subtitle_lines.append(f"{i+1}")
            subtitle_lines.append(f"{format_time(start_time)} --> {format_time(end_time)}")
            subtitle_lines.append(text)
            subtitle_lines.append("")
            
            current_time = end_time
        
        # 寫入文件
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(subtitle_lines))
        
        print(f"✅ 字幕文件已創建: {output_path}（使用實際音頻時長）")
        return output_path
    
    def create_video_segment(
        self,
        image_path: str,
        audio_path: str,
        duration: float,
        output_path: str,
        effect: str = "zoom"
    ) -> str:
        """
        創建單個影片片段（帶效果）
        
        Args:
            image_path: 圖片路徑
            audio_path: 音頻路徑
            duration: 片段時長
            output_path: 輸出路徑
            effect: 效果類型 (zoom, shake, pan)
            
        Returns:
            生成的片段路徑
        """
        try:
            # 根據效果類型生成 FFmpeg 濾鏡
            # 使用 letterboxing（黑邊）保持原始比例，不拉伸
            # scale 保持比例，pad 添加黑邊
            # 確保所有片段都使用相同的寬高比處理
            base_scale = f"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease"
            base_pad = f"pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2:color=black"
            
            if effect == "zoom":
                # 縮放效果（帶 letterboxing）
                vf = f"{base_scale},{base_pad},zoompan=z='min(zoom+0.0015,1.5)':d={int(duration * self.fps)}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            elif effect == "shake":
                # 震動效果（帶 letterboxing）
                vf = f"{base_scale},{base_pad},crop=in_w:in_h:random(1)*10:random(1)*10"
            elif effect == "pan":
                # 平移效果（帶 letterboxing）
                vf = f"scale={int(self.width*1.2)}:{self.height}:force_original_aspect_ratio=decrease,pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2:color=black,crop={self.width}:{self.height}:'(t*20)':0"
            else:
                # 無效果（帶 letterboxing）
                vf = f"{base_scale},{base_pad}"
            
            # 生成片段
            # 使用 -shortest 確保視頻長度匹配音頻長度（不會提前結束）
            # 使用 -t 作為時長限制
            subprocess.run(
                [
                    "ffmpeg",
                    "-loop", "1",
                    "-i", image_path,
                    "-i", audio_path,
                    "-vf", vf,
                    "-t", str(duration),  # 設置時長
                    "-shortest",  # 確保不超過音頻長度，匹配音頻結束
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "23",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-pix_fmt", "yuv420p",
                    "-r", str(self.fps),
                    "-s", f"{self.width}x{self.height}",  # 明確指定輸出尺寸，確保一致性
                    "-map", "0:v:0",  # 明確映射視頻流
                    "-map", "1:a:0",  # 明確映射音頻流
                    "-y",
                    output_path
                ],
                check=True,
                capture_output=True
            )
            
            # 驗證生成的片段時長是否匹配音頻
            try:
                generated_duration = self.get_audio_duration(output_path)
                audio_duration = self.get_audio_duration(audio_path)
                if abs(generated_duration - audio_duration) > 0.3:  # 允許 0.3 秒誤差
                    print(f"⚠️  時長不匹配: 音頻 {audio_duration:.2f}秒，視頻 {generated_duration:.2f}秒")
            except:
                pass  # 驗證失敗不影響流程
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 片段生成失敗: {e.stderr.decode()}")
            raise
    
    def generate_video(
        self,
        script_data: Dict,
        image_paths: List[str],
        audio_paths: List[str],
        output_path: str = None,
        style: str = "zoom"
    ) -> str:
        """
        生成完整影片
        
        Args:
            script_data: 劇本數據
            image_paths: 圖片路徑列表
            audio_paths: 音頻路徑列表
            output_path: 輸出路徑
            style: 效果風格
            
        Returns:
            生成的影片路徑
        """
        if not self.check_ffmpeg():
            raise RuntimeError("FFmpeg 不可用，請安裝 FFmpeg")
        
        if output_path is None:
            import time
            timestamp = int(time.time())
            output_path = os.path.join(self.output_dir, f"video_{timestamp}.mp4")
        
        # 確保圖片和音頻數量匹配
        # 檢查是否有缺失的音頻文件
        print(f"📊 檢查文件匹配: {len(image_paths)} 張圖片, {len(audio_paths)} 個音頻")
        
        # 確保所有音頻文件都存在
        valid_audio_paths = []
        valid_image_paths = []
        for i, audio_path in enumerate(audio_paths):
            if os.path.exists(audio_path):
                valid_audio_paths.append(audio_path)
                if i < len(image_paths):
                    valid_image_paths.append(image_paths[i])
            else:
                print(f"⚠️  音頻文件不存在: {audio_path}")
        
        min_count = min(len(valid_image_paths), len(valid_audio_paths))
        if min_count == 0:
            raise ValueError("沒有可用的圖片或音頻文件")
        
        print(f"🎬 開始生成影片，共 {min_count} 個片段...")
        
        # 生成每個片段
        segment_paths = []
        effects = ["zoom", "shake", "pan"]  # 輪流使用效果
        
        for i in range(min_count):
            image_path = valid_image_paths[i]
            audio_path = valid_audio_paths[i]
            
            # 驗證圖片文件存在
            if not os.path.exists(image_path):
                print(f"⚠️  圖片文件不存在: {image_path}")
                continue
            
            # 獲取實際音頻時長（確保精確匹配）
            duration = self.get_audio_duration(audio_path)
            print(f"🎵 片段 {i+1}: 音頻時長 {duration:.2f} 秒, 圖片: {os.path.basename(image_path)}")
            
            effect = effects[i % len(effects)] if style == "mixed" else style
            
            segment_path = os.path.join(self.output_dir, f"segment_{i+1:02d}.mp4")
            
            try:
                print(f"📹 生成片段 {i+1}/{min_count}（時長: {duration:.2f}秒）...")
                # 確保所有片段都使用相同的寬高比處理
                self.create_video_segment(
                    image_path, audio_path, duration,
                    segment_path, effect
                )
                # 驗證生成的片段
                if os.path.exists(segment_path):
                    segment_paths.append(segment_path)
                    print(f"✅ 片段 {i+1} 生成成功")
                else:
                    print(f"⚠️  片段 {i+1} 文件未生成: {segment_path}")
            except Exception as e:
                print(f"⚠️  片段 {i+1} 生成失敗: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not segment_paths:
            raise RuntimeError("沒有成功生成的片段")
        
        # 合併所有片段
        print("🔗 正在合併片段...")
        self._concat_segments(segment_paths, output_path)
        
        # 添加字幕（使用實際音頻時長）
        print("📝 正在添加字幕...")
        subtitle_path = os.path.join(self.output_dir, "subtitles.srt")
        # 使用有效的音頻路徑列表
        self.create_subtitle_file(script_data, valid_audio_paths[:min_count], subtitle_path)
        final_output = self._add_subtitles(output_path, subtitle_path)
        
        # 清理臨時文件
        for seg_path in segment_paths:
            if os.path.exists(seg_path):
                os.unlink(seg_path)
        
        print(f"✅ 影片生成完成: {final_output}")
        return final_output
    
    def _concat_segments(self, segment_paths: List[str], output_path: str):
        """合併影片片段，確保保持正確的寬高比"""
        # 創建文件列表
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            for path in segment_paths:
                if os.path.exists(path):
                    # Windows 路徑處理
                    abs_path = os.path.abspath(path).replace("\\", "/")
                    f.write(f"file '{abs_path}'\n")
                else:
                    print(f"⚠️  片段文件不存在: {path}")
            list_file = f.name
        
        try:
            # 使用 filter_complex 確保所有片段保持一致的寬高比
            # 重新編碼以確保所有片段都有相同的解析度和格式
            subprocess.run(
                [
                    "ffmpeg",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", list_file,
                    "-vf", f"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2:color=black",
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "23",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-pix_fmt", "yuv420p",
                    "-r", str(self.fps),
                    "-y",
                    output_path
                ],
                check=True,
                capture_output=True
            )
            print(f"✅ 片段合併完成: {len(segment_paths)} 個片段")
        finally:
            if os.path.exists(list_file):
                os.unlink(list_file)
    
    def _add_subtitles(self, video_path: str, subtitle_path: str) -> str:
        """添加字幕到影片，確保保持正確的寬高比"""
        output_path = video_path.replace(".mp4", "_with_subtitles.mp4")
        
        try:
            # Windows 路徑處理
            import platform
            if platform.system() == "Windows":
                # 轉換路徑為 FFmpeg 可用的格式
                subtitle_path_escaped = subtitle_path.replace("\\", "/").replace(":", "\\:")
            else:
                subtitle_path_escaped = subtitle_path
            
            # 確保字幕添加時也保持寬高比
            # 使用 scale 和 pad 確保輸出尺寸正確
            # 改進字幕樣式：清晰易讀
            subtitle_style = (
                "FontName=Microsoft YaHei,"
                "FontSize=12,"  # 字體大小
                "PrimaryColour=&Hffffff,"  # 白色文字
                "OutlineColour=&H000000,"  # 黑色描邊
                "Outline=2,"  # 描邊寬度
                "Shadow=1,"  # 陰影
                "MarginV=30,"  # 底部邊距
                "Bold=0"  # 不使用粗體
            )
            
            subprocess.run(
                [
                    "ffmpeg",
                    "-i", video_path,
                    "-vf", f"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2:color=black,subtitles={subtitle_path_escaped}:force_style='{subtitle_style}'",
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "23",
                    "-c:a", "copy",
                    "-pix_fmt", "yuv420p",
                    "-y",
                    output_path
                ],
                check=True,
                capture_output=True
            )
            return output_path
        except subprocess.CalledProcessError as e:
            # 如果字幕添加失敗，返回原影片
            print(f"⚠️  字幕添加失敗: {e.stderr.decode() if e.stderr else 'Unknown error'}")
            print("⚠️  返回原影片（無字幕）")
            return video_path


def main():
    """測試用主函數"""
    if len(sys.argv) < 4:
        print("用法: python generate_video.py <script.json> <images_dir> <audio_dir> [output.mp4]")
        sys.exit(1)
    
    script_file = sys.argv[1]
    images_dir = sys.argv[2]
    audio_dir = sys.argv[3]
    output = sys.argv[4] if len(sys.argv) > 4 else None
    
    with open(script_file, "r", encoding="utf-8") as f:
        script_data = json.load(f)
    
    # 獲取圖片和音頻文件列表
    import glob
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    audio_paths = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
    
    generator = VideoGenerator()
    
    try:
        video_path = generator.generate_video(script_data, image_paths, audio_paths, output)
        print(f"\n生成的影片: {video_path}")
    except Exception as e:
        print(f"錯誤: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

