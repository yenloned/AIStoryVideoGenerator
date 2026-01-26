"""
批次生成腳本
一次生成多個影片
"""

import sys
from main import VideoPipeline

# 預定義的關鍵字列表
KEYWORDS = [
    "成語故事：守株待兔",
    "成語故事：畫蛇添足",
    "歷史典故：三顧茅廬",
    "冷知識：為什麼天空是藍色的",
    "都市傳說：電梯遊戲",
]

# 風格配置
STYLES = {
    "成語故事": "chinese_ink",
    "歷史典故": "ancient",
    "冷知識": "cinematic",
    "都市傳說": "horror",
}


def batch_generate(keywords=None, style=None, tts_engine="coqui", image_model="sd15"):
    """
    批次生成影片
    
    Args:
        keywords: 關鍵字列表，如果為 None 則使用預定義列表
        style: 統一樣式，如果為 None 則根據關鍵字自動選擇
        tts_engine: TTS 引擎
        image_model: 圖片模型
    """
    if keywords is None:
        keywords = KEYWORDS
    
    print(f"🚀 開始批次生成，共 {len(keywords)} 個影片\n")
    
    results = []
    
    for i, keyword in enumerate(keywords, 1):
        print(f"\n{'='*60}")
        print(f"進度: {i}/{len(keywords)}")
        print(f"關鍵字: {keyword}")
        print(f"{'='*60}\n")
        
        # 自動選擇風格
        if style is None:
            auto_style = "cinematic"
            for key, val in STYLES.items():
                if key in keyword:
                    auto_style = val
                    break
        else:
            auto_style = style
        
        try:
            pipeline = VideoPipeline(
                keyword=keyword,
                style=auto_style,
                tts_engine=tts_engine,
                image_model=image_model
            )
            
            video_path = pipeline.run()
            results.append({
                "keyword": keyword,
                "status": "success",
                "path": video_path
            })
            
            print(f"✅ 完成: {keyword}")
            
        except Exception as e:
            print(f"❌ 失敗: {keyword} - {e}")
            results.append({
                "keyword": keyword,
                "status": "failed",
                "error": str(e)
            })
    
    # 輸出總結
    print(f"\n{'='*60}")
    print("批次生成完成")
    print(f"{'='*60}")
    print(f"成功: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"失敗: {sum(1 for r in results if r['status'] == 'failed')}")
    print("\n詳細結果:")
    for result in results:
        if result['status'] == 'success':
            print(f"  ✅ {result['keyword']}: {result['path']}")
        else:
            print(f"  ❌ {result['keyword']}: {result['error']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="批次生成影片")
    parser.add_argument(
        "--keywords",
        type=str,
        nargs="+",
        help="關鍵字列表（用空格分隔）"
    )
    parser.add_argument(
        "--style",
        type=str,
        choices=["cinematic", "chinese_ink", "ancient", "fantasy", "horror", "hand_drawn"],
        help="統一樣式（覆蓋自動選擇）"
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
        help="圖片模型"
    )
    
    args = parser.parse_args()
    
    batch_generate(
        keywords=args.keywords,
        style=args.style,
        tts_engine=args.tts,
        image_model=args.image_model
    )








