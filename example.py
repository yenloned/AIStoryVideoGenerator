"""
快速開始範例
演示如何使用 VideoPipeline 生成影片
"""

from main import VideoPipeline

if __name__ == "__main__":
    # 範例 1: 基本用法
    print("範例 1: 生成成語故事影片")
    pipeline1 = VideoPipeline(
        keyword="成語故事：守株待兔",
        style="chinese_ink",
        tts_engine="coqui",
        image_model="sd15"
    )
    # pipeline1.run()  # 取消註釋以執行
    
    # 範例 2: 歷史典故
    print("\n範例 2: 生成歷史典故影片")
    pipeline2 = VideoPipeline(
        keyword="歷史典故：三顧茅廬",
        style="ancient",
        tts_engine="coqui",
        image_model="sd15"
    )
    # pipeline2.run()  # 取消註釋以執行
    
    # 範例 3: 冷知識
    print("\n範例 3: 生成冷知識影片")
    pipeline3 = VideoPipeline(
        keyword="冷知識：為什麼天空是藍色的",
        style="cinematic",
        tts_engine="coqui",
        image_model="sd15"
    )
    # pipeline3.run()  # 取消註釋以執行
    
    print("\n💡 提示: 取消註釋 pipeline.run() 以執行生成")

