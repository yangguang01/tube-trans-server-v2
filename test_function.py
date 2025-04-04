import asyncio
import os
import sys

# 添加项目根目录到Python路径，确保可以导入app模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.config import SUBTITLES_DIR, TRANSCRIPTS_DIR, TMP_DIR, AUDIO_DIR


# 导入要测试的函数
from app.services.translation import split_sentence, format_time, get_video_info, download_audio_webm, get_video_info_and_download


# 运行测试
if __name__ == "__main__":
    task_id = '532b68d5-03e2-4c19-b92c-a7bccea9b2d8'
    url = 'https://www.youtube.com/watch?v=tDmjz6HB-yw'
    file_path = AUDIO_DIR / f"{task_id}.webm"
    video_info = get_video_info_and_download(url, file_path)
    print(video_info)
