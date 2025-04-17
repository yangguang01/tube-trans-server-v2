from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import FileResponse
from pathlib import Path

from app.models.schemas import TranslationRequest, TaskResponse, TaskStatus
from app.services.processor import create_translation_task, get_task_status
from app.core.config import SUBTITLES_DIR

from app.core.logging import logger


router = APIRouter()


@router.post("/translate", response_model=TaskResponse)
async def translate_video(request: TranslationRequest):
    """
    创建新的视频字幕翻译任务
    
    参数:
        request (TranslationRequest): 包含YouTube URL和翻译选项的请求体
        
    返回:
        TaskResponse: 包含任务ID和状态的响应体
    """
    task_id = create_translation_task(
        youtube_url=str(request.youtube_url),
        custom_prompt=request.custom_prompt,
        special_terms=request.special_terms,
        content_name=request.content_name,
        language=request.language,
        model=request.model
    )
    
    return {"task_id": task_id, "status": "pending"}


@router.get("/tasks/{task_id}", response_model=TaskStatus)
async def get_task(task_id: str):
    """
    获取任务状态
    
    参数:
        task_id (str): 任务ID
        
    返回:
        TaskStatus: 任务状态信息
    """
    task_info = get_task_status(task_id)
    if task_info is None:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return TaskStatus(**task_info)


@router.get("/subtitles/{task_id}")
async def get_subtitle_file(task_id: str):
    """
    获取任务生成的字幕文件
    
    参数:
        task_id (str): 任务ID
        
    返回:
        FileResponse: 字幕文件
    """
    task_info = get_task_status(task_id)
    if task_info is None:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task_info["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not yet completed")
    
    # # 查找字幕文件
    # video_title = task_info.get("video_title", task_id)
    # # 清理文件名
    # import re
    # safe_title = re.sub(r'[^\w\s-]', '', video_title)
    # safe_title = re.sub(r'\s+', '-', safe_title)

    # 使用视频id查找字幕文件
    video_id = task_info.get("video_id", task_id)
    subtitle_file = SUBTITLES_DIR / f"{task_id}.srt"
    logger.info(f"字幕文件路径: {subtitle_file}")
    
    
    if not subtitle_file.exists():
        raise HTTPException(status_code=404, detail="Subtitle file not found")
    
    return FileResponse(
        path=subtitle_file,
        filename=f"{video_id}.srt",
        media_type="application/x-subrip"
    )


@router.get("/health")
async def health_check():
    """
    健康检查端点
    
    返回:
        dict: 状态信息
    """
    return {"status": "ok", "version": "0.1.0"} 