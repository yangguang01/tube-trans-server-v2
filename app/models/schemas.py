from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, Any


class TranslationRequest(BaseModel):
    """翻译请求模型"""
    youtube_url: HttpUrl
    custom_prompt: Optional[str] = ""
    special_terms: Optional[str] = ""
    content_name: Optional[str] = ""
    language: str = "zh-CN"


class TaskResponse(BaseModel):
    """任务响应模型"""
    task_id: str
    status: str


class TaskStatus(BaseModel):
    """任务状态模型"""
    status: str
    progress: Optional[float] = None
    result_url: Optional[str] = None
    error: Optional[str] = None
    video_title: Optional[str] = None 