from fastapi import APIRouter
from app.api.endpoints import router as endpoints_router

# 创建API路由器
api_router = APIRouter()

# 添加各个端点
api_router.include_router(endpoints_router, prefix="") 