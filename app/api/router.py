from fastapi import APIRouter
from app.api.endpoints import router as endpoints_router
from app.api.auth import router as auth_router

# 创建API路由器
api_router = APIRouter()

# 添加各个端点
api_router.include_router(endpoints_router, prefix="")
api_router.include_router(auth_router, prefix="") 