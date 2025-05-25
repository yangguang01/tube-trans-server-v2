from fastapi import APIRouter, Request, Response, HTTPException
from pydantic import BaseModel
import datetime
import firebase_admin
from firebase_admin import auth

# 初始化 Firebase Admin SDK
try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app()

router = APIRouter()

class SessionLoginRequest(BaseModel):
    idToken: str

@router.post("/sessionLogin")
async def session_login(data: SessionLoginRequest, response: Response):
    expires_in = datetime.timedelta(days=5)
    try:
        session_cookie = auth.create_session_cookie(data.idToken, expires_in=expires_in)
    except Exception:
        raise HTTPException(status_code=400, detail="创建会话 Cookie 失败")
    response.set_cookie(
        key="session",
        value=session_cookie,
        #domain=".rxaigc.com",
        domain="localhost",
        secure=False,
        httponly=True,
        samesite="none",
        max_age=int(expires_in.total_seconds())
    )
    return {"status": "success"}

@router.post("/sessionLogout")
async def session_logout(response: Response):
    response.delete_cookie(
        key="session",
        domain=".rxaigc.com",
        secure=True,
        httponly=True,
        samesite="none",
    )
    return {"status": "success"}

@router.get("/me")
async def get_current_user(request: Request):
    session_cookie = request.cookies.get("session")
    if not session_cookie:
        raise HTTPException(status_code=401, detail="未登录或会话已过期")
    try:
        decoded_claims = auth.verify_session_cookie(session_cookie, check_revoked=True)
    except auth.InvalidSessionCookieError:
        raise HTTPException(status_code=401, detail="无效的会话 Cookie")
    except Exception:
        raise HTTPException(status_code=401, detail="验证会话失败")
    return decoded_claims 