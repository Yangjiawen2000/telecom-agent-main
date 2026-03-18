from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import chat
from app.config import settings

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化 Milvus 集合
    from app.api.chat import ltm
    try:
        await ltm.init_collections()
    except Exception as e:
        print(f"Failed to initialize Milvus: {e}")
    yield

app = FastAPI(
    title=settings.APP_NAME,
    description="通信运营商多智能体客服系统",
    version="0.1.0",
    lifespan=lifespan
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中建议使用具体的域名，或通过环境变量配置
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api", tags=["chat"])

@app.get("/api")
async def root():
    return {"message": "Welcome to Telecom Agent API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
