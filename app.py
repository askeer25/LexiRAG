import os
import logging
import dotenv
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from rag import LawRAG, QueryService

# 加载环境变量
dotenv.load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("law_rag_app")

# 创建FastAPI应用
app = FastAPI(
    title="法律RAG系统",
    description="基于LlamaIndex的法律条文检索与问答系统",
    version="1.0.0",
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源，生产环境应限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局RAG系统
law_rag = None
query_service = None

# Pydantic模型
class Query(BaseModel):
    query: str
    top_k: Optional[int] = 5

class Source(BaseModel):
    text: str
    metadata: Dict[str, str]

class ResponseModel(BaseModel):
    answer: str
    sources: List[Source]

class InitStatusModel(BaseModel):
    status: str
    processed_files: List[str]
    total_files: int

class ProviderModel(BaseModel):
    provider: str

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化RAG系统"""
    global law_rag, query_service
    
    # 使用环境变量或默认值
    llm_model_name = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    provider = os.getenv("MODEL_PROVIDER", "openai")  # 默认使用OpenAI
    
    # 创建法律RAG系统
    law_rag = LawRAG(
        laws_path="laws_files",  # 法律文件目录
        index_path=f"laws_index_{provider}",  # 向量索引目录
        llm_model_name=llm_model_name,
        embedding_model_name=embedding_model_name,
        provider=provider,
    )
    
    # 创建查询服务
    query_service = QueryService(law_rag)
    
    logger.info("RAG系统已初始化")

@app.get("/api/status")
async def check_status():
    """检查系统初始化状态"""
    global law_rag
    
    if law_rag is None:
        raise HTTPException(status_code=500, detail="RAG系统未初始化")
    
    try:
        # 检查是否有处理过的文件
        processed_files = law_rag.vector_store.processed_files
        
        # 检查lawsfiles目录下的文件总数
        total_files = []
        supported_extensions = [".pdf", ".md", ".txt"]
        for filepath, _, filenames in os.walk(law_rag.laws_path):
            for filename in filenames:
                if any(filename.lower().endswith(ext) for ext in supported_extensions):
                    total_files.append(os.path.join(filepath, filename))
        
        # 如果没有文件需要处理，或者所有文件都已处理，则认为系统已初始化
        is_initialized = len(processed_files) > 0 and len(processed_files) >= len(total_files)
        
        return {
            "initialized": is_initialized,
            "processed_files_count": len(processed_files),
            "total_files_count": len(total_files),
        }
    except Exception as e:
        logger.error(f"检查系统状态时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"检查系统状态时出错: {str(e)}")

@app.get("/api")
async def root():
    """健康检查接口"""
    return {"status": "ok", "message": "法律RAG系统正在运行"}

@app.get("/api/init", response_model=InitStatusModel)
async def initialize_system():
    """初始化或重新初始化RAG系统"""
    global law_rag
    
    if law_rag is None:
        raise HTTPException(status_code=500, detail="RAG系统未初始化")
    
    # 获取lawsfiles目录下的所有文件
    total_files = []
    supported_extensions = [".pdf", ".md", ".txt"]
    for filepath, _, filenames in os.walk(law_rag.laws_path):
        for filename in filenames:
            if any(filename.lower().endswith(ext) for ext in supported_extensions):
                total_files.append(os.path.join(filepath, filename))
    
    # 初始化法律文档库
    processed_files = law_rag.init_laws_library()
    
    logger.info(f"已处理 {len(processed_files)} 个法律文档文件")
    
    return {
        "status": "success",
        "processed_files": processed_files,
        "total_files": len(total_files),
    }

@app.post("/api/query", response_model=ResponseModel)
async def query(query_data: Query = Body(...)):
    """处理用户查询"""
    global query_service
    
    if query_service is None:
        raise HTTPException(status_code=500, detail="查询服务未初始化")
    
    try:
        response = await query_service.process_query(query_data.query, query_data.top_k)
        return response
    except Exception as e:
        logger.error(f"处理查询时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理查询时出错: {str(e)}")

@app.get("/api/laws")
async def get_laws():
    """获取已处理的法律列表"""
    global law_rag
    
    if law_rag is None:
        raise HTTPException(status_code=500, detail="RAG系统未初始化")
    
    try:
        # 从处理过的文件中提取法律名称
        law_names = set()
        for file_path in law_rag.vector_store.processed_files:
            if os.path.isfile(file_path):
                law_name = os.path.basename(file_path).split('.')[0]
                law_names.add(law_name)
        
        return {"laws": sorted(list(law_names))}
    except Exception as e:
        logger.error(f"获取法律列表时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取法律列表时出错: {str(e)}")

@app.post("/api/set_provider")
async def set_provider(data: ProviderModel):
    """更改模型提供者（OpenAI或Ollama）"""
    global law_rag, query_service
    
    if data.provider not in ["openai", "ollama"]:
        raise HTTPException(status_code=400, detail="提供者必须是'openai'或'ollama'")
    
    try:
        # 使用环境变量或默认值
        llm_model_name = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        
        # 为 Ollama 使用不同的默认模型
        if data.provider == "ollama":
            llm_model_name = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:3b")
            embedding_model_name = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3")
        
        # 创建法律RAG系统
        law_rag = LawRAG(
            laws_path="laws_files",  # 法律文件目录
            index_path=f"laws_index_{data.provider}",  # 向量索引目录
            llm_model_name=llm_model_name,
            embedding_model_name=embedding_model_name,
            provider=data.provider,
        )
        
        # 创建查询服务
        query_service = QueryService(law_rag)
        
        # 更新环境变量
        os.environ["MODEL_PROVIDER"] = data.provider
        
        logger.info(f"切换到模型提供者: {data.provider}")
        
        return {"status": "success", "provider": data.provider}
    except Exception as e:
        logger.error(f"切换模型提供者时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"切换模型提供者时出错: {str(e)}")

@app.get("/api/get_provider")
async def get_provider():
    """获取当前使用的模型提供者"""
    provider = os.getenv("MODEL_PROVIDER", "openai")
    return {"provider": provider}

# 添加静态文件服务
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)