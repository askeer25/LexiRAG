import os

# 路径配置
LAWS_PATH = "laws_files"
INDEX_PATH_BASE = "laws_index_"

# 文档处理配置
CHUNK_SIZE = 1024

# 检索配置
DEFAULT_TOP_K = 5

# API配置
DEFAULT_OPENAI_LLM_MODEL = "gpt-4o-2024-11-20"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_OLLAMA_LLM_MODEL = "qwen2.5:3b"
DEFAULT_OLLAMA_EMBEDDING_MODEL = "bge-m3:latest"
DEFAULT_PROVIDER = "openai"

# Ollama配置
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_REQUEST_TIMEOUT = 300

# 支持的文件类型
SUPPORTED_EXTENSIONS = [".pdf", ".md", ".txt"]