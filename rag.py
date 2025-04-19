#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
from typing import List, Dict, Any
import dotenv

from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import NodeWithScore

from document_loader import DocumentLoader
from document_processor import DocumentProcessor
from embedding_service import EmbeddingService
from vector_store import VectorStore
from retriever import SimpleRetriever
from response_generator import ResponseGenerator, QueryService
import config

# 加载环境变量
dotenv.load_dotenv()

logger = logging.getLogger(__name__)

class LawRAG:
    """法律RAG系统，专门针对法律文档的检索和问答"""
    
    def __init__(
        self,
        laws_path: str,
        index_path: str,
        llm_model_name: str,
        embedding_model_name: str,
        provider: str,
    ) -> None:
        # 初始化路径
        self.laws_path = laws_path
        self.index_path = index_path
        self.provider = provider  # 添加provider属性保存
        os.makedirs(self.laws_path, exist_ok=True)
        os.makedirs(self.index_path, exist_ok=True)
        
        # 初始化模型
        if provider == "openai":
            llm_api_key = os.getenv("OPENAI_API_KEY")
            llm_api_base = os.getenv("OPENAI_BASE_URL")
            embed_api_key = llm_api_key
            embed_api_base = llm_api_base
            llm_client = OpenAI(
                model=llm_model_name,
                api_key=llm_api_key,
                api_base=llm_api_base,
                max_tokens=4096,
            )

            embedding_model = OpenAIEmbedding(
                model=embedding_model_name,
                api_key=embed_api_key,
                api_base=embed_api_base,
            )
        elif provider == "ollama":
            llm_client = Ollama(
                model=llm_model_name,
                base_url=config.OLLAMA_BASE_URL, 
                request_timeout=config.OLLAMA_REQUEST_TIMEOUT,
            )
            embedding_model = OllamaEmbedding(
                model_name=embedding_model_name,
                base_url=config.OLLAMA_BASE_URL,
            )
        else:
            raise ValueError(f"不支持的提供商: {provider}")
        
        # 初始化各组件
        self.document_loader = DocumentLoader()
        self.document_processor = DocumentProcessor(chunk_size=config.CHUNK_SIZE)
        self.embedding_service = EmbeddingService(embedding_model)
        
        # 使用provider特定的集合名称
        collection_name = f"law_collection_{provider}"
        self.vector_store = VectorStore(index_path=self.index_path, collection_name=collection_name)
        self.response_generator = ResponseGenerator(llm_client)
    
    def process_file(self, file_path: str) -> bool:
        """处理单个文件"""
        if self.vector_store.is_processed(file_path):
            logger.info(f"文件已处理过，跳过: {file_path}")
            return False
            
        try:
            # 加载文档
            documents = self.document_loader.load_document(file_path)
            if not documents:
                return False
                
            # 处理文档，基于法律结构进行切分
            nodes = self.document_processor.process_documents(documents)
            
            # 向量化
            nodes = self.embedding_service.embed_nodes(nodes)
            
            # 存储向量
            self.vector_store.add_nodes(nodes)
            self.vector_store.mark_as_processed(file_path)
            logger.info(f"已处理文件: {file_path}")
            return True
        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {e}")
            return False
    
    def init_laws_library(self) -> List[str]:
        """初始化法律文档库，处理所有文件"""
        file_list = []
        for filepath, _, filenames in os.walk(self.laws_path):
            for filename in filenames:
                if any(filename.lower().endswith(ext) for ext in config.SUPPORTED_EXTENSIONS):
                    file_path = os.path.join(filepath, filename)
                    if not self.vector_store.is_processed(file_path):
                        file_list.append(file_path)
        
        processed_count = 0
        processed_files = []
        for file_path in file_list:
            if self.process_file(file_path):
                processed_count += 1
                processed_files.append(file_path)
        
        self.vector_store.save_processed_files()
        logger.info(f"初始化时处理了 {processed_count} 个文件")
        return processed_files
    
    def retrieve(self, query_str: str, top_k: int = config.DEFAULT_TOP_K) -> List[NodeWithScore]:
        """检索相关内容，使用简单检索器"""
        retriever = SimpleRetriever(
            self.vector_store.get_vector_store(),
            self.embedding_service,
            similarity_top_k=top_k,
        )
        retrieved_nodes = retriever.retrieve(query_str)
        logger.info(f"为查询检索到 {len(retrieved_nodes)} 个节点")
        return retrieved_nodes
    
    def generate_response(self, query_str: str, top_k: int = config.DEFAULT_TOP_K) -> str:
        """生成回答"""
        # 检索阶段
        retrieved_nodes = self.retrieve(query_str, top_k)
        
        # 生成阶段
        response = self.response_generator.generate_response(query_str, retrieved_nodes)
        return response


if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 测试OpenAI API接入
    print("\n=== 测试OpenAI API接入 ===")
    # 使用环境变量或默认值
    openai_llm_model = os.getenv("LLM_MODEL", config.DEFAULT_OPENAI_LLM_MODEL)
    openai_embedding_model = os.getenv("EMBEDDING_MODEL", config.DEFAULT_OPENAI_EMBEDDING_MODEL)
    provider = "openai"

    # 创建OpenAI法律RAG系统
    openai_law_rag = LawRAG(
        laws_path=config.LAWS_PATH,  # 法律文件目录
        index_path=f"{config.INDEX_PATH_BASE}{provider}",  # 向量索引目录
        llm_model_name=openai_llm_model,
        embedding_model_name=openai_embedding_model,
        provider=provider,
    )
    
    # 初始化法律文档库
    processed_files = openai_law_rag.init_laws_library()
    print(f"OpenAI模式: 已处理 {len(processed_files)} 个法律文档文件")

    # 测试查询
    print("\n--- OpenAI测试查询: 土地征收补偿 ---")
    openai_response = openai_law_rag.generate_response("土地征收时，农民应该获得哪些补偿？", top_k=5)
    print(openai_response)
    
    # 测试Ollama本地模型接入
    print("\n=== 测试Ollama本地模型接入 ===")
    # 使用本地Ollama模型
    ollama_llm_model = config.DEFAULT_OLLAMA_LLM_MODEL
    ollama_embedding_model = config.DEFAULT_OLLAMA_EMBEDDING_MODEL
    provider = "ollama"

    # 创建Ollama法律RAG系统
    ollama_law_rag = LawRAG(
        laws_path=config.LAWS_PATH,  # 法律文件目录
        index_path=f"{config.INDEX_PATH_BASE}{provider}",  # 向量索引目录
        llm_model_name=ollama_llm_model,
        embedding_model_name=ollama_embedding_model,
        provider=provider,
    )
    
    # 初始化法律文档库
    processed_files = ollama_law_rag.init_laws_library()
    print(f"Ollama模式: 已处理 {len(processed_files)} 个法律文档文件")

    # 测试查询
    print("\n--- Ollama测试查询: 土地征收补偿 ---")
    ollama_response = ollama_law_rag.generate_response("土地征收时，农民应该获得哪些补偿？", top_k=5)
    print(ollama_response)


