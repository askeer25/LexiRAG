#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
from typing import List, Set, Any
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import TextNode
import chromadb
import config

logger = logging.getLogger(__name__)

class VectorStore:
    """负责向量数据库的管理"""
    
    def __init__(self, index_path: str, collection_name: str = "law_collection"):
        self.index_path = index_path
        self.collection_name = collection_name
        os.makedirs(self.index_path, exist_ok=True)
        self.vector_store = self._init_vector_store()
        self.processed_files = self._load_processed_files()
    
    def _init_vector_store(self):
        """初始化向量存储"""
        chroma_client = chromadb.PersistentClient(path=self.index_path)
        chroma_collection = chroma_client.get_or_create_collection(self.collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        return vector_store
    
    def _load_processed_files(self) -> Set[str]:
        """加载已处理过的文件列表"""
        # 使用集合特定的处理文件记录
        processed_file_path = os.path.join(self.index_path, f"{self.collection_name}_processed_files.txt")
        if os.path.exists(processed_file_path):
            with open(processed_file_path, "r", encoding="utf-8") as f:
                files = [line.strip() for line in f.readlines()]
            return set(files)
        else:
            return set()
    
    def save_processed_files(self):
        """保存已处理过的文件列表"""
        # 使用集合特定的处理文件记录
        processed_file_path = os.path.join(self.index_path, f"{self.collection_name}_processed_files.txt")
        with open(processed_file_path, "w", encoding="utf-8") as f:
            for file in self.processed_files:
                f.write(file + "\n")
    
    def add_nodes(self, nodes: List[TextNode]):
        """添加节点到向量存储"""
        self.vector_store.add(nodes)
    
    def mark_as_processed(self, file_path: str):
        """标记文件为已处理"""
        self.processed_files.add(file_path)
    
    def is_processed(self, file_path: str) -> bool:
        """检查文件是否已处理"""
        return file_path in self.processed_files
    
    def get_vector_store(self):
        """获取原始向量存储对象"""
        return self.vector_store