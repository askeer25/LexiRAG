#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
from typing import List, Any
from llama_index.core.schema import TextNode
import config

logger = logging.getLogger(__name__)

class EmbeddingService:
    """负责文本向量化"""
    
    def __init__(self, embedding_model):
        self.model = embedding_model
    
    def embed_nodes(self, nodes: List[TextNode], batch_size: int = 50) -> List[TextNode]:
        """对节点进行向量化"""
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i : i + batch_size]
            try:
                for node in batch:
                    node_embedding = self.model.get_text_embedding(
                        node.get_content(metadata_mode="all")
                    )
                    node.embedding = node_embedding
                logger.info(
                    f"处理嵌入向量批次 {i//batch_size + 1}/{(len(nodes)-1)//batch_size + 1}"
                )
            except Exception as e:
                logger.error(
                    f"为批次 {i//batch_size + 1} 创建嵌入向量时出错: {e}"
                )
        return nodes
    
    def get_query_embedding(self, query: str) -> List[float]:
        """获取查询的嵌入向量"""
        return self.model.get_query_embedding(query)