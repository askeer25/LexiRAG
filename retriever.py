#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from typing import List, Any
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import VectorStoreQuery
import config

logger = logging.getLogger(__name__)

class SimpleRetriever:
    """简单检索器，通过向量相似度直接检索文档"""
    
    def __init__(
        self,
        vector_store: Any,
        embed_model: Any,
        similarity_top_k: int = config.DEFAULT_TOP_K,
    ) -> None:
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._similarity_top_k = similarity_top_k
        
    def retrieve(self, query_str: str) -> List[NodeWithScore]:
        """根据查询检索相关文档"""
        try:
            # 获取查询的嵌入向量
            query_embedding = self._embed_model.get_query_embedding(query_str)
            
            # 构建向量查询
            vector_query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=self._similarity_top_k,
            )
            
            # 执行查询
            results = self._vector_store.query(vector_query)
            
            # 构建结果列表
            nodes_with_scores = []
            for i, node in enumerate(results.nodes):
                score = None
                if results.similarities is not None and i < len(results.similarities):
                    score = results.similarities[i]
                nodes_with_scores.append(NodeWithScore(node=node, score=score))
            
            # 按相关性排序
            nodes_with_scores.sort(key=lambda x: x.score if x.score is not None else 0, reverse=True)
            
            logger.info(f"检索到 {len(nodes_with_scores)} 个相关节点")
            return nodes_with_scores[:self._similarity_top_k]
        except Exception as e:
            logger.error(f"检索过程中出错: {str(e)}")
            return []