#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from typing import List
from llama_index.core import PromptTemplate
from llama_index.core.schema import NodeWithScore
import config

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """负责生成最终回答"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
        self.law_prompt = PromptTemplate(
            """\
你是一位经验丰富的法律顾问，擅长解释和应用中国法律。

用户问题: {query_str}

以下是相关的法律条文:
------------
{context_str}
------------

请基于上述法律条文回答用户的问题。你的回答应该:
1. 引用具体的法律条款（包括法律名称、章节和条款号）
2. 清晰解释法律条款的含义和适用情况
3. 针对用户问题提供准确的法律建议
4. 使用简明易懂的语言，避免过多专业术语
5. 如有必要，解释法律条款背后的立法原意

如果提供的法律条文不足以完整回答问题，请明确指出信息的局限性，并基于已有信息提供最佳回答。

回答必须使用中文，并保持客观、准确的法律态度。
"""
        )
    
    def generate_response(self, query_str: str, context_nodes: List[NodeWithScore]) -> str:
        """根据查询和上下文生成回答"""
        if not context_nodes:
            logger.warning("未找到与查询相关的节点")
            return "未找到相关法律条文。请尝试重新表述您的问题，或查询其他法律领域。"

        # 构建包含元数据的上下文字符串
        context_parts = []
        for r in context_nodes:
            node = r.node
            metadata = node.metadata
            
            # 构建法律条文引用信息
            reference = ""
            if "law_name" in metadata:
                reference += f"《{metadata['law_name']}》"
            if "chapter" in metadata:
                reference += f" {metadata['chapter']}"
            if "article_number" in metadata:
                reference += f" {metadata['article_number']}"
            
            # 添加带引用的条文内容
            if reference:
                context_parts.append(f"{reference}:\n{node.get_content()}")
            else:
                context_parts.append(node.get_content())
        
        context_str = "\n\n".join(context_parts)
        
        fmt_prompt = self.law_prompt.format(
            context_str=context_str,
            query_str=query_str,
        )

        try:
            response = self.llm.complete(fmt_prompt)
            logger.info(f"为查询生成回答: {query_str}")
            return str(response)
        except Exception as e:
            logger.error(f"生成回答时出错: {e}")
            return "生成回答时发生错误，请稍后再试。"

class QueryService:
    """处理用户查询的服务"""
    
    def __init__(self, rag_system):
        self.rag = rag_system
    
    async def process_query(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> dict:
        """处理用户查询并返回结果"""
        try:
            # 生成回答
            response = self.rag.generate_response(query, top_k)
            
            # 获取用于回答的相关法律条文
            retrieved_nodes = self.rag.retrieve(query, top_k)
            sources = []
            
            for node in retrieved_nodes:
                metadata = node.node.metadata
                source = {
                    "text": node.node.get_content(),
                    "metadata": {},
                }
                
                # 添加法律引用信息
                if "law_name" in metadata:
                    source["metadata"]["law_name"] = metadata["law_name"]
                if "chapter" in metadata:
                    source["metadata"]["chapter"] = metadata["chapter"]
                if "article_number" in metadata:
                    source["metadata"]["article_number"] = metadata["article_number"]
                
                sources.append(source)
            
            return {
                "answer": response,
                "sources": sources,
            }
        except Exception as e:
            logger.error(f"处理查询时出错: {str(e)}")
            return {
                "answer": "处理您的查询时发生错误，请稍后再试。",
                "sources": [],
            }