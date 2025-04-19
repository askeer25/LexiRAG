#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import logging
from typing import List, Tuple, Dict
from llama_index.core.schema import Document, TextNode
from llama_index.core.text_splitter import TokenTextSplitter
import config

logger = logging.getLogger(__name__)

class LawTextSplitter:
    """专为法律文档设计的文本分割器，按照章-节-条的层级结构进行分割"""
    
    def __init__(self, chunk_size: int = config.CHUNK_SIZE):
        self.token_splitter = TokenTextSplitter(chunk_size=chunk_size)
        # 中文法律文本结构正则表达式
        self.article_pattern = re.compile(r'第[一二三四五六七八九十百千]+条')
        self.chapter_pattern = re.compile(r'第[一二三四五六七八九十百千]+章')
        self.section_pattern = re.compile(r'第[一二三四五六七八九十百千]+节')
    
    def _extract_article_number(self, text: str) -> str:
        """从文本中提取条款号"""
        match = self.article_pattern.search(text)
        if match:
            return match.group(0)
        return ""
    
    def _extract_chapter(self, text: str) -> str:
        """从文本中提取章节"""
        match = self.chapter_pattern.search(text)
        if match:
            return match.group(0)
        return ""
    
    def _extract_section(self, text: str) -> str:
        """从文本中提取节"""
        match = self.section_pattern.search(text)
        if match:
            return match.group(0)
        return ""
        
    def _remove_spaces(self, text: str) -> str:
        """删除文本中的空格"""
        return re.sub(r'\s+', '', text)
        
    def split_text(self, text: str) -> List[Tuple[str, Dict[str, str]]]:
        """将文本按章-节-条层级结构分割，返回文本块和相关元数据"""
        # 先删除空格
        text = self._remove_spaces(text)
        
        # 按条款分割
        article_matches = list(self.article_pattern.finditer(text))
        
        if len(article_matches) <= 1:  # 如果没有条款标记或只有一个，则使用常规分割
            chunks = self.token_splitter.split_text(text)
            return [(chunk, {}) for chunk in chunks]
            
        # 找出所有条款的起始位置
        article_positions = [m.start() for m in article_matches]
        
        # 处理每个条款
        chunks_with_metadata = []
        for i in range(len(article_positions)):
            start = article_positions[i]
            end = article_positions[i+1] if i+1 < len(article_positions) else len(text)
            
            # 确保包含条款号
            article_text = text[start:end].strip()
            article_number = article_matches[i].group(0)
            chapter = ""
            section = ""
            
            # 向前查找章节信息
            if i == 0:
                chapter_text = text[:start]
                chapter = self._extract_chapter(chapter_text)
                section = self._extract_section(chapter_text)
            else:
                # 从当前条款向前查找到上一个条款的位置
                prev_start = 0
                search_start = max(0, article_positions[i] - 1000)  # 向前查找最多1000个字符
                
                # 查找最近的章节和节信息
                chapter_text = text[search_start:start]
                chapter_match = self.chapter_pattern.search(chapter_text)
                section_match = self.section_pattern.search(chapter_text)
                
                # 提取章节信息
                if chapter_match:
                    chapter = chapter_match.group(0)
                elif i > 0 and chunks_with_metadata:
                    # 使用上一个条款的章信息
                    prev_chunk_metadata = chunks_with_metadata[i-1][1]
                    if prev_chunk_metadata.get("chapter"):
                        chapter = prev_chunk_metadata["chapter"]
                
                # 提取节信息
                if section_match:
                    section = section_match.group(0)
                elif i > 0 and chunks_with_metadata:
                    # 如果没有找到新的节信息，沿用之前的节信息
                    prev_chunk_metadata = chunks_with_metadata[i-1][1]
                    if prev_chunk_metadata.get("section"):
                        section = prev_chunk_metadata["section"]
                
                # 如果找到新的章信息，清除之前的节信息（因为新章节开始了）
                if chapter_match and i > 0 and chunks_with_metadata:
                    prev_chunk_metadata = chunks_with_metadata[i-1][1]
                    if prev_chunk_metadata.get("chapter") != chapter:
                        section = self._extract_section(chapter_text[chapter_match.end():])
            
            # 构建元数据
            metadata = {
                "article_number": article_number,
                "chapter": chapter,
                "section": section
            }
            
            # 添加条款
            if len(article_text.strip()) > 0:  # 只添加非空内容
                chunks_with_metadata.append((article_text, metadata))
        
        return chunks_with_metadata


class DocumentProcessor:
    """负责文档的切分和处理"""
    
    def __init__(self, chunk_size: int = config.CHUNK_SIZE):
        self.text_splitter = TokenTextSplitter(chunk_size=chunk_size)
        self.law_splitter = LawTextSplitter(chunk_size=chunk_size)
    
    def process_documents(self, documents: List[Document]) -> List[TextNode]:
        """将文档切分成块并返回TextNode对象"""
        nodes = []
        for doc in documents:
            # 判断是否是法律文档
            is_law_document = "law_name" in doc.metadata
            
            if is_law_document:
                # 使用法律专用分割器
                chunks_with_metadata = self.law_splitter.split_text(doc.text)
                for chunk, chunk_metadata in chunks_with_metadata:
                    node = TextNode(text=chunk)
                    node.metadata = doc.metadata.copy()
                    # 添加条款、章和节信息到元数据
                    node.metadata.update(chunk_metadata)
                    nodes.append(node)
            else:
                # 使用通用分割器
                chunks = self.text_splitter.split_text(doc.text)
                for chunk in chunks:
                    node = TextNode(text=chunk)
                    node.metadata = doc.metadata.copy()
                    nodes.append(node)
        
        return nodes
    
    def test_law_document_processing(self, law_text: str) -> List[Dict]:
        """测试函数：处理法律文档并返回3个条款的内容及其层级索引
        
        Args:
            law_text: 法律文本内容
            
        Returns:
            包含3个条款信息的列表，每个条款包含内容和章-节-条索引
        """
        # 创建一个示例法律文档
        example_doc = Document(
            text=law_text,
            metadata={"law_name": "测试法律文档"}
        )
        
        # 处理文档
        nodes = self.process_documents([example_doc])
        
        # 获取前3个条款的信息（如果有的话）
        result = []
        for i, node in enumerate(nodes[:3]):
            article_info = {
                "内容": node.text,
                "法律文档": node.metadata.get("law_name", ""),
                "章": node.metadata.get("chapter", ""),
                "节": node.metadata.get("section", ""),
                "条": node.metadata.get("article_number", "")
            }
            result.append(article_info)
            
        return result


def main():
    """主函数，用于测试法律文档处理功能"""
    # 示例法律文本（包含章-节-条结构）
    example_law_text = """
    中华人民共和国测试法

    第一章 总则
    
    第一条 为了规范测试活动，保障测试质量，制定本法。
    
    第二条 本法适用于中华人民共和国境内的测试活动。
    
    第二章 测试要求
    
    第一节 测试准备
    
    第三条 测试前应当制定完善的测试计划。
    
    第四条 测试人员应当具备相应的专业知识和技能。
    
    第二节 测试实施
    
    第五条 测试应当按照测试计划进行，并做好记录。
    
    第六条 发现问题应当及时报告并处理。
    
    第三章 测试结果管理
    
    第七条 测试完成后应当形成测试报告。
    
    第八条 测试报告应当客观反映测试结果。
    """
    
    # 创建文档处理器
    processor = DocumentProcessor()
    
    # 测试法律文档处理
    results = processor.test_law_document_processing(example_law_text)
    
    # 打印测试结果
    print("法律文档切分测试结果：")
    for i, result in enumerate(results):
        print(f"\n条款 {i+1}:")
        for key, value in result.items():
            print(f"{key}: {value}")
    
    return results


if __name__ == "__main__":
    main()