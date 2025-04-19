#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
from typing import List
from llama_index.core.schema import Document
from llama_index.readers.file import PyMuPDFReader
from pymupdf import EmptyFileError
import config

logger = logging.getLogger(__name__)

class DocumentLoader:
    """负责加载和解析不同格式的文档"""
    
    def __init__(self):
        self.pdf_loader = PyMuPDFReader()
    
    def load_document(self, file_path: str) -> List[Document]:
        """加载文档并返回Document对象列表"""
        try:
            if file_path.lower().endswith(".pdf"):
                documents = self.pdf_loader.load(file_path=file_path)
                # 添加法律名称到元数据
                law_name = os.path.basename(file_path).split('.')[0]
                for doc in documents:
                    doc.metadata["law_name"] = law_name
            elif file_path.lower().endswith((".md", ".txt")):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                law_name = os.path.basename(file_path).split('.')[0]
                metadata = {
                    "file_path": file_path,
                    "file_name": os.path.basename(file_path),
                    "file_type": "md" if file_path.lower().endswith(".md") else "txt",
                    "law_name": law_name,
                }
                documents = [Document(text=text, metadata=metadata)]
            else:
                logger.warning(f"不支持的文件格式: {file_path}")
                return []

            return documents
        except EmptyFileError as e:
            logger.error(f"无法打开空文件: {file_path}")
            logger.warning(f"移除空文件: {file_path}")
            os.remove(file_path)
            return []
        except Exception as e:
            logger.error(f"加载文件 {file_path} 失败: {str(e)}")
            os.remove(file_path)
            logger.warning(f"移除有问题的文件: {file_path}")
            return []