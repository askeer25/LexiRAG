import os
import logging
import re
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.schema import TextNode, Document, NodeWithScore
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import PromptTemplate
from typing import Optional, Any, List, Set, Tuple, Dict
from pymupdf import EmptyFileError
import chromadb

logger = logging.getLogger("__name__")

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


class LawTextSplitter:
    """专为法律文档设计的文本分割器，按照条款进行分割"""
    
    def __init__(self, chunk_size: int = 1024):
        self.token_splitter = TokenTextSplitter(chunk_size=chunk_size)
        # 中文法律条款正则表达式
        self.article_pattern = re.compile(r'第[一二三四五六七八九十百千]+条')
        self.chapter_pattern = re.compile(r'第[一二三四五六七八九十百千]+章')
    
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
        
    def split_text(self, text: str) -> List[Tuple[str, Dict[str, str]]]:
        """将文本按条款分割，返回文本块和相关元数据"""
        # 先尝试按条款分割
        articles = self.article_pattern.split(text)
        if len(articles) <= 1:  # 如果没有条款标记，则使用常规分割
            chunks = self.token_splitter.split_text(text)
            return [(chunk, {}) for chunk in chunks]
            
        # 找出所有条款的起始位置
        article_positions = [m.start() for m in self.article_pattern.finditer(text)]
        
        # 处理每个条款
        chunks_with_metadata = []
        for i in range(len(article_positions)):
            start = article_positions[i]
            end = article_positions[i+1] if i+1 < len(article_positions) else len(text)
            
            article_text = text[start:end]
            article_number = self._extract_article_number(article_text)
            chapter = ""
            
            # 向前查找章节信息
            if i == 0:
                chapter_text = text[:start]
                chapter = self._extract_chapter(chapter_text)
            else:
                # 从当前条款向前查找到上一个条款的位置
                prev_start = article_positions[i-1]
                chapter_text = text[prev_start:start]
                chapter_match = self.chapter_pattern.search(chapter_text)
                if chapter_match:
                    chapter = chapter_match.group(0)
                elif i > 1:
                    # 如果当前条款与上一个条款之间没有找到章节信息，则使用上一个条款的章节信息
                    for j in range(i-1, -1, -1):
                        prev_chunk_metadata = chunks_with_metadata[j][1]
                        if prev_chunk_metadata.get("chapter"):
                            chapter = prev_chunk_metadata["chapter"]
                            break
            
            # 构建元数据
            metadata = {
                "article_number": article_number,
                "chapter": chapter,
            }
            
            # 添加条款
            chunks_with_metadata.append((article_text, metadata))
        
        return chunks_with_metadata


class DocumentProcessor:
    """负责文档的切分和处理"""
    
    def __init__(self, chunk_size: int = 1024):
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
                    # 添加条款和章节信息到元数据
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


class EmbeddingService:
    """负责文本向量化"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, 
                api_base: Optional[str] = None, dimensions: int = 512):
        self.model = OpenAIEmbedding(
            model=model_name,
            dimensions=dimensions,
            api_key=api_key,
            api_base=api_base,
        )
    
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


class VectorStore:
    """负责向量数据库的管理"""
    
    def __init__(self, index_path: str):
        self.index_path = index_path
        os.makedirs(self.index_path, exist_ok=True)
        self.vector_store = self._init_vector_store()
        self.processed_files = self._load_processed_files()
    
    def _init_vector_store(self):
        """初始化向量存储"""
        chroma_client = chromadb.PersistentClient(path=self.index_path)
        chroma_collection = chroma_client.get_or_create_collection("law_collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        return vector_store
    
    def _load_processed_files(self) -> Set[str]:
        """加载已处理过的文件列表"""
        processed_file_path = os.path.join(self.index_path, "processed_files.txt")
        if os.path.exists(processed_file_path):
            with open(processed_file_path, "r") as f:
                files = [line.strip() for line in f.readlines()]
            return set(files)
        else:
            return set()
    
    def save_processed_files(self):
        """保存已处理过的文件列表"""
        processed_file_path = os.path.join(self.index_path, "processed_files.txt")
        with open(processed_file_path, "w") as f:
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


class HierarchicalRetriever:
    """分层检索器，利用法律文档的层次结构进行检索"""
    
    def __init__(
        self,
        vector_store: Any,
        embed_model: Any,
        similarity_top_k: int = 5,
    ) -> None:
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._similarity_top_k = similarity_top_k
        
    def retrieve(self, query_str: str) -> List[NodeWithScore]:
        """执行分层检索"""
        logger.info(f"开始分层检索，查询: {query_str}")
        
        # 记录所有已检索的节点ID以避免重复
        retrieved_node_ids = set()
        article_nodes_with_scores = []
        
        try:
            # 第一步：检索相关法律文档
            law_query = f"哪部法律涉及 {query_str}"
            logger.info(f"法律级检索，查询: {law_query}")
            law_embedding = self._embed_model.get_query_embedding(law_query)
            law_vector_query = VectorStoreQuery(
                query_embedding=law_embedding,
                similarity_top_k=3,  # 限制返回的法律数量
                mode="default",
            )
            law_results = self._vector_store.query(law_vector_query)
            
            # 提取相关法律名称
            relevant_laws = set()
            for node in law_results.nodes:
                if "law_name" in node.metadata:
                    relevant_laws.add(node.metadata["law_name"])
            
            logger.info(f"找到相关法律: {relevant_laws if relevant_laws else '无'}")
            
            # 第二步：在相关法律中查找相关章节（如果找到了相关法律）
            chapter_nodes = []
            if relevant_laws:
                for law_name in relevant_laws:
                    chapter_query = f"在{law_name}中，哪一章节涉及 {query_str}"
                    logger.info(f"章节级检索，查询: {chapter_query}")
                    chapter_embedding = self._embed_model.get_query_embedding(chapter_query)
                    chapter_vector_query = VectorStoreQuery(
                        query_embedding=chapter_embedding,
                        similarity_top_k=3,  # 限制每部法律返回的章节数量
                        mode="default",
                        # 过滤条件：只检索特定法律
                        filter=lambda node: node.metadata.get("law_name") == law_name,
                    )
                    chapter_results = self._vector_store.query(chapter_vector_query)
                    chapter_nodes.extend(chapter_results.nodes)
            
            # 提取相关章节
            relevant_chapters = set()
            for node in chapter_nodes:
                if "chapter" in node.metadata and "law_name" in node.metadata:
                    relevant_chapters.add((node.metadata["law_name"], node.metadata["chapter"]))
            
            logger.info(f"找到相关章节: {relevant_chapters if relevant_chapters else '无'}")
            
            # 第三步：在相关章节中查找具体条款（如果找到了相关章节）
            if relevant_chapters:
                for law_name, chapter in relevant_chapters:
                    article_query = f"在{law_name}的{chapter}中，哪些条款涉及 {query_str}"
                    logger.info(f"条款级检索，查询: {article_query}")
                    article_embedding = self._embed_model.get_query_embedding(article_query)
                    article_vector_query = VectorStoreQuery(
                        query_embedding=article_embedding,
                        similarity_top_k=self._similarity_top_k,
                        mode="default",
                        # 过滤条件：只检索特定法律的特定章节
                        filter=lambda node: (
                            node.metadata.get("law_name") == law_name and 
                            node.metadata.get("chapter") == chapter
                        ),
                    )
                    article_results = self._vector_store.query(article_vector_query)
                    
                    # 构建NodeWithScore对象
                    for i, node in enumerate(article_results.nodes):
                        score = None
                        if article_results.similarities is not None:
                            score = article_results.similarities[i]
                        
                        if node.node_id not in retrieved_node_ids:
                            article_nodes_with_scores.append(NodeWithScore(node=node, score=score))
                            retrieved_node_ids.add(node.node_id)
        except Exception as e:
            logger.error(f"分层检索过程中出错: {str(e)}")
        
        # 记录分层检索结果数量
        logger.info(f"分层检索找到 {len(article_nodes_with_scores)} 个相关条款")
        
        # 无论分层检索结果如何，始终进行常规检索
        try:
            remaining_k = max(self._similarity_top_k - len(article_nodes_with_scores), 3)
            logger.info(f"执行直接检索，查询: {query_str}, top_k: {remaining_k}")
            
            # 执行常规检索
            direct_embedding = self._embed_model.get_query_embedding(query_str)
            direct_vector_query = VectorStoreQuery(
                query_embedding=direct_embedding,
                similarity_top_k=remaining_k,
                mode="default",
            )
            direct_results = self._vector_store.query(direct_vector_query)
            
            logger.info(f"直接检索找到 {len(direct_results.nodes)} 个结果")
            
            # 添加结果，避免重复
            for i, node in enumerate(direct_results.nodes):
                if node.node_id not in retrieved_node_ids:
                    score = None
                    if direct_results.similarities is not None:
                        score = direct_results.similarities[i]
                    article_nodes_with_scores.append(NodeWithScore(node=node, score=score))
                    retrieved_node_ids.add(node.node_id)
        except Exception as e:
            logger.error(f"直接检索过程中出错: {str(e)}")
        
        # 按相关性对结果进行排序
        article_nodes_with_scores.sort(key=lambda x: x.score if x.score is not None else 0, reverse=True)
        
        # 限制返回结果数量
        results = article_nodes_with_scores[:self._similarity_top_k]
        logger.info(f"最终检索到 {len(results)} 个节点")
        return results


class ResponseGenerator:
    """负责生成最终回答"""
    
    def __init__(self, llm_model_name: str, api_key: Optional[str] = None, 
                api_base: Optional[str] = None):
        self.llm = OpenAI(
            model=llm_model_name,
            api_key=api_key,
            api_base=api_base,
            max_tokens=4096,
            temperature=0.1,
        )
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


class LawRAG:
    """法律RAG系统，专门针对法律文档的检索和问答"""
    
    def __init__(
        self,
        laws_path: str,
        index_path: str,
        llm_model_name: str,
        embedding_model_name: str,
    ) -> None:
        # 初始化路径
        self.laws_path = laws_path
        self.index_path = index_path
        os.makedirs(self.laws_path, exist_ok=True)
        os.makedirs(self.index_path, exist_ok=True)
        
        # 获取API密钥
        llm_api_key = os.getenv("OPENAI_API_KEY")
        llm_api_base = os.getenv("OPENAI_BASE_URL")
        embed_api_key = os.getenv("OPENAI_API_KEY")
        embed_api_base = os.getenv("OPENAI_BASE_URL")
        
        # 初始化各组件
        self.document_loader = DocumentLoader()
        self.document_processor = DocumentProcessor(chunk_size=1024)
        self.embedding_service = EmbeddingService(
            model_name=embedding_model_name,
            api_key=embed_api_key,
            api_base=embed_api_base,
        )
        self.vector_store = VectorStore(index_path=self.index_path)
        self.response_generator = ResponseGenerator(
            llm_model_name=llm_model_name,
            api_key=llm_api_key,
            api_base=llm_api_base,
        )
    
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
        supported_extensions = [".pdf", ".md", ".txt"]
        file_list = []
        for filepath, _, filenames in os.walk(self.laws_path):
            for filename in filenames:
                if any(filename.lower().endswith(ext) for ext in supported_extensions):
                    file_path = os.path.join(filepath, filename)
                    if not self.vector_store.is_processed(file_path):
                        file_list.append(file_path)
        
        processed_count = 0
        for file_path in file_list:
            if self.process_file(file_path):
                processed_count += 1
        
        self.vector_store.save_processed_files()
        logger.info(f"初始化时处理了 {processed_count} 个文件")
        return file_list
    
    def retrieve(self, query_str: str, top_k: int = 5) -> List[NodeWithScore]:
        """检索相关内容，使用分层检索器"""
        retriever = HierarchicalRetriever(
            self.vector_store.get_vector_store(),
            self.embedding_service,
            similarity_top_k=top_k,
        )
        retrieved_nodes = retriever.retrieve(query_str)
        logger.info(f"为查询检索到 {len(retrieved_nodes)} 个节点")
        return retrieved_nodes
    
    def generate_response(self, query_str: str, top_k: int = 5) -> str:
        """生成回答"""
        # 检索阶段
        retrieved_nodes = self.retrieve(query_str, top_k)
        
        # 生成阶段
        response = self.response_generator.generate_response(query_str, retrieved_nodes)
        return response


# 用于FastAPI的查询接口
class QueryService:
    """处理用户查询的服务"""
    
    def __init__(self, rag_system: LawRAG):
        self.rag = rag_system
    
    async def process_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
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


if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()

    # 使用环境变量或默认值
    llm_model_name = os.getenv("LLM_MODEL", "gpt-4o")
    embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

    # 创建法律RAG系统
    law_rag = LawRAG(
        laws_path="lawsfiles",  # 法律文件目录
        index_path="law_index",  # 向量索引目录
        llm_model_name=llm_model_name,
        embedding_model_name=embedding_model_name,
    )
    
    # 初始化法律文档库
    processed_files = law_rag.init_laws_library()
    print(f"已处理 {len(processed_files)} 个法律文档文件")

    # 测试查询
    print("\n--- 测试查询: 土地征收补偿 ---")
    response = law_rag.generate_response("土地征收时，农民应该获得哪些补偿？", top_k=5)
    print(response)


