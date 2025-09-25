import os
import time
import logging
from typing import Optional, List
import torch

# 容错导入
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from config import Config

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """向量库管理器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.vector_store: Optional[FAISS] = None
        self.symptom_vector_store: Optional[FAISS] = None
    
    def init_embeddings(self) -> bool:
        """初始化嵌入模型"""
        logger.info("初始化嵌入模型...")
        logger.info(f"设备: {self.config.device}")
        logger.info(f"模型路径: {self.config.embedding_model}")
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={"device": self.config.device, "trust_remote_code": True},
                encode_kwargs={"normalize_embeddings": True}
            )
            embed_dim = len(self.embeddings.embed_query("医学测试文本"))
            logger.info(f"嵌入模型加载成功，维度: {embed_dim}")
            return True
        except Exception as e:
            logger.error(f"嵌入模型加载失败: {str(e)}")
            return False
    
    def load_vector_stores(self, is_multi_shard: bool = False) -> bool:
        """加载向量库"""
        logger.info(f"开始加载向量库... (multi_shard={is_multi_shard})")
        logger.info(f"主向量库路径: {self.config.vector_path}")
        logger.info("开始加载向量库...")
        start_time = time.time()
        
        if not self.init_embeddings():
            return False
        
        embed_dim = len(self.embeddings.embed_query("测试"))
        
        try:
            # 加载主向量库
            if os.path.exists(self.config.vector_path):
                if is_multi_shard:
                    self.vector_store = self._load_multi_shard(self.config.vector_path, embed_dim)
                else:
                    self.vector_store = self._load_single_shard(self.config.vector_path, embed_dim)
                
                if self.vector_store:
                    doc_count = len(self.vector_store.docstore._dict)
                    logger.info(f"主向量库加载完成: {doc_count}文档")
            
            # 加载症状向量库
            if self.config.symptom_path and os.path.exists(self.config.symptom_path):
                self.symptom_vector_store = self._load_single_shard(self.config.symptom_path, embed_dim)
                if self.symptom_vector_store:
                    symptom_count = len(self.symptom_vector_store.docstore._dict)
                    logger.info(f"症状向量库加载完成: {symptom_count}文档")
            
            if self.vector_store or self.symptom_vector_store:
                load_time = time.time() - start_time
                logger.info(f"向量库加载耗时: {load_time:.2f}秒")
                return True
            else:
                logger.error("未成功加载任何向量库")
                return False
                
        except Exception as e:
            logger.error(f"向量库加载异常: {str(e)}")
            return False
    
    def _load_single_shard(self, path: str, embed_dim: int) -> Optional[FAISS]:
        """加载单个分片"""
        try:
            store = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
            if store.index.d != embed_dim:
                logger.warning(f"向量库维度不匹配: 期望{embed_dim}, 实际{store.index.d}")
                return None
            return store
        except Exception as e:
            logger.warning(f"加载向量库分片失败: {str(e)}")
            return None
    
    def _load_multi_shard(self, root_path: str, embed_dim: int) -> Optional[FAISS]:
        """加载多分片向量库"""
        shard_paths = [os.path.join(root_path, s) for s in os.listdir(root_path) 
                      if os.path.isdir(os.path.join(root_path, s))]
        valid_shards = [p for p in shard_paths if all(os.path.exists(os.path.join(p, f)) 
                       for f in ["index.faiss", "index.pkl"])]
        
        if not valid_shards:
            logger.error(f"未找到有效分片")
            return None
            
        try:
            base_store = self._load_single_shard(valid_shards[0], embed_dim)
            if not base_store:
                return None
                
            for shard_path in valid_shards[1:]:
                shard = self._load_single_shard(shard_path, embed_dim)
                if shard:
                    base_store.merge_from(shard)
                    
            return base_store
        except Exception as e:
            logger.error(f"合并多分片向量库异常: {str(e)}")
            return None
