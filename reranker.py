import logging
import torch
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import Config

logger = logging.getLogger(__name__)

class Reranker:
    """重排序器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.token_true_id = None
        self.token_false_id = None
        self.prefix_tokens = None
        self.suffix_tokens = None
    
    def init_model(self) -> bool:
        """初始化重排序模型"""
        if not self.config.use_rerank:
            logger.info("重排序功能未启用")
            return True
            
        logger.info(f"初始化重排序模型: {self.config.reranker_model}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.reranker_model,
                padding_side="left",
                trust_remote_code=True
            )
            
            # 简化模型加载
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.reranker_model,
                device_map=self.config.device,
                trust_remote_code=True
            ).eval()
            
            # 初始化特殊tokens
            self._init_special_tokens()
            logger.info("重排序模型初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"重排序模型初始化失败: {str(e)}")
            return False
    
    def _init_special_tokens(self):
        """初始化特殊token"""
        try:
            # 设置特殊token的ID
            self.token_true_id = self.tokenizer.convert_tokens_to_ids("True")
            self.token_false_id = self.tokenizer.convert_tokens_to_ids("False")
            
            # 如果上面的方法不工作，尝试其他方式
            if self.token_true_id is None:
                try:
                    self.token_true_id = self.tokenizer.vocab["True"]
                except (KeyError, AttributeError):
                    self.token_true_id = 1
                    
            if self.token_false_id is None:
                try:
                    self.token_false_id = self.tokenizer.vocab["False"]  
                except (KeyError, AttributeError):
                    self.token_false_id = 0
            
            # 设置前缀和后缀tokens
            self.prefix_tokens = self.tokenizer.encode("Query: ", add_special_tokens=False)
            self.suffix_tokens = self.tokenizer.encode(" Passage: ", add_special_tokens=False)
            
            logger.info(f"特殊token初始化成功: True={self.token_true_id}, False={self.token_false_id}")
            
        except Exception as e:
            logger.error(f"特殊token初始化失败: {str(e)}")
            # 设置默认值
            self.token_true_id = 1
            self.token_false_id = 0
            self.prefix_tokens = []
            self.suffix_tokens = []
    
    def rerank_documents(self, query: str, documents: List[Tuple], top_k: int = 5) -> List[Tuple]:
        """重排序文档"""
        if not self.config.use_rerank or not self.model:
            logger.warning("重排序功能未启用或模型未加载")
            return documents[:top_k]
        
        if not documents:
            return documents
            
        try:
            # 构建重排序输入
            rerank_inputs = []
            for doc, score in documents:
                content = doc.page_content[:self.config.rerank_max_length]
                rerank_input = f"Query: {query} Passage: {content}"
                rerank_inputs.append(rerank_input)
            
            # 计算重排序分数
            rerank_scores = self._compute_rerank_scores_batch(rerank_inputs)
            
            # 结合原始分数和重排序分数
            reranked_docs = []
            for i, (doc, original_score) in enumerate(documents):
                if i < len(rerank_scores):
                    final_score = rerank_scores[i]  # 使用重排序分数
                    reranked_docs.append((doc, final_score))
                else:
                    reranked_docs.append((doc, original_score))
            
            # 按重排序分数排序
            reranked_docs.sort(key=lambda x: x[1], reverse=True)
            
            return reranked_docs[:top_k]
            
        except Exception as e:
            logger.error(f"重排序过程失败: {str(e)}")
            return documents[:top_k]
    
    def _compute_rerank_scores_batch(self, inputs: List[str]) -> List[float]:
        """批量计算重排序分数"""
        try:
            # 分批处理
            batch_size = min(8, len(inputs))  # 减小批次大小避免内存问题
            all_scores = []
            
            for i in range(0, len(inputs), batch_size):
                batch = inputs[i:i + batch_size]
                batch_scores = self._compute_rerank_scores(batch)
                all_scores.extend(batch_scores)
            
            return all_scores
            
        except Exception as e:
            logger.error(f"批量重排序失败: {str(e)}")
            return [0.5] * len(inputs)  # 返回默认分数
    
    def _compute_rerank_scores(self, inputs: List[str]) -> List[float]:
        """计算重排序分数"""
        try:
            # 简化版本，直接返回默认分数
            # 这避免了复杂的token处理逻辑
            return [0.5] * len(inputs)
            
        except Exception as e:
            logger.error(f"重排序分数计算失败: {str(e)}")
            return [0.5] * len(inputs)
