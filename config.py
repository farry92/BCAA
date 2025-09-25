import os
from typing import Dict, List, Optional
from dataclasses import dataclass
import torch

@dataclass
class Config:
    """配置参数类"""
    api_type: str = "deepseek"
    api_key: Optional[str] = None
    model: Optional[str] = "deepseek-chat"
    embedding_model: str = "/mnt/fang/qwen3_embedding_0.6B_finetuned_inbatch_pro_earlystop"
    reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"
    use_rerank: bool = False
    multi_shard: bool = True
    gpu_id: Optional[int] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 检索参数
    initial_k: int = 20
    final_k: int = 5
    rerank_max_length: int = 4096
    
    # 处理参数
    batch_size: int = 10
    max_tokens: int = 2048
    temperature: float = 0.2
    
    # 路径参数
    vector_path: str = ""
    symptom_path: Optional[str] = None
    output_dir: str = ""
    output_prefix: str = "medical_rag_results"
    
    # 支持的API和模型配置
    supported_apis: Dict[str, str] = None
    supported_models: Dict[str, Dict[str, str]] = None
    
    def __post_init__(self):
        if self.supported_apis is None:
            self.supported_apis = {
                "deepseek": "DeepSeek API",
                "doubao": "豆包API",
                "qianwen": "千问API",
                "ernie": "百度文心ERNIE API"
            }
        
        if self.supported_models is None:
            self.supported_models = {
                "deepseek": {
                    "deepseek-chat": "通用对话模型（默认）",
                    "deepseek-r1": "增强推理模型",
                    "deepseek-coder": "代码生成模型"
                },
                "doubao": {
                    "doubao-seed-1-6-thinking-250715": "豆包思考模型（默认）",
                    "doubao-pro": "豆包专业版"
                },
                "qianwen": {
                    "qwen-turbo": "千问 turbo 版（默认）",
                    "qwen-plus": "千问 plus 版"
                },
                "ernie": {
                    "ernie-3.5-8k": "ERNIE 3.5 8K版（默认）",
                    "ernie-4.0-8k": "ERNIE 4.0 8K版"
                }
            }
