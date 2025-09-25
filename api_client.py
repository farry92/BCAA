import os
import logging
import requests
from typing import List, Dict, Tuple, Optional, Any
from config import Config

logger = logging.getLogger(__name__)

class APIClient:
    """大模型API客户端管理器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client: Any = None
    
    def init_client(self) -> bool:
        """初始化API客户端"""
        logger.info(f"初始化API客户端: {self.config.supported_apis.get(self.config.api_type)}")
        
        api_key = self.config.api_key or self._get_api_key_from_env()
        if not api_key:
            logger.error("请提供API密钥")
            return False

        try:
            if self.config.api_type == "deepseek":
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com"
                )
            # 可以添加其他API的客户端初始化
            
            logger.info("API客户端初始化成功")
            return True
        except Exception as e:
            logger.error(f"API客户端初始化失败: {str(e)}")
            return False
    
    def _get_api_key_from_env(self) -> Optional[str]:
        """从环境变量获取API密钥"""
        key_mapping = {
            "deepseek": "DEEPSEEK_API_KEY",
            "doubao": "DOUBAO_API_KEY", 
            "qianwen": "QIANWEN_API_KEY",
            "ernie": "BAIDU_API_KEY"
        }
        return os.environ.get(key_mapping.get(self.config.api_type))
    
    def generate_response(self, messages: List[dict]) -> Tuple[Optional[str], bool]:
        """调用大模型API生成回答"""
        try:
            if self.config.api_type == "deepseek":
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
                return response.choices[0].message.content, True
            
            elif self.config.api_type == "doubao":
                return self._call_doubao_api(messages)
            
            elif self.config.api_type == "qianwen":
                return self._call_qianwen_api(messages)
            
            elif self.config.api_type == "ernie":
                return self._call_ernie_api(messages)
            else:
                logger.error(f"不支持的API类型: {self.config.api_type}")
                return None, False
                
        except Exception as e:
            logger.error(f"API调用异常: {str(e)}")
            return None, False
    
    def _call_doubao_api(self, messages: List[dict]) -> Tuple[Optional[str], bool]:
        """调用豆包API"""
        try:
            response = requests.post(
                "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}", 
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.config.model, 
                    "messages": messages, 
                    "max_tokens": self.config.max_tokens, 
                    "temperature": self.config.temperature
                }
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = content[0]["text"]
            return content, True
        except Exception as e:
            logger.error(f"豆包API调用失败: {str(e)}")
            return None, False
    
    def _call_qianwen_api(self, messages: List[dict]) -> Tuple[Optional[str], bool]:
        """调用千问API"""
        try:
            response = requests.post(
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}", 
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.config.model, 
                    "messages": messages,
                    "max_tokens": self.config.max_tokens, 
                    "temperature": self.config.temperature
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"], True
        except Exception as e:
            logger.error(f"千问API调用失败: {str(e)}")
            return None, False
    
    def _call_ernie_api(self, messages: List[dict]) -> Tuple[Optional[str], bool]:
        """调用文心API"""
        try:
            response = requests.post(
                "https://qianfan.baidubce.com/v2/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}", 
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.config.model, 
                    "messages": messages,
                    "max_tokens": self.config.max_tokens, 
                    "temperature": self.config.temperature
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"], True
        except Exception as e:
            logger.error(f"文心API调用失败: {str(e)}")
            return None, False
