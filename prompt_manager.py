import os
import logging
from typing import List, Dict, Optional, Tuple
from langchain_core.documents import Document
from config import Config

logger = logging.getLogger(__name__)

class PromptManager:
    """提示词管理器"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def build_prompt(self, query: str, retrieved_docs: List[Tuple[Document, float]], 
                     task_type: str, is_symptom_store: bool = False, 
                     symptom_options_for_prompt: Optional[List[str]] = None,
                     is_meta_evaluation: bool = False,
                     kb1_answer: Optional[str] = None, kb1_details: Optional[str] = None,
                     kb2_answer: Optional[str] = None, kb2_details: Optional[str] = None) -> List[dict]:
        """构建LLM的提示词"""
        
        if is_meta_evaluation:
            return self._build_meta_evaluation_prompt(query, kb1_answer, kb1_details, kb2_answer, kb2_details)
        else:
            return self._build_normal_prompt(query, retrieved_docs, task_type, is_symptom_store, symptom_options_for_prompt)
    
    def _build_meta_evaluation_prompt(self, query: str, kb1_answer: Optional[str], kb1_details: Optional[str],
                                     kb2_answer: Optional[str], kb2_details: Optional[str]) -> List[dict]:
        """构建元评估提示词"""
        system_prompt = ("你是专业的医学症状整合和判断专家。你将收到一个用户描述、两个知识库（主知识库和症状知识库）"
                        "各自提取出的症状，以及它们各自的检索来源和评分。你的任务是基于所有提供的证据，评估两个知识库的"
                        "症状提取结果，并给出一个最终的、最准确和最完整的症状列表。如果两个知识库结果一致，直接返回该列表；"
                        "如果不一致，请说明你的判断依据并给出最合理的症状列表。请只返回症状本身，用逗号分隔，不添加解释或客套话。"
                        "如果无法提取，请回答'未提取到症状'。")
        
        user_content = f"用户描述: {query}\n\n"
        user_content += f"--- 主知识库提取结果 ---\n"
        user_content += f"症状: {kb1_answer if kb1_answer else '无有效症状'}\n"
        user_content += f"参考来源详情:\n{kb1_details if kb1_details else '无检索结果'}\n\n"
        user_content += f"--- 症状知识库提取结果 ---\n"
        user_content += f"症状: {kb2_answer if kb2_answer else '无有效症状'}\n"
        user_content += f"参考来源详情:\n{kb2_details if kb2_details else '无检索结果'}\n\n"
        user_content += "请根据上述信息，给出最终的、最准确的症状列表："

        return self._format_messages(system_prompt, user_content)
    
    def _build_normal_prompt(self, query: str, retrieved_docs: List[Tuple[Document, float]], 
                           task_type: str, is_symptom_store: bool, 
                           symptom_options_for_prompt: Optional[List[str]]) -> List[dict]:
        """构建正常查询的提示词"""
        
        # 构建上下文
        context = self._build_context(retrieved_docs)
        kb_label_display = "症状知识库" if is_symptom_store else "主知识库"
        
        # 构建系统提示词
        system_prompt = self._get_system_prompt(task_type, symptom_options_for_prompt, kb_label_display)
        
        # 构建用户内容
        has_context = len(retrieved_docs) > 0
        if has_context:
            user_content = f"用户问题: {query}\n\n参考来源:\n{context}"
        else:
            user_content = f"用户问题: {query}\n\n注意：本次查询没有找到相关的参考来源，请尽力回答。"
        
        return self._format_messages(system_prompt, user_content)
    
    def _build_context(self, retrieved_docs: List[Tuple[Document, float]]) -> str:
        """构建参考文档上下文"""
        context = ""
        for i, (doc, score) in enumerate(retrieved_docs, 1):
            source = os.path.basename(doc.metadata.get("source", "未知来源"))
            context += f"===== 参考来源 {i} =====\n"
            context += f"文件: {source}\n相似度: {score:.4f}\n"
            context += f"内容:\n{doc.page_content}\n\n"
        return context
    
    def _get_system_prompt(self, task_type: str, symptom_options: Optional[List[str]], kb_label: str) -> str:
        """根据任务类型获取系统提示词"""
        prompts = {
            "symptom_extraction": self._get_symptom_extraction_prompt(symptom_options),
            "definition": "你是医学术语定义专家，用1-3句话简洁解释术语。回答时请清晰、准确，并避免不必要的客套话。",
            "treatment": "你是治疗方案专家，请根据提供的参考来源，分步骤列出治疗方法，确保信息准确、完整，并易于理解。",
            "normal": "你是专业医学专家，请基于提供的参考来源回答用户问题。请确保回答准确、专业，并标注信息来源（如\"根据[来源X]\"或在末尾统一列出）。如果参考来源不足以回答问题，请说明。"
        }
        
        base_prompt = prompts.get(task_type, prompts["normal"])
        return f"({kb_label}) {base_prompt}"
    
    def _get_symptom_extraction_prompt(self, symptom_options: Optional[List[str]]) -> str:
        """获取症状提取的提示词"""
        base_prompt = "你是医学症状提取专家，需从用户提供的口语化文本中精准识别症状。"
        
        if symptom_options:
            options_str = ", ".join(symptom_options)
            base_prompt += f"请根据参考来源，并参考以下可能的症状名称进行识别：[{options_str}]。"
        
        base_prompt += "要求：1. 只返回症状本身，用逗号分隔，不添加解释；2. 确保症状与原文一致。如果未找到匹配症状，请回答'未提取到症状'。"
        return base_prompt
    
    def _format_messages(self, system_prompt: str, user_content: str) -> List[dict]:
        """根据API类型格式化消息"""
        if self.config.api_type in ["deepseek", "qianwen", "ernie"]:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        elif self.config.api_type == "doubao":
            return [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": user_content}]}
            ]
        else:
            logger.error(f"不支持的API类型: {self.config.api_type}")
            return []
    
    @staticmethod
    def format_messages_to_text(messages: List[dict]) -> str:
        """将消息格式转换为可读文本"""
        prompt_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if isinstance(content, list):
                text_content = ""
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_content += item.get("text", "")
                content = text_content
            
            prompt_parts.append(f"[{role.upper()}]\n{content}\n")
        
        return "\n".join(prompt_parts)
