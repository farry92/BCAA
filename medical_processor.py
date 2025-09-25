import time
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
import os
from config import Config
from vector_store_manager import VectorStoreManager
from reranker import Reranker
from api_client import APIClient
from prompt_manager import PromptManager

logger = logging.getLogger(__name__)

class MedicalProcessor:
    """医学知识检索和LLM问答的主处理器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.vector_manager = VectorStoreManager(config)
        self.reranker = Reranker(config)
        self.api_client = APIClient(config)
        self.prompt_manager = PromptManager(config)
    
    def initialize(self, is_multi_shard: bool = False) -> bool:
        """初始化所有组件"""
        logger.info("开始初始化 MedicalProcessor 组件")
        
        try:
            # 1. 初始化API客户端
            logger.info("初始化API客户端...")
            if not self.api_client.init_client():
                logger.error("API客户端初始化失败")
                return False
            logger.info("API客户端初始化成功")
            
            # 2. 初始化重排序模型
            logger.info("初始化重排序模型...")
            if not self.reranker.init_model():
                logger.error("重排序模型初始化失败")
                return False
            logger.info("重排序模型初始化成功")
            
            # 3. 加载向量库
            logger.info("开始加载向量库...")
            start_time = time.time()
            
            success = self.vector_manager.load_vector_stores(is_multi_shard)
            
            load_time = time.time() - start_time
            logger.info(f"向量库加载耗时: {load_time:.2f}秒")
            
            if not success:
                logger.error("向量库加载失败")
                return False
            
            logger.info("MedicalProcessor 初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"初始化过程中发生异常: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def process_single_query(self, term: str, question_type: str, query_id: int, 
                           task_type: str) -> Dict[str, Any]:
        """处理单个查询"""
        full_query = f"{term}的{question_type}"
        logger.info(f"--- 启动处理查询 ID {query_id}: '{full_query}' (任务类型: {task_type}) ---")
        
        result_data = self._init_result_data(query_id, term, question_type, full_query, task_type)
        
        total_start_time = time.time()
        
        # 处理症状知识库
        symptom_kb_success = False
        symptom_options = []
        
        if self.vector_manager.symptom_vector_store:
            symptom_kb_success = self._process_knowledge_base(
                full_query, task_type, "symptom_kb", 
                self.vector_manager.symptom_vector_store, 
                result_data, is_symptom_store=True
            )
            
            if task_type == "symptom_extraction":
                symptom_options = self._parse_symptoms_from_answer(result_data["symptom_kb_answer"])
        
        # 处理主知识库
        main_kb_success = False
        if self.vector_manager.vector_store:
            main_kb_success = self._process_knowledge_base(
                full_query, task_type, "main_kb",
                self.vector_manager.vector_store,
                result_data, is_symptom_store=False,
                symptom_options=symptom_options
            )
        
        # 整合结果
        self._integrate_results(result_data, main_kb_success, symptom_kb_success, task_type, full_query)
        
        result_data["total_processing_time"] = round(time.time() - total_start_time, 2)
        logger.info(f"--- 查询 ID {query_id} 处理完毕。总状态: {result_data['overall_status']} ---")
        
        return result_data
    
    def _process_knowledge_base(self, query: str, task_type: str, kb_label: str,
                               vector_store, result_data: Dict, is_symptom_store: bool,
                               symptom_options: Optional[List[str]] = None) -> bool:
        """处理单个知识库的检索和生成"""
        prefix = f"{kb_label}_"
        
        try:
            # 检索文档
            initial_docs = vector_store.similarity_search_with_score(query, k=self.config.initial_k)
            
            # 重排序
            if self.config.use_rerank and initial_docs:
                final_docs = self.reranker.rerank_documents(query, initial_docs, top_k=self.config.final_k)
            else:
                final_docs = initial_docs[:self.config.final_k]
            
            # 构建提示词
            if is_symptom_store and task_type == "symptom_extraction" and final_docs:
                # 从症状库文档标题提取选项
                extracted_titles = sorted(list(set(
                    doc.metadata.get("title", "").strip() 
                    for doc, _ in final_docs 
                    if doc.metadata.get("title")
                )))
                if extracted_titles:
                    symptom_options = extracted_titles
            
            messages = self.prompt_manager.build_prompt(
                query, final_docs, task_type, 
                is_symptom_store=is_symptom_store,
                symptom_options_for_prompt=symptom_options
            )
            
            # 调用LLM
            start_time = time.time()
            answer, success = self.api_client.generate_response(messages)
            processing_time = time.time() - start_time
            
            # 更新结果
            self._update_kb_results(result_data, prefix, initial_docs, final_docs, messages, answer, success, processing_time, kb_label)
            
            return success and answer
            
        except Exception as e:
            logger.error(f"处理 {kb_label} 时出现异常: {str(e)}")
            result_data[f"{prefix}answer"] = f"处理失败: {str(e)}"
            return False
    
    def _integrate_results(self, result_data: Dict, main_kb_success: bool, symptom_kb_success: bool, task_type: str, query: str):
        """整合两个知识库的结果"""
        # 添加知识库标识后缀
        if result_data["main_kb_use_kb"] and "模型调用失败" not in result_data["main_kb_answer"]:
            result_data["main_kb_answer"] += f"\n\n【基于主知识库{' (重排序)' if self.config.use_rerank else ''}】"
        
        if result_data["symptom_kb_use_kb"] and "模型调用失败" not in result_data["symptom_kb_answer"]:
            result_data["symptom_kb_answer"] += f"\n\n【基于症状知识库{' (重排序)' if self.config.use_rerank else ''}】"
        
        if task_type == "symptom_extraction":
            self._handle_symptom_extraction_integration(result_data, query)
        else:
            self._handle_normal_integration(result_data, main_kb_success, symptom_kb_success)
    
    def _handle_symptom_extraction_integration(self, result_data: Dict, query: str):
        """处理症状提取任务的结果整合"""
        main_symptoms = self._parse_symptoms_from_answer(result_data["main_kb_answer"].split("【基于")[0].strip())
        symptom_symptoms = self._parse_symptoms_from_answer(result_data["symptom_kb_answer"].split("【基于")[0].strip())
        
        main_has_valid = bool(main_symptoms)
        symptom_has_valid = bool(symptom_symptoms)
        
        # 构建基础的 overall_answer（无论什么情况都要设置）
        base_answer = (
            f"--- 主知识库结果 ---\n{result_data['main_kb_answer']}\n\n"
            f"--- 症状知识库结果 ---\n{result_data['symptom_kb_answer']}"
        )
        
        if main_has_valid and symptom_has_valid:
            if set(main_symptoms) == set(symptom_symptoms):
                result_data["is_symptom_consistent"] = "一致"
                result_data["overall_pure_symptoms"] = ', '.join(main_symptoms)
                result_data["overall_answer"] = f"{base_answer}\n\n--- 症状提取结果 ---\n症状提取结果一致：{', '.join(main_symptoms)}"
                result_data["overall_status"] = "completed"
            else:
                # 进行元评估
                self._perform_meta_evaluation(result_data, query, main_symptoms, symptom_symptoms)
                # 无论元评估是否成功，都要设置 overall_answer
                meta_result = result_data.get("overall_meta_evaluation_answer", "N/A")
                final_symptoms = result_data.get("overall_pure_symptoms", "未确定")
                
                result_data["overall_answer"] = (
                    f"{base_answer}\n\n"
                    f"--- 症状提取结果 ---\n"
                    f"两个知识库结果不一致\n"
                    f"主知识库: {', '.join(main_symptoms)}\n"
                    f"症状知识库: {', '.join(symptom_symptoms)}\n"
                    f"元评估结果: {meta_result}\n"
                    f"最终确定症状: {final_symptoms}"
                )
        elif main_has_valid:
            result_data["is_symptom_consistent"] = "主知识库有，症状知识库无"
            result_data["overall_pure_symptoms"] = ', '.join(main_symptoms)
            result_data["overall_answer"] = f"{base_answer}\n\n--- 症状提取结果 ---\n{', '.join(main_symptoms)}"
            result_data["overall_status"] = "partially_completed"
        elif symptom_has_valid:
            result_data["is_symptom_consistent"] = "症状知识库有，主知识库无"  
            result_data["overall_pure_symptoms"] = ', '.join(symptom_symptoms)
            result_data["overall_answer"] = f"{base_answer}\n\n--- 症状提取结果 ---\n{', '.join(symptom_symptoms)}"
            result_data["overall_status"] = "partially_completed"
        else:
            result_data["is_symptom_consistent"] = "均无有效症状"
            result_data["overall_pure_symptoms"] = "未提取到症状"
            result_data["overall_answer"] = f"{base_answer}\n\n--- 症状提取结果 ---\n未能从任何知识库中提取到有效症状"
            result_data["overall_status"] = "failed"
    
    def _perform_meta_evaluation(self, result_data: Dict, query: str, main_symptoms: List[str], symptom_symptoms: List[str]):
        """执行元评估"""
        result_data["is_symptom_consistent"] = "不一致"
        
        messages = self.prompt_manager.build_prompt(
            query, [], "normal",
            is_meta_evaluation=True,
            kb1_answer=', '.join(main_symptoms),
            kb1_details=result_data['main_kb_final_retrieval_details'],
            kb2_answer=', '.join(symptom_symptoms),
            kb2_details=result_data['symptom_kb_final_retrieval_details']
        )
        
        meta_answer, meta_success = self.api_client.generate_response(messages)
        
        if meta_success and meta_answer:
            meta_symptoms = self._parse_symptoms_from_answer(meta_answer)
            if meta_symptoms:
                result_data["overall_meta_evaluation_answer"] = ', '.join(meta_symptoms)
                result_data["overall_pure_symptoms"] = ', '.join(meta_symptoms)
                result_data["overall_status"] = "completed_with_meta_eval"
            else:
                result_data["overall_meta_evaluation_answer"] = "元评估未能提取到有效症状"
                result_data["overall_status"] = "inconsistent_no_meta_eval"
        else:
            result_data["overall_meta_evaluation_answer"] = "元评估失败或无响应"
            result_data["overall_status"] = "inconsistent_no_meta_eval"
    
    def _parse_symptoms_from_answer(self, answer_text: str) -> List[str]:
        """从答案中解析症状列表"""
        if not answer_text:
            return []
        
        # 检查无效答案
        invalid_markers = ["模型调用失败", "未提取到症状", "未加载", "处理失败", "无响应"]
        for marker in invalid_markers:
            if marker in answer_text:
                return []
        
        # 清洗文本
        cleaned_text = re.sub(r'\s+', ' ', answer_text).strip()
        cleaned_text = re.sub(r',+', ',', cleaned_text)
        cleaned_text = cleaned_text.strip(',')
        
        # 分割并去重
        symptoms = [s.strip().lower() for s in cleaned_text.split(',') if s.strip()]
        return sorted(list(set(symptoms)))
    
    def _init_result_data(self, query_id: int, term: str, question_type: str, full_query: str, task_type: str) -> Dict[str, Any]:
        """初始化结果数据结构"""
        result_data = {
            "id": query_id,
            "term": term, 
            "question_type": question_type,
            "query": full_query,
            "task_type": task_type,
            "overall_status": "failed",
            "overall_answer": "未能获得有效答案。",
            "overall_meta_evaluation_answer": "N/A",
            "overall_pure_symptoms": "N/A",
            "is_symptom_consistent": "N/A",
            "total_processing_time": 0.0,
        }
        
        # 初始化所有KB字段
        for kb_tag in ["main_kb", "symptom_kb"]:
            prefix = f"{kb_tag}_"
            kb_label_display = {"main_kb": "主知识库", "symptom_kb": "症状知识库"}[kb_tag]
            
            result_data.update({
                f"{prefix}answer": f"{kb_label_display}未处理（知识库未加载或配置）",
                f"{prefix}final_prompt": "",
                f"{prefix}initial_retrieved_count": 0,
                f"{prefix}initial_doc_sources": "",
                f"{prefix}initial_similarity_scores": "",
                f"{prefix}initial_retrieval_details": "未处理",
                f"{prefix}has_reference": False,
                f"{prefix}final_retrieved_count": 0,
                f"{prefix}use_rerank": self.config.use_rerank,
                f"{prefix}use_kb": False,
                f"{prefix}final_doc_sources": "",
                f"{prefix}final_similarity_scores": "",
                f"{prefix}final_rerank_scores": "",
                f"{prefix}final_retrieval_details": "未处理",
                f"{prefix}processing_time": 0.0
            })
        
        return result_data
    
    def _update_kb_results(self, result_data: Dict, prefix: str, initial_docs: List, final_docs: List, 
                          messages: List[dict], answer: Optional[str], success: bool, processing_time: float, kb_label: str):
        """更新知识库处理结果"""
        result_data[f"{prefix}use_kb"] = True
        result_data[f"{prefix}processing_time"] = round(processing_time, 2)
        result_data[f"{prefix}final_prompt"] = PromptManager.format_messages_to_text(messages)
        
        if success and answer:
            result_data[f"{prefix}answer"] = answer
        else:
            result_data[f"{prefix}answer"] = "模型调用失败或无响应"
        
        # 更新检索详情
        self._update_retrieval_details(result_data, prefix, initial_docs, final_docs)
    
    def _update_retrieval_details(self, result_data: Dict, prefix: str, initial_docs: List, final_docs: List):
        """更新检索详情信息"""
        result_data[f"{prefix}initial_retrieved_count"] = len(initial_docs)
        result_data[f"{prefix}final_retrieved_count"] = len(final_docs)
        result_data[f"{prefix}has_reference"] = len(final_docs) > 0
        
        # 填充源文件列表和分数
        if initial_docs:
            initial_sources = []
            initial_scores = []
            initial_info = ""
            for i, (doc, score) in enumerate(initial_docs, 1):
                source = os.path.basename(doc.metadata.get("source", "未知来源"))
                initial_sources.append(source)
                initial_scores.append(f"{score:.4f}")
                initial_info += f"[原始来源{i}] {source} (相似度:{score:.4f})\n"
                initial_info += f"内容片段: {doc.page_content[:200]}...\n\n"
            
            result_data[f"{prefix}initial_doc_sources"] = "; ".join(initial_sources)
            result_data[f"{prefix}initial_similarity_scores"] = "; ".join(initial_scores)
            result_data[f"{prefix}initial_retrieval_details"] = initial_info.strip()
        
        if final_docs:
            final_sources = []
            final_scores = []
            final_rerank_scores = []
            final_info = ""
            for i, (doc, score) in enumerate(final_docs, 1):
                source = os.path.basename(doc.metadata.get("source", "未知来源"))
                final_sources.append(source)
                final_scores.append(f"{score:.4f}")
                if self.config.use_rerank:
                    final_rerank_scores.append(f"{score:.4f}")
                score_type = "重排分数" if self.config.use_rerank else "相似度"
                final_info += f"[最终来源{i}] {source} ({score_type}:{score:.4f})\n"
                final_info += f"内容片段: {doc.page_content[:200]}...\n\n"
            
            result_data[f"{prefix}final_doc_sources"] = "; ".join(final_sources)
            result_data[f"{prefix}final_similarity_scores"] = "; ".join(final_scores)
            if self.config.use_rerank:
                result_data[f"{prefix}final_rerank_scores"] = "; ".join(final_rerank_scores)
            result_data[f"{prefix}final_retrieval_details"] = final_info.strip()

    def _perform_meta_evaluation(self, result_data: Dict, query: str, main_symptoms: List[str], symptom_symptoms: List[str]):
      """执行元评估"""
      result_data["is_symptom_consistent"] = "不一致"
      
      logger.info(f"开始元评估 - 主知识库症状: {main_symptoms}, 症状知识库症状: {symptom_symptoms}")
      
      try:
          messages = self.prompt_manager.build_prompt(
              query, [], "normal",
              is_meta_evaluation=True,
              kb1_answer=', '.join(main_symptoms),
              kb1_details=result_data['main_kb_final_retrieval_details'],
              kb2_answer=', '.join(symptom_symptoms),
              kb2_details=result_data['symptom_kb_final_retrieval_details']
          )
          
          logger.info("元评估提示词构建完成，开始调用API")
          logger.debug(f"元评估提示词: {messages}")
          
          meta_answer, meta_success = self.api_client.generate_response(messages)
          
          logger.info(f"元评估API调用结果 - 成功: {meta_success}, 答案: {meta_answer}")
          
          if meta_success and meta_answer:
              meta_symptoms = self._parse_symptoms_from_answer(meta_answer)
              logger.info(f"元评估解析出的症状: {meta_symptoms}")
              
              if meta_symptoms:
                  result_data["overall_meta_evaluation_answer"] = ', '.join(meta_symptoms)
                  result_data["overall_pure_symptoms"] = ', '.join(meta_symptoms)
                  result_data["overall_status"] = "completed_with_meta_eval"
                  logger.info("元评估成功完成")
              else:
                  result_data["overall_meta_evaluation_answer"] = "元评估未能提取到有效症状"
                  result_data["overall_pure_symptoms"] = "未提取到症状"
                  result_data["overall_status"] = "inconsistent_no_meta_eval"
                  logger.warning("元评估返回但无法解析症状")
          else:
              result_data["overall_meta_evaluation_answer"] = "元评估失败或无响应"
              result_data["overall_pure_symptoms"] = "未提取到症状"  
              result_data["overall_status"] = "inconsistent_no_meta_eval"
              logger.error(f"元评估失败 - API成功: {meta_success}, 答案: {meta_answer}")
              
      except Exception as e:
          logger.error(f"元评估过程中出现异常: {str(e)}")
          result_data["overall_meta_evaluation_answer"] = f"元评估异常: {str(e)}"
          result_data["overall_pure_symptoms"] = "未提取到症状"
          result_data["overall_status"] = "inconsistent_no_meta_eval"

