import os
import json
import pandas as pd
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def load_terms_from_file(file_path: Optional[str] = None, single_term: Optional[str] = None) -> List[str]:
    """从文件或命令行参数加载术语列表"""
    if single_term:
        logger.info(f"使用单个术语: '{single_term}'")
        return [single_term.strip()] if single_term.strip() else []
    
    if not file_path or not os.path.exists(file_path):
        logger.error(f"术语文件不存在: {file_path}")
        return []
    
    logger.info(f"加载术语文件: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            terms = [line.strip() for line in f if line.strip()]
        logger.info(f"成功加载术语: {len(terms)}个")
        return terms
    except Exception as e:
        logger.error(f"加载术语失败: {str(e)}")
        return []

def load_question_types(file_path: Optional[str] = None, single_type: Optional[str] = None) -> List[str]:
    """从文件或命令行参数加载问题类型列表"""
    if single_type:
        logger.info(f"使用单个问题类型: '{single_type}'")
        return [single_type.strip()] if single_type.strip() else []
        
    if not file_path or not os.path.exists(file_path):
        logger.error(f"问题类型文件不存在: {file_path}")
        return []
    
    logger.info(f"加载问题类型文件: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            types = [line.strip() for line in f if line.strip()]
        logger.info(f"成功加载问题类型: {len(types)}个")
        return types
    except Exception as e:
        logger.error(f"加载问题类型失败: {str(e)}")
        return []

def load_progress(output_dir: str, output_prefix: str) -> dict:
    """加载处理进度"""
    progress_file = os.path.join(output_dir, f"{output_prefix}_progress.json")
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
            completed_count = len(progress.get('completed_ids', []))
            logger.info(f"发现历史进度文件，{completed_count}个查询已完成")
            return progress
        except Exception as e:
            logger.warning(f"加载进度文件失败: {str(e)}")
    return {"completed_ids": [], "total_queries": 0, "batches_completed": 0}

def save_progress(output_dir: str, output_prefix: str, progress_data: dict) -> bool:
    """保存处理进度"""
    os.makedirs(output_dir, exist_ok=True)
    progress_file = os.path.join(output_dir, f"{output_prefix}_progress.json")
    try:
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.warning(f"保存进度失败: {str(e)}")
        return False

def save_excel_results(output_dir: str, output_prefix: str, results: List[dict], is_final: bool = False) -> Optional[str]:
    """保存结果到Excel文件"""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    
    # 定义列顺序
    columns = [
        "id", "term", "question_type", "query", "task_type",
        "overall_status", "is_symptom_consistent", "overall_meta_evaluation_answer",
        "overall_pure_symptoms", "overall_answer", "total_processing_time",
        
        # 主知识库结果
        "main_kb_answer", "main_kb_final_prompt", "main_kb_initial_retrieved_count", 
        "main_kb_initial_doc_sources", "main_kb_initial_similarity_scores", 
        "main_kb_initial_retrieval_details", "main_kb_has_reference", 
        "main_kb_final_retrieved_count", "main_kb_use_rerank", "main_kb_use_kb",
        "main_kb_final_doc_sources", "main_kb_final_similarity_scores",
        "main_kb_final_rerank_scores", "main_kb_final_retrieval_details",
        "main_kb_processing_time",
        
        # 症状知识库结果
        "symptom_kb_answer", "symptom_kb_final_prompt", "symptom_kb_initial_retrieved_count",
        "symptom_kb_initial_doc_sources", "symptom_kb_initial_similarity_scores",
        "symptom_kb_initial_retrieval_details", "symptom_kb_has_reference",
        "symptom_kb_final_retrieved_count", "symptom_kb_use_rerank", "symptom_kb_use_kb",
        "symptom_kb_final_doc_sources", "symptom_kb_final_similarity_scores",
        "symptom_kb_final_rerank_scores", "symptom_kb_final_retrieval_details",
        "symptom_kb_processing_time",
    ]
    
    # 确保所有列都存在
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    
    df = df[columns]
    excel_path = os.path.join(output_dir, f"{output_prefix}_summary.xlsx")
    
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='查询结果', index=False)
            
            # 设置列宽和格式
            worksheet = writer.sheets['查询结果']
            from openpyxl.styles import Alignment
            wrap_alignment = Alignment(wrap_text=True, vertical='top')
            
            for row in worksheet.iter_rows(min_row=2):
                for cell in row:
                    cell.alignment = wrap_alignment
        
        if is_final:
            logger.info(f"最终结果已保存到: {excel_path}")
        else:
            logger.info(f"中间结果已保存到: {excel_path}")
        return excel_path
    except Exception as e:
        logger.error(f"保存Excel文件失败: {str(e)}")
        return None
