import os
import time
import argparse
import logging
import torch
import pandas as pd
from typing import List, Dict, Any

from config import Config
from medical_processor import MedicalProcessor
from utils import (load_terms_from_file, load_question_types, load_progress, 
                   save_progress, save_excel_results)

def setup_logging(log_level: str = "INFO"):
    """设置日志配置"""
    log_directory = "log"
    os.makedirs(log_directory, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_directory, "medical_processor.log"))
        ]
    )
    
    # 禁用 Hugging Face tokenizers 的并行处理警告
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='医学检索批处理工具（支持重排序RAG）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--api_key', type=str, default="sk-f139cfa5682b44c4b65cc203dd5e53ce", help='大模型API密钥')
    parser.add_argument('--api_type', type=str, default="deepseek", 
                        choices=["deepseek", "doubao", "qianwen", "ernie"],
                        help='要使用的大模型API类型')
    parser.add_argument('--batch_size', type=int, default=10, help='批次大小')
    parser.add_argument('--embedding_model', type=str, default="/mnt/fang/qwen3_embedding_0.6B_finetuned_inbatch_pro_earlystop", help='嵌入模型路径')
    parser.add_argument('--final_k', type=int, default=5, help='最终使用文档数量')
    parser.add_argument('--gpu_id', type=int, default=None, help='GPU设备ID')
    parser.add_argument('--initial_k', type=int, default=20, help='初始检索文档数量')
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                        help='日志级别')
    parser.add_argument('--model', type=str, help='要使用的大模型名称')
    parser.add_argument('--multi_shard', action='store_true', default=True, help='启用多分片存储模式')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--output_prefix', type=str, default="medical_rag_results", help='输出文件前缀')
    parser.add_argument('--question_type', type=str, help='要查询的单个问题类型')
    parser.add_argument('--question_types', type=str, help='包含问题类型列表的文件路径')
    parser.add_argument('--reranker_model', type=str, default="Qwen/Qwen3-Reranker-0.6B", help='重排序模型路径')
    parser.add_argument('--reset', action='store_true', help='重置进度从头开始')
    parser.add_argument('--retry_failed', action='store_true', help='重试失败的查询')
    parser.add_argument('--symptom_path', type=str, default="/home/fangyi1/Deepseek项目/RAG项目/medical_retrieval/MedSE2RAG/data/vector_database/0.6B/Definition_vector_store/", help='症状向量库的路径（可选）')
    parser.add_argument('--task_type', type=str, default="symptom_extraction", 
                        choices=["normal", "symptom_extraction", "definition", "treatment"],
                        help='任务类型')
    parser.add_argument('--term', type=str, help='要查询的单个医学术语')
    parser.add_argument('--term_file', type=str, help='包含医学术语列表的文件路径')
    parser.add_argument('--use_rerank', action='store_true', default=True, help='启用重排序功能')
    parser.add_argument('--vector_path', type=str, required=True, default="/home/fangyi1/Deepseek项目/RAG项目/medical_retrieval/data/vector_database/all_pdfs_vector_db_qwen3/", help='主向量库的根路径')

    
    return parser.parse_args()

def create_config_from_args(args: argparse.Namespace) -> Config:
    """从命令行参数创建配置对象"""
    # 设置GPU
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        device = f"cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 设置默认模型
    if not args.model:
        default_models = {
            "deepseek": "deepseek-chat", 
            "doubao": "doubao-seed-1-6-thinking-250715",
            "qianwen": "qwen-turbo", 
            "ernie": "ernie-3.5-8k"
        }
        args.model = default_models.get(args.api_type, "unknown-model")
    
    return Config(
        api_type=args.api_type,
        api_key=args.api_key,
        model=args.model,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        use_rerank=args.use_rerank,
        gpu_id=args.gpu_id,
        device=device,
        initial_k=args.initial_k,
        final_k=args.final_k,
        batch_size=args.batch_size,
        vector_path=args.vector_path,
        symptom_path=args.symptom_path,
        output_dir=os.path.abspath(args.output_dir),
        output_prefix=args.output_prefix
    )

def validate_inputs(args: argparse.Namespace, logger: logging.Logger) -> bool:
    """验证输入参数"""
    # 参数互斥性校验
    if (args.term and args.term_file) or (not args.term and not args.term_file):
        logger.error("请提供 '--term' 或 '--term_file' 中的一个，不能同时提供或都不提供")
        return False
    
    if (args.question_type and args.question_types) or (not args.question_type and not args.question_types):
        logger.error("请提供 '--question_type' 或 '--question_types' 中的一个，不能同时提供或都不提供")
        return False
    
    # 路径存在性校验
    if not os.path.exists(args.vector_path):
        logger.error(f"主向量库路径不存在: {args.vector_path}")
        return False
    
    if args.symptom_path and not os.path.exists(args.symptom_path):
        logger.error(f"症状向量库路径不存在: {args.symptom_path}")
        return False
    
    # 创建输出目录
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"创建输出目录失败: {str(e)}")
        return False
    
    return True

def load_input_data(args: argparse.Namespace, logger: logging.Logger) -> tuple:
    """加载输入数据"""
    logger.info("加载输入数据")
    
    # 加载术语
    terms = load_terms_from_file(args.term_file, args.term)
    if not terms:
        logger.error("未能加载术语")
        return None, None
    
    # 加载问题类型
    question_types = load_question_types(args.question_types, args.question_type)
    if not question_types:
        logger.error("未能加载问题类型")
        return None, None
    
    return terms, question_types

def generate_queries(terms: List[str], question_types: List[str]) -> List[Dict[str, Any]]:
    """生成查询任务组合"""
    all_queries = []
    query_id = 1
    for term in terms:
        for q_type in question_types:
            all_queries.append({"term": term, "question_type": q_type, "id": query_id})
            query_id += 1
    return all_queries

def determine_queries_to_process(all_queries: List[Dict], args: argparse.Namespace, 
                               progress: Dict, logger: logging.Logger) -> List[Dict]:
    """确定需要处理的查询列表"""
    completed_ids = set(progress.get("completed_ids", []))
    
    if args.retry_failed and not args.reset:
        logger.info("重试失败模式：加载历史结果以重试失败的查询")
        try:
            excel_path = os.path.join(args.output_dir, f"{args.output_prefix}_summary.xlsx")
            df_history = pd.read_excel(excel_path)
            failed_ids_history = set(df_history[
                df_history["overall_status"].isin(["failed", "inconsistent", "inconsistent_no_meta_eval"])
            ]["id"].tolist())
            
            queries_to_process = [q for q in all_queries if q["id"] in failed_ids_history]
            logger.info(f"发现 {len(failed_ids_history)} 个失败或不一致的查询，将重试")
        except Exception as e:
            logger.warning(f"加载历史结果失败: {str(e)}，将处理所有未完成查询")
            queries_to_process = [q for q in all_queries if q["id"] not in completed_ids]
    else:
        queries_to_process = [q for q in all_queries if q["id"] not in completed_ids]
    
    logger.info(f"将处理 {len(queries_to_process)} 个查询")
    return queries_to_process

def load_historical_results(args: argparse.Namespace, queries_to_process: List[Dict], 
                          logger: logging.Logger) -> List[Dict[str, Any]]:
    """加载历史结果"""
    if args.reset:
        return []
    
    logger.info("加载历史结果以便合并")
    excel_path_history = os.path.join(args.output_dir, f"{args.output_prefix}_summary.xlsx")
    
    if not os.path.exists(excel_path_history):
        return []
    
    try:
        df_existing = pd.read_excel(excel_path_history)
        # 过滤掉待重新处理的查询
        existing_ids_to_keep = {
            row["id"] for idx, row in df_existing.iterrows()
            if row["id"] not in {q["id"] for q in queries_to_process}
        }
        existing_results_filtered = [
            row.to_dict() for idx, row in df_existing.iterrows()
            if row["id"] in existing_ids_to_keep
        ]
        logger.info(f"已加载 {len(existing_results_filtered)} 条历史结果")
        return existing_results_filtered
    except Exception as e:
        logger.warning(f"加载历史结果失败: {str(e)}")
        return []

def process_batch(processor: MedicalProcessor, batch_queries: List[Dict], 
                 task_type: str, batch_num: int, total_batches: int, 
                 logger: logging.Logger) -> tuple:
    """处理单个批次"""
    logger.info(f"--- 批次 {batch_num}/{total_batches} 开始处理 ({len(batch_queries)}个查询) ---")
    
    batch_results = []
    batch_completed_ids = []
    
    for j, query in enumerate(batch_queries, 1):
        logger.info(f"[{batch_num}-{j}/{len(batch_queries)}] 启动处理查询 ID {query['id']}: '{query['term']}的{query['question_type']}'")
        
        result = processor.process_single_query(
            query["term"], query["question_type"], 
            query["id"], task_type
        )
        
        batch_results.append(result)
        
        # 记录处理状态
        status_icon = "✅" if result["overall_status"].startswith("completed") else "⚠️" if result["overall_status"].startswith("partially") or result["overall_status"].startswith("inconsistent") else "❌"
        kb_status_info = _get_kb_status_info(result)
        
        logger.info(f"  {status_icon} ID {query['id']} ({result['total_processing_time']}s) -> 总状态: {result['overall_status']} ({kb_status_info})")
        
        # 确定是否应该标记为完成
        if _should_mark_as_completed(result):
            batch_completed_ids.append(query["id"])
    
    return batch_results, batch_completed_ids

def _get_kb_status_info(result: Dict[str, Any]) -> str:
    """获取知识库状态信息"""
    kb_status_info = ""
    
    if result.get("main_kb_use_kb"):
        kb_status_info += f"主知识库: {'✅' if result.get('main_kb_answer') != '模型调用失败或无响应' and '未加载' not in result.get('main_kb_answer', '') else '❌'}"
    else:
        kb_status_info += "主知识库: (未启用)"
    
    if result.get("symptom_kb_use_kb"):
        kb_status_info += f", 症状知识库: {'✅' if result.get('symptom_kb_answer') != '模型调用失败或无响应' and '未加载' not in result.get('symptom_kb_answer', '') else '❌'}"
    else:
        kb_status_info += ", 症状知识库: (未启用)"
    
    return kb_status_info

def _should_mark_as_completed(result: Dict[str, Any]) -> bool:
    """判断查询是否应该标记为完成"""
    # 成功完成的状态
    success_statuses = ["completed", "completed_with_meta_eval", "partially_completed"]
    
    if result["overall_status"] in success_statuses:
        return True
    
    # 如果是失败但由于知识库未加载导致的，也标记为完成（避免无意义重试）
    if result["overall_status"] == "failed":
        main_answer = result.get("main_kb_answer", "")
        symptom_answer = result.get("symptom_kb_answer", "")
        if "主知识库未加载" in main_answer and "症状知识库未加载" in symptom_answer:
            return True
    
    return False

def generate_final_statistics(all_results: List[Dict[str, Any]]) -> Dict[str, int]:
    """生成最终统计信息"""
    stats = {
        'completed': sum(1 for r in all_results if r.get('overall_status') == 'completed'),
        'completed_with_meta_eval': sum(1 for r in all_results if r.get('overall_status') == 'completed_with_meta_eval'),
        'partially_completed': sum(1 for r in all_results if r.get('overall_status') == 'partially_completed'),
        'inconsistent_no_meta_eval': sum(1 for r in all_results if r.get('overall_status') == 'inconsistent_no_meta_eval'),
        'failed': sum(1 for r in all_results if r.get('overall_status') == 'failed'),
        'main_kb_used': sum(1 for r in all_results if r.get('main_kb_use_kb')),
        'symptom_kb_used': sum(1 for r in all_results if r.get('symptom_kb_use_kb')),
        'main_kb_reranked': sum(1 for r in all_results if r.get('main_kb_use_kb') and r.get('main_kb_use_rerank')),
        'symptom_kb_reranked': sum(1 for r in all_results if r.get('symptom_kb_use_kb') and r.get('symptom_kb_use_rerank')),
        'total': len(all_results)
    }
    return stats

def print_final_report(stats: Dict[str, int], total_time: float, logger: logging.Logger):
    """打印最终统计报告"""
    logger.info("=" * 50)
    logger.info("          批处理任务完成          ")
    logger.info("=" * 50)
    logger.info(f"总耗时: {total_time:.1f} 秒")
    logger.info("结果统计:")
    logger.info(f"  完全成功: {stats['completed']} 个查询")
    logger.info(f"  元评估后成功整合: {stats['completed_with_meta_eval']} 个查询")
    logger.info(f"  部分成功: {stats['partially_completed']} 个查询")
    logger.info(f"  不一致但元评估失败: {stats['inconsistent_no_meta_eval']} 个查询")
    logger.info(f"  完全失败: {stats['failed']} 个查询")
    logger.info(f"  主知识库被使用: {stats['main_kb_used']} 个")
    logger.info(f"  主知识库启用重排序: {stats['main_kb_reranked']} 个")
    logger.info(f"  症状知识库被使用: {stats['symptom_kb_used']} 个")
    logger.info(f"  症状知识库启用重排序: {stats['symptom_kb_reranked']} 个")
    logger.info(f"  总计记录的查询: {stats['total']} 个")
    logger.info("=" * 50)

def main():
    """主程序入口"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("医学检索批处理工具启动")
    logger.info("=" * 50)
    
    # 解析参数并设置日志
    args = parse_arguments()
    setup_logging(args.log_level)
    
    # 验证输入参数
    if not validate_inputs(args, logger):
        return
    
    # 创建配置对象
    config = create_config_from_args(args)
    logger.info(f"使用设备: {config.device}")
    logger.info(f"使用模型: {config.model}")
    
    # 初始化处理器
    processor = MedicalProcessor(config)
    if not processor.initialize(is_multi_shard=args.multi_shard):
        logger.error("MedicalProcessor 初始化失败，程序退出")
        return
    
    # 加载输入数据
    terms, question_types = load_input_data(args, logger)
    if not terms or not question_types:
        return
    
    # 生成查询任务
    all_queries = generate_queries(terms, question_types)
    total_queries = len(all_queries)
    logger.info(f"总查询数: {total_queries} ({len(terms)}个术语 × {len(question_types)}个问题类型)")
    
    # 加载处理进度
    logger.info("检查历史处理进度")
    if args.reset:
        logger.info("重置模式：忽略历史进度，从头开始")
        progress = {"completed_ids": [], "total_queries": total_queries, "batches_completed": 0}
        completed_ids = set()
    else:
        progress = load_progress(args.output_dir, args.output_prefix)
        completed_ids = set(progress.get("completed_ids", []))
        logger.info(f"已加载 {len(completed_ids)} 个已完成查询的历史记录")
    
    # 确定待处理查询
    queries_to_process = determine_queries_to_process(all_queries, args, progress, logger)
    if not queries_to_process:
        logger.info("所有查询已完成，或没有待处理的查询")
        return
    
    # 加载历史结果
    all_results_memory = load_historical_results(args, queries_to_process, logger)
    
    # 开始批量处理
    logger.info("=" * 50)
    logger.info("开始批量处理查询")
    logger.info(f"待处理: {len(queries_to_process)} / 总计: {total_queries} 个查询")
    logger.info("=" * 50)
    
    total_start_time = time.time()
    batch_num = progress["batches_completed"] + 1
    total_batches = (len(queries_to_process) + config.batch_size - 1) // config.batch_size
    
    try:
        # 分批处理查询
        for i in range(0, len(queries_to_process), config.batch_size):
            batch_queries = queries_to_process[i:i + config.batch_size]
            
            batch_results, batch_completed_ids = process_batch(
                processor, batch_queries, args.task_type, 
                batch_num, total_batches, logger
            )
            
            # 更新进度
            completed_ids.update(batch_completed_ids)
            progress["completed_ids"] = list(completed_ids)
            progress["batches_completed"] = batch_num
            
            # 保存进度和结果
            logger.info("保存批次进度和中间结果")
            save_progress(args.output_dir, args.output_prefix, progress)
            all_results_memory.extend(batch_results)
            save_excel_results(args.output_dir, args.output_prefix, all_results_memory)
            
            batch_num += 1

    except KeyboardInterrupt:
        logger.warning("用户中断处理进程，正在保存当前进度")
        save_progress(args.output_dir, args.output_prefix, progress)
        save_excel_results(args.output_dir, args.output_prefix, all_results_memory)
        logger.info("进度和结果已保存")
        return
    
    # 保存最终结果
    logger.info("所有批次处理完毕，保存最终结果")
    save_excel_results(args.output_dir, args.output_prefix, all_results_memory, is_final=True)
    
    # 生成并打印最终统计报告
    total_time = time.time() - total_start_time
    stats = generate_final_statistics(all_results_memory)
    print_final_report(stats, total_time, logger)

if __name__ == "__main__":
    main()
