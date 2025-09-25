import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

class QueryRecorder:
    """查询记录管理器，用于强化学习数据收集"""
    
    def __init__(self, records_file="query_records.jsonl"):
        self.records_file = records_file
        self.ensure_file_exists()
    
    def ensure_file_exists(self):
        """确保记录文件存在"""
        if not os.path.exists(self.records_file):
            with open(self.records_file, 'w', encoding='utf-8') as f:
                pass  # 创建空文件
    
    def save_query_record(self, 
                         user_input: str,
                         question_type: str, 
                         task_type: str,
                         result: Dict[str, Any],
                         processing_time: float,
                         feedback: Optional[Dict[str, Any]] = None) -> str:
        """
        保存查询记录
        返回记录ID
        """
        record_id = f"query_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        record = {
            "record_id": record_id,
            "timestamp": datetime.now().isoformat(),
            "query_data": {
                "user_input": user_input,
                "question_type": question_type,
                "task_type": task_type,
                "processing_time": processing_time
            },
            "system_output": {
                "main_kb_answer": result.get('main_kb_answer', ''),
                "symptom_kb_answer": result.get('symptom_kb_answer', ''),
                "overall_answer": result.get('overall_answer', ''),
                "main_kb_has_reference": result.get('main_kb_has_reference', False),
                "symptom_kb_has_reference": result.get('symptom_kb_has_reference', False),
                "main_kb_final_retrieved_count": result.get('main_kb_final_retrieved_count', 0),
                "symptom_kb_final_retrieved_count": result.get('symptom_kb_final_retrieved_count', 0),
                "overall_status": result.get('overall_status', 'unknown')
            },
            "performance_metrics": {
                "main_kb_processing_time": result.get('main_kb_processing_time', 0),
                "symptom_kb_processing_time": result.get('symptom_kb_processing_time', 0),
                "total_processing_time": result.get('total_processing_time', 0),
                "main_kb_use_rerank": result.get('main_kb_use_rerank', False),
                "symptom_kb_use_rerank": result.get('symptom_kb_use_rerank', False)
            },
            "user_feedback": feedback,
            "reinforcement_learning_data": {
                "reward_signal": None,  # 将基于用户反馈计算
                "action_quality": None,  # 系统响应质量评估
                "context_relevance": None,  # 上下文相关性
                "retrieval_effectiveness": {
                    "main_kb_relevance": None,
                    "symptom_kb_relevance": None,
                    "combined_effectiveness": None
                }
            }
        }
        
        # 保存记录
        with open(self.records_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        return record_id
    
    def update_feedback(self, record_id: str, feedback_text: str, rating: int, 
                       additional_feedback: Dict[str, Any] = None):
        """更新用户反馈并计算强化学习信号"""
        
        # 读取所有记录
        records = self.load_all_records()
        
        # 找到对应记录并更新
        updated = False
        for record in records:
            if record.get('record_id') == record_id:
                # 更新反馈
                record['user_feedback'] = {
                    "feedback_text": feedback_text,
                    "rating": rating,
                    "feedback_timestamp": datetime.now().isoformat(),
                    "additional_feedback": additional_feedback or {}
                }
                
                # 计算强化学习信号
                record['reinforcement_learning_data'].update(
                    self._calculate_rl_signals(record, rating, feedback_text)
                )
                
                updated = True
                break
        
        if updated:
            # 重写整个文件
            with open(self.records_file, 'w', encoding='utf-8') as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        return updated
    
    def _calculate_rl_signals(self, record: Dict[str, Any], rating: int, 
                             feedback_text: str) -> Dict[str, Any]:
        """计算强化学习信号"""
        
        # 基于评分的奖励信号 (归一化到-1到1)
        reward_signal = (rating - 3) / 2.0
        
        # 系统响应质量评估
        action_quality = self._assess_action_quality(record, rating)
        
        # 上下文相关性评估
        context_relevance = self._assess_context_relevance(record, feedback_text)
        
        # 检索有效性评估
        retrieval_effectiveness = self._assess_retrieval_effectiveness(record, rating)
        
        return {
            "reward_signal": reward_signal,
            "action_quality": action_quality,
            "context_relevance": context_relevance,
            "retrieval_effectiveness": retrieval_effectiveness
        }
    
    def _assess_action_quality(self, record: Dict[str, Any], rating: int) -> float:
        """评估系统行为质量"""
        quality_score = rating / 5.0
        
        # 考虑检索成功率
        retrieval_bonus = 0
        if record['system_output']['main_kb_has_reference']:
            retrieval_bonus += 0.1
        if record['system_output']['symptom_kb_has_reference']:
            retrieval_bonus += 0.1
        
        # 考虑处理时间效率
        processing_time = record['performance_metrics']['total_processing_time']
        if processing_time > 0:
            efficiency_bonus = max(0, (10 - processing_time) / 10 * 0.1)
        else:
            efficiency_bonus = 0
        
        return min(1.0, quality_score + retrieval_bonus + efficiency_bonus)
    
    def _assess_context_relevance(self, record: Dict[str, Any], feedback_text: str) -> float:
        """评估上下文相关性"""
        # 简单的关键词匹配评估
        positive_keywords = ['准确', '有用', '详细', '清楚', '满意', '好']
        negative_keywords = ['错误', '不准确', '无关', '不清楚', '不满意', '差']
        
        positive_count = sum(1 for word in positive_keywords if word in feedback_text)
        negative_count = sum(1 for word in negative_keywords if word in feedback_text)
        
        if positive_count + negative_count == 0:
            return 0.5  # 中性
        
        return positive_count / (positive_count + negative_count)
    
    def _assess_retrieval_effectiveness(self, record: Dict[str, Any], rating: int) -> Dict[str, float]:
        """评估检索有效性"""
        main_kb_effectiveness = 0.5
        symptom_kb_effectiveness = 0.5
        
        # 基于是否有检索结果
        if record['system_output']['main_kb_has_reference']:
            main_kb_effectiveness = rating / 5.0
        
        if record['system_output']['symptom_kb_has_reference']:
            symptom_kb_effectiveness = rating / 5.0
        
        combined_effectiveness = (main_kb_effectiveness + symptom_kb_effectiveness) / 2
        
        return {
            "main_kb_relevance": main_kb_effectiveness,
            "symptom_kb_relevance": symptom_kb_effectiveness,
            "combined_effectiveness": combined_effectiveness
        }
    
    def load_all_records(self) -> List[Dict[str, Any]]:
        """加载所有记录"""
        records = []
        try:
            with open(self.records_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        except FileNotFoundError:
            pass
        return records
    
    def get_rl_training_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取强化学习训练数据"""
        records = self.load_all_records()
        
        # 只返回有用户反馈的记录
        rl_data = [
            record for record in records 
            if record.get('user_feedback') and 
               record.get('reinforcement_learning_data', {}).get('reward_signal') is not None
        ]
        
        # 按时间倒序排列
        rl_data.sort(key=lambda x: x['timestamp'], reverse=True)
        
        if limit:
            rl_data = rl_data[:limit]
        
        return rl_data
    
    def export_training_data(self, output_file: str = "rl_training_data.json"):
        """导出强化学习训练数据"""
        training_data = self.get_rl_training_data()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        return len(training_data)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        records = self.load_all_records()
        
        if not records:
            return {"total_queries": 0}
        
        # 基本统计
        total_queries = len(records)
        feedback_queries = len([r for r in records if r.get('user_feedback')])
        
        # 评分统计
        ratings = [
            r['user_feedback']['rating'] 
            for r in records 
            if r.get('user_feedback', {}).get('rating')
        ]
        
        # 强化学习数据统计
        rl_records = self.get_rl_training_data()
        
        avg_reward = 0
        if rl_records:
            rewards = [
                r['reinforcement_learning_data']['reward_signal'] 
                for r in rl_records
            ]
            avg_reward = sum(rewards) / len(rewards)
        
        return {
            "total_queries": total_queries,
            "feedback_count": feedback_queries,
            "feedback_rate": feedback_queries / total_queries if total_queries > 0 else 0,
            "average_rating": sum(ratings) / len(ratings) if ratings else 0,
            "rating_distribution": {i: ratings.count(i) for i in range(1, 6)},
            "rl_training_samples": len(rl_records),
            "average_reward_signal": avg_reward,
            "avg_processing_time": sum(r.get('query_data', {}).get('processing_time', 0) for r in records) / len(records)
        }
