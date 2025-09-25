import streamlit as st
import pandas as pd
import time
import os
import sqlite3
import json
from datetime import datetime
from config import Config
from medical_processor import MedicalProcessor

# 页面配置
st.set_page_config(
    page_title="医学症状提取系统",
    page_icon="🏥",
    layout="wide"
)

# 查询记录管理类
class QueryRecorder:
    """查询记录管理器，用于强化学习数据收集"""
    
    def __init__(self, records_file="medical_query_records.jsonl"):
        self.records_file = records_file
        self.ensure_file_exists()
    
    def ensure_file_exists(self):
        """确保记录文件存在"""
        if not os.path.exists(self.records_file):
            with open(self.records_file, 'w', encoding='utf-8') as f:
                pass  # 创建空文件
    
    def save_query_record(self, user_input, question_type, task_type, result, processing_time):
        """保存查询记录并返回记录ID"""
        record_id = f"query_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        record = {
            "record_id": record_id,
            "timestamp": datetime.now().isoformat(),
            "query_info": {
                "user_input": user_input,
                "question_type": question_type,
                "task_type": task_type,
                "processing_time": processing_time
            },
            "system_response": {
                "main_kb_answer": result.get('main_kb_answer', ''),
                "symptom_kb_answer": result.get('symptom_kb_answer', ''),
                "overall_answer": result.get('overall_answer', ''),
                "overall_status": result.get('overall_status', 'unknown'),
                "main_kb_has_reference": result.get('main_kb_has_reference', False),
                "symptom_kb_has_reference": result.get('symptom_kb_has_reference', False),
                "main_kb_final_retrieved_count": result.get('main_kb_final_retrieved_count', 0),
                "symptom_kb_final_retrieved_count": result.get('symptom_kb_final_retrieved_count', 0),
                "main_kb_processing_time": result.get('main_kb_processing_time', 0),
                "symptom_kb_processing_time": result.get('symptom_kb_processing_time', 0),
                # 添加症状提取结果
                "extracted_symptoms": result.get('overall_pure_symptoms', ''),
                "symptom_consistent": result.get('is_symptom_consistent', ''),
                "total_processing_time": result.get('total_processing_time', 0)
            },
            "user_feedback": None,
            "reinforcement_learning": {
                "reward_signal": None,
                "quality_score": None,
                "retrieval_effectiveness": None,
                "user_satisfaction": None
            }
        }
        
        # 保存记录
        with open(self.records_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        return record_id
    
    def update_feedback(self, record_id, feedback_text, rating, detailed_feedback=None):
        """更新用户反馈并计算强化学习信号"""
        records = self.load_all_records()
        updated = False
        
        for i, record in enumerate(records):
            if record.get('record_id') == record_id:
                # 更新反馈
                record['user_feedback'] = {
                    "feedback_text": feedback_text,
                    "rating": rating,
                    "detailed_feedback": detailed_feedback or {},
                    "feedback_timestamp": datetime.now().isoformat()
                }
                
                # 计算强化学习信号
                reward_signal = self._calculate_reward_signal(record, rating)
                quality_score = self._calculate_quality_score(record, rating, feedback_text)
                retrieval_effectiveness = self._calculate_retrieval_effectiveness(record, rating)
                
                record['reinforcement_learning'] = {
                    "reward_signal": reward_signal,
                    "quality_score": quality_score,
                    "retrieval_effectiveness": retrieval_effectiveness,
                    "user_satisfaction": rating / 5.0
                }
                
                records[i] = record # 更新列表中的记录
                updated = True
                break
        
        if updated:
            # 重写整个文件
            with open(self.records_file, 'w', encoding='utf-8') as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            return True
        return False
    
    def _calculate_reward_signal(self, record, rating):
        """计算奖励信号 (-1 到 1 之间)"""
        # 基础奖励基于用户评分
        base_reward = (rating - 3) / 2.0
        
        # 根据系统性能调整
        performance_bonus = 0
        if record['system_response']['main_kb_has_reference']:
            performance_bonus += 0.1
        if record['system_response']['symptom_kb_has_reference']:
            performance_bonus += 0.1
        
        # 处理时间惩罚
        processing_time = record['query_info']['processing_time']
        time_penalty = max(0, (processing_time - 5) * 0.02)  # 超过5秒开始惩罚
        
        return min(1.0, max(-1.0, base_reward + performance_bonus - time_penalty))
    
    def _calculate_quality_score(self, record, rating, feedback_text):
        """计算响应质量分数"""
        quality = rating / 5.0
        
        # 文本情感分析（简化版）
        positive_keywords = ['准确', '有用', '详细', '清楚', '满意', '好', '不错', '完整']
        negative_keywords = ['错误', '不准确', '无关', '不清楚', '不满意', '差', '无用', '不完整']
        
        positive_count = sum(1 for word in positive_keywords if word in feedback_text)
        negative_count = sum(1 for word in negative_keywords if word in feedback_text)
        
        if positive_count + negative_count > 0:
            sentiment_score = positive_count / (positive_count + negative_count)
            quality = (quality + sentiment_score) / 2
        
        return quality
    
    def _calculate_retrieval_effectiveness(self, record, rating):
        """计算检索有效性"""
        main_effective = record['system_response']['main_kb_has_reference']
        symptom_effective = record['system_response']['symptom_kb_has_reference']
        
        effectiveness = 0
        if main_effective and symptom_effective:
            effectiveness = rating / 5.0
        elif main_effective or symptom_effective:
            effectiveness = (rating / 5.0) * 0.7
        else:
            effectiveness = (rating / 5.0) * 0.3
        
        return effectiveness
    
    def load_all_records(self):
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
    
    def get_rl_training_data(self):
        """获取强化学习训练数据"""
        records = self.load_all_records()
        return [r for r in records if r.get('user_feedback') and r.get('reinforcement_learning', {}).get('reward_signal') is not None]
    
    def export_rl_data(self, output_file="rl_training_data.json"):
        """导出强化学习数据"""
        rl_data = self.get_rl_training_data()
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(rl_data, f, ensure_ascii=False, indent=2)
        return len(rl_data)

# 初始化查询记录器
# 移除 @st.cache_resource 装饰器，确保每次都能获取最新实例
def get_query_recorder_instance():
    return QueryRecorder()

recorder = get_query_recorder_instance()

# 数据库初始化（兼容旧版本）
def init_database():
    """初始化SQLite数据库"""
    conn = sqlite3.connect('query_logs.db')
    cursor = conn.cursor()
    
    # 创建查询记录表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_input TEXT NOT NULL,
            question_type TEXT NOT NULL,
            task_type TEXT NOT NULL,
            result_data TEXT NOT NULL,
            processing_time REAL,
            status TEXT,
            main_kb_used BOOLEAN,
            symptom_kb_used BOOLEAN,
            user_feedback TEXT,
            feedback_rating INTEGER,
            feedback_timestamp TEXT
        )
    ''')
    
    # 检查并添加新列
    cursor.execute("PRAGMA table_info(query_logs)")
    columns = [column[1] for column in cursor.fetchall()]
    
    new_columns_with_types = {
        'record_id': 'TEXT',
        'extracted_symptoms': 'TEXT',
        'symptom_consistent': 'TEXT'
    }
    
    for col_name, col_type in new_columns_with_types.items():
        if col_name not in columns:
            try:
                cursor.execute(f'ALTER TABLE query_logs ADD COLUMN {col_name} {col_type}')
                print(f"Added column {col_name} to query_logs table.")
            except sqlite3.OperationalError as e:
                print(f"Error adding column {col_name}: {e}") # Debugging
    
    conn.commit()
    conn.close()

def save_query_log(user_input, question_type, task_type, result, processing_time, record_id):
    """保存查询记录到数据库"""
    conn = sqlite3.connect('query_logs.db')
    cursor = conn.cursor()
    
    # 从result获取extracted_symptoms和symptom_consistent
    extracted_symptoms = result.get('overall_pure_symptoms', '')
    symptom_consistent = result.get('is_symptom_consistent', '')

    try:
        cursor.execute('''
            INSERT INTO query_logs 
            (timestamp, user_input, question_type, task_type, result_data, processing_time, status, 
             main_kb_used, symptom_kb_used, record_id, extracted_symptoms, symptom_consistent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            user_input,
            question_type,
            task_type,
            json.dumps(result, ensure_ascii=False),
            processing_time,
            result.get('overall_status', 'unknown'),
            result.get('main_kb_has_reference', False),
            result.get('symptom_kb_has_reference', False),
            record_id,
            extracted_symptoms,
            symptom_consistent
        ))
    except sqlite3.OperationalError as e:
        print(f"数据库插入失败，可能缺少列或数据类型不匹配: {e}") # Debugging
        # 如果某些列不存在，使用基本插入
        cursor.execute('''
            INSERT INTO query_logs 
            (timestamp, user_input, question_type, task_type, result_data, processing_time, status, main_kb_used, symptom_kb_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            user_input,
            question_type,
            task_type,
            json.dumps(result, ensure_ascii=False),
            processing_time,
            result.get('overall_status', 'unknown'),
            result.get('main_kb_has_reference', False),
            result.get('symptom_kb_has_reference', False)
        ))
    
    query_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return query_id

def save_user_feedback(query_id, record_id, feedback_text, rating, detailed_feedback=None):
    """保存用户反馈到数据库和JSONL文件"""
    try:
        # 更新数据库
        conn = sqlite3.connect('query_logs.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE query_logs 
            SET user_feedback = ?, feedback_rating = ?, feedback_timestamp = ?
            WHERE id = ?
        ''', (feedback_text, rating, datetime.now().isoformat(), query_id))
        
        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()
        
        print(f"数据库更新：查询ID={query_id}, 影响行数={rows_affected}, 评分={rating}, 反馈文本长度={len(feedback_text) if feedback_text else 0}") # 增加调试输出
        
        # 更新JSONL文件
        jsonl_success = False
        if record_id and record_id != 'N/A':
            recorder_instance = get_query_recorder_instance() # 确保获取到最新的实例
            jsonl_success = recorder_instance.update_feedback(record_id, feedback_text, rating, detailed_feedback)
            print(f"JSONL更新结果: {jsonl_success}")
        
        return rows_affected > 0 or jsonl_success
        
    except Exception as e:
        print(f"保存反馈时出错: {e}")
        return False

def get_query_logs(limit=50, force_refresh=False):
    """获取查询记录"""
    # 使用缓存键，但允许强制刷新
    cache_key = f'query_logs_cache_{limit}'
    
    if not force_refresh and cache_key in st.session_state:
        return st.session_state[cache_key]
    
    conn = sqlite3.connect('query_logs.db')
    cursor = conn.cursor()
    
    # 检查表结构
    cursor.execute("PRAGMA table_info(query_logs)")
    columns = [column[1] for column in cursor.fetchall()]
    
    # 构建查询语句
    base_columns = "id, timestamp, user_input, question_type, processing_time, status, user_feedback, feedback_rating"
    
    select_columns_list = [base_columns]
    
    if 'record_id' in columns:
        select_columns_list.append("CASE WHEN record_id IS NULL THEN 'N/A' ELSE record_id END as record_id")
    else:
        select_columns_list.append("'N/A' as record_id")
    
    if 'extracted_symptoms' in columns:
        select_columns_list.append("CASE WHEN extracted_symptoms IS NULL THEN '' ELSE extracted_symptoms END as extracted_symptoms")
    else:
        select_columns_list.append("'' as extracted_symptoms")
    
    if 'symptom_consistent' in columns:
        select_columns_list.append("CASE WHEN symptom_consistent IS NULL THEN '' ELSE symptom_consistent END as symptom_consistent")
    else:
        select_columns_list.append("'' as symptom_consistent")
    
    select_clause = ", ".join(select_columns_list)
    
    cursor.execute(f'''
        SELECT {select_clause}
        FROM query_logs 
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    # 缓存结果
    st.session_state[cache_key] = rows
    
    return rows

def export_to_excel():
    """导出查询记录到Excel文件"""
    logs = get_query_logs(1000, force_refresh=True)  # 强制刷新获取最新数据
    
    if not logs:
        return None
    
    # 转换为DataFrame
    df_data = []
    # 假设 get_query_logs 返回的顺序是 id, timestamp, user_input, question_type, processing_time, status, user_feedback, feedback_rating, record_id, extracted_symptoms, symptom_consistent
    for log in logs:
        df_data.append({
            "ID": log[0],
            "时间": log[1][:19],
            "用户输入": log[2],
            "问题类型": log[3],
            "处理时间(秒)": log[4],
            "状态": log[5],
            "用户反馈": log[6] if log[6] else "",
            "反馈评分": log[7] if log[7] else "",
            "记录ID": log[8],
            "提取症状": log[9], # 从log[9]获取
            "症状一致性": log[10] # 从log[10]获取
        })
    
    df = pd.DataFrame(df_data)
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"medical_query_logs_{timestamp}.xlsx"
    
    # 保存Excel文件
    df.to_excel(filename, index=False, engine='openpyxl')
    return filename

# 初始化数据库
init_database()

# 初始化session state
if 'processor' not in st.session_state:
    st.session_state.processor = None
    st.session_state.initialized = False
if 'current_query_id' not in st.session_state:
    st.session_state.current_query_id = None
if 'current_record_id' not in st.session_state:
    st.session_state.current_record_id = None

def initialize_system(config, multi_shard):
    """初始化医学处理系统"""
    if st.session_state.processor is None:
        with st.spinner('正在初始化系统...'):
            try:
                st.write("🔄 开始初始化...")
                processor = MedicalProcessor(config)
                st.write("✅ MedicalProcessor 创建成功")
                
                # 显示初始化进度
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("正在初始化API客户端...")
                progress_bar.progress(25)
                
                status_text.text("正在初始化重排序模型...")
                progress_bar.progress(50)
                
                status_text.text(f"正在加载向量库... (多分片模式: {multi_shard})")
                progress_bar.progress(75)
                
                success = processor.initialize(is_multi_shard=multi_shard)
                
                if success:
                    progress_bar.progress(100)
                    status_text.text("初始化完成！")
                    
                    # 显示加载状态
                    st.write("📊 向量库加载状态:")
                    try:
                        # 尝试检测是否有有效的向量库
                        # 这里需要根据vector_manager的实际实现来判断是否加载成功
                        # 例如，如果vector_manager维护一个all_vector_stores列表
                        main_loaded = False
                        if hasattr(processor.vector_manager, 'main_vector_store') and processor.vector_manager.main_vector_store is not None:
                             main_loaded = True
                        elif hasattr(processor.vector_manager, 'vector_stores') and len(processor.vector_manager.vector_stores) > 0:
                             main_loaded = True

                        if main_loaded:
                            st.write("✅ 主知识库加载成功")
                        else:
                            st.warning("⚠️ 主知识库加载失败（请在查询后观察是否实际工作正常）") # 修正提示
                    except Exception as e:
                        st.warning(f"⚠️ 无法确定主知识库状态（错误: {e}），请在查询后观察是否实际工作正常")
                    
                    if hasattr(processor.vector_manager, 'symptom_vector_store') and processor.vector_manager.symptom_vector_store:
                        st.write("✅ 症状知识库加载成功")
                    else:
                        st.warning("⚠️ 症状知识库加载失败")
                    
                    st.session_state.processor = processor
                    st.session_state.initialized = True
                    st.write("✅ 系统初始化成功")
                    return True
                else:
                    st.error("❌ 系统初始化失败")
                    return False
                    
            except Exception as e:
                st.error(f"❌ 初始化过程中发生错误: {str(e)}")
                st.write(f"错误类型: {type(e).__name__}")
                import traceback
                st.code(traceback.format_exc())
                return False
    else:
        st.info("✅ 系统已经初始化")
        return True

def main():
    st.title("🏥 医学症状提取系统")
    st.markdown("---")
    
    # 创建标签页，增加强化学习数据标签
    tab1, tab2, tab3, tab4 = st.tabs(["💬 症状查询", "⚙️ 系统配置", "📊 查询记录", "🤖 强化学习数据"])
    
    with tab2:  # 系统配置标签页
        st.header("系统配置")
        
        # API配置
        api_type = st.selectbox(
            "选择API类型", 
            ["deepseek", "doubao", "qianwen", "ernie"]
        )
        
        api_key = st.text_input(
            "API密钥", 
            type="password",
            help="输入您的API密钥"
        )
        
        # 路径配置
        vector_path = st.text_input(
            "主向量库路径",
            value="/home/fangyi1/Deepseek项目/RAG项目/medical_retrieval/MedSE2RAG/data/vector_database/all_pdfs_vector_db_qwen3/",
            help="主知识库的向量存储路径"
        )
        
        symptom_path = st.text_input(
            "症状向量库路径",
            value="/home/fangyi1/Deepseek项目/RAG项目/medical_retrieval/MedSE2RAG/data/vector_database/0.6B/Definition_vector_store/", 
            help="症状知识库的向量存储路径"
        )
        
        # 模型配置
        embedding_model = st.text_input(
            "嵌入模型路径",
            value="/mnt/fang/qwen3_embedding_0.6B_finetuned_inbatch_pro_earlystop",
            help="嵌入模型的路径"
        )
        
        reranker_model = st.text_input(
            "重排序模型路径",
            value="Qwen/Qwen3-Reranker-0.6B",
            help="重排序模型的路径（支持Hugging Face模型名称或本地路径）"
        )
        
        # 高级选项
        with st.expander("高级选项"):
            use_rerank = st.checkbox("启用重排序", value=True)
            multi_shard = st.checkbox("使用多分片向量库", value=True)
            initial_k = st.slider("初始检索数量", 10, 50, 20)
            final_k = st.slider("最终文档数量", 3, 15, 5)
            model_name = st.text_input(
                "API模型名称",
                value="deepseek-chat" if api_type == "deepseek" else "",
                help="指定要使用的具体模型名称"
            )
            task_type = st.selectbox(
                "任务类型",
                ["symptom_extraction", "normal", "definition", "treatment"],
                help="选择处理任务的类型"
            )
        
        # 系统初始化
        if st.button("🔄 初始化系统", type="primary"):
            config = Config(
                api_type=api_type,
                api_key=api_key,
                model=model_name,
                vector_path=vector_path,
                symptom_path=symptom_path,
                embedding_model=embedding_model,
                reranker_model=reranker_model,
                use_rerank=use_rerank,
                multi_shard=multi_shard,
                initial_k=initial_k,
                final_k=final_k
            )

            st.write("配置信息:")
            st.write(f"- 多分片模式: {multi_shard}")
            st.write(f"- 主向量库路径: {vector_path}")
            st.write(f"- API模型: {model_name}")
            st.write(f"- 启用重排序: {use_rerank}")

            # 检查分片文件
            if os.path.exists(vector_path):
                files = os.listdir(vector_path)
                shard_files = [f for f in files if 'index_' in f and '.faiss' in f]
                st.write(f"- 发现分片文件: {len(shard_files)} 个")
                if shard_files:
                    st.write(f"- 分片文件示例: {shard_files[:3]}")
            else:
                st.warning(f"⚠️ 主向量库路径不存在: {vector_path}")
            
            if initialize_system(config, multi_shard):
                st.success("✅ 系统初始化成功！")
            else:
                st.error("❌ 系统初始化失败")
    
    with tab1:  # 症状查询标签页
        if st.session_state.initialized:
            st.header("症状查询")
            
            # 输入区域
            col1, col2 = st.columns([3, 1])
            
            with col1:
                user_input = st.text_area(
                    "请描述您的症状或医学问题：",
                    placeholder="例如：我感觉头晕晕的，很不舒服...",
                    height=100
                )
            
            with col2:
                question_type = st.selectbox(
                    "问题类型",
                    ["症状", "治疗方法", "诊断标准", "预防措施", "并发症"]
                )
                
                if st.button("🔍 开始分析", type="primary", use_container_width=True):
                    if user_input.strip():
                        analyze_symptoms(user_input, question_type, task_type)
                    else:
                        st.warning("请输入症状描述")
            
            # 用户反馈区域
            if st.session_state.current_query_id:
                st.markdown("---")
                st.header("📝 结果反馈")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    feedback_text = st.text_area(
                        "请对本次查询结果进行详细评价：",
                        placeholder="请评价结果的准确性、完整性、有用性等方面，并提出改进建议...",
                        height=120
                    )
                    
                    # 详细反馈选项
                    st.write("**详细评价维度：**")
                    col1a, col1b = st.columns(2)
                    with col1a:
                        accuracy = st.selectbox("准确性", ["很准确", "较准确", "一般", "不太准确", "很不准确"])
                        completeness = st.selectbox("完整性", ["很完整", "较完整", "一般", "不够完整", "很不完整"])
                    with col1b:
                        relevance = st.selectbox("相关性", ["很相关", "较相关", "一般", "不太相关", "很不相关"])
                        usefulness = st.selectbox("有用性", ["很有用", "较有用", "一般", "不太有用", "很无用"])
                
                with col2:
                    rating = st.selectbox(
                        "整体满意度评分：",
                        [5, 4, 3, 2, 1],
                        format_func=lambda x: f"{x}分 - {'非常满意' if x==5 else '满意' if x==4 else '一般' if x==3 else '不满意' if x==2 else '非常不满意'}"
                    )
                    
                    if st.button("💾 提交反馈", type="secondary", use_container_width=True):
                        if feedback_text.strip():
                            detailed_feedback = {
                                "accuracy": accuracy,
                                "completeness": completeness,
                                "relevance": relevance,
                                "usefulness": usefulness
                            }
                            
                            success = save_user_feedback(
                                st.session_state.current_query_id, 
                                st.session_state.current_record_id,
                                feedback_text, 
                                rating, 
                                detailed_feedback
                            )
                            
                            if success:
                                st.success("✅ 反馈已保存并用于系统学习优化，感谢您的宝贵意见！")
                            else:
                                st.warning("⚠️ 反馈保存可能不完整，但已记录到数据库")
                            
                            st.session_state.current_query_id = None
                            st.session_state.current_record_id = None
                            
                            # 强制清除所有相关缓存，以确保统计数据实时更新
                            st.session_state['query_logs_cache_1000'] = None # 清空具体缓存
                            st.session_state['rl_data_cache'] = None # 清空强化学习数据缓存
                            
                            time.sleep(0.5)  # 短暂延迟确保数据写入完成
                            st.rerun()
                        else:
                            st.warning("请输入反馈内容")
        else:
            st.info("👈 请先在「系统配置」标签页中配置并初始化系统")
    
    with tab3:  # 查询记录标签页
        st.header("查询记录与统计")
        
        # 添加刷新按钮
        col_refresh, col_export = st.columns([1, 4])
        with col_refresh:
            if st.button("🔄 刷新数据"):
                st.session_state['query_logs_cache_1000'] = None # 清空具体缓存
                st.session_state['rl_data_cache'] = None # 清空强化学习数据缓存
                st.rerun()
        
        # 统计信息
        logs = get_query_logs(1000)  # 获取更多数据用于统计
        if logs:
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("总查询次数", len(logs))
            
            with col2:
                feedback_count = sum(1 for log in logs if log[6])  # user_feedback不为空
                st.metric("反馈数量", feedback_count)
            
            with col3:
                ratings = [log[7] for log in logs if log[7]]  # feedback_rating不为空
                avg_rating = sum(ratings) / len(ratings) if ratings else 0
                st.metric("平均评分", f"{avg_rating:.1f}")
            
            with col4:
                successful_queries = sum(1 for log in logs if 'completed' in str(log[5]))
                success_rate = (successful_queries / len(logs) * 100) if logs else 0
                st.metric("成功率", f"{success_rate:.1f}%")
            
            with col5:
                if st.button("📁 导出Excel", type="secondary"):
                    try:
                        filename = export_to_excel()
                        if filename:
                            st.success(f"✅ 查询记录已导出到: {filename}")
                            # 提供下载链接
                            with open(filename, "rb") as file:
                                st.download_button(
                                    label="📥 下载Excel文件",
                                    data=file,
                                    file_name=filename,
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                        else:
                            st.warning("暂无数据可导出")
                    except Exception as e:
                        st.error(f"导出失败: {str(e)}")
        
        # 调试信息（可选展开）
        with st.expander("🔧 调试信息"):
            debug_logs = get_query_logs(5, force_refresh=True)
            st.write("最近5条记录的反馈状态:")
            for i, log in enumerate(debug_logs):
                feedback_status = "有反馈" if log[6] else "无反馈"
                rating = log[7] if log[7] else "无评分"
                st.write(f"{i+1}. ID:{log[0]} - {feedback_status} - 评分:{rating} - 时间:{log[1][:19]}")
        
        # 查询记录表
        if logs:
            st.subheader("最近查询记录")
            
            # 转换为DataFrame便于显示
            df_logs = []
            for log in logs[:20]:  # 只显示最近20条
                df_logs.append({
                    "时间": log[1][:19],  # 截取到秒
                    "查询内容": log[2][:50] + "..." if len(log[2]) > 50 else log[2],
                    "问题类型": log[3],
                    "处理时间(秒)": f"{log[4]:.2f}" if log[4] else "N/A",
                    "状态": log[5],
                    "评分": log[7] if log[7] else "未评分",
                    "有反馈": "是" if log[6] else "否",
                    "提取症状": (log[9][:30] + "...") if len(log) > 9 and log[9] and len(log[9]) > 30 else (log[9] if len(log) > 9 else ""),
                    "症状一致性": log[10] if len(log) > 10 and log[10] else "",
                    "记录ID": log[8][:8] + "..." if log[8] and log[8] != 'N/A' else "N/A"
                })
            
            df = pd.DataFrame(df_logs)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("暂无查询记录")
    
    with tab4:  # 强化学习数据标签页
        st.header("强化学习训练数据")
        
        # 获取强化学习数据
        # 强制刷新确保获取最新数据
        recorder_for_rl = get_query_recorder_instance()
        rl_data = recorder_for_rl.get_rl_training_data()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("训练样本数", len(rl_data))
        
        with col2:
            if rl_data:
                avg_reward = sum(r['reinforcement_learning']['reward_signal'] for r in rl_data) / len(rl_data)
                st.metric("平均奖励信号", f"{avg_reward:.3f}")
            else:
                st.metric("平均奖励信号", "0.000")
        
        with col3:
            if st.button("📁 导出RL数据"):
                if rl_data:
                    count = recorder_for_rl.export_rl_data()
                    st.success(f"已导出 {count} 条强化学习数据到 rl_training_data.json")
                else:
                    st.warning("暂无可导出的强化学习数据")
        
        if rl_data:
            # 奖励信号趋势图
            st.subheader("奖励信号趋势")
            reward_data = []
            for i, record in enumerate(rl_data[-50:]):  # 最近50条
                reward_data.append({
                    "序号": i + 1,
                    "奖励信号": record['reinforcement_learning']['reward_signal'],
                    "质量分数": record['reinforcement_learning']['quality_score'],
                    "用户满意度": record['reinforcement_learning']['user_satisfaction']
                })
            
            if reward_data:
                df_rewards = pd.DataFrame(reward_data)
                st.line_chart(df_rewards.set_index("序号"))
            
            # 详细数据表
            st.subheader("详细训练数据")
            table_data = []
            for record in rl_data[-15:]:  # 显示最近15条
                table_data.append({
                    "时间": record['timestamp'][:19],
                    "查询内容": record['query_info']['user_input'][:40] + "...",
                    "用户评分": record['user_feedback']['rating'],
                    "奖励信号": f"{record['reinforcement_learning']['reward_signal']:.3f}",
                    "质量分数": f"{record['reinforcement_learning']['quality_score']:.3f}",
                    "检索效果": f"{record['reinforcement_learning']['retrieval_effectiveness']:.3f}",
                    "用户满意度": f"{record['reinforcement_learning']['user_satisfaction']:.3f}"
                })
            
            if table_data:
                df_rl = pd.DataFrame(table_data)
                st.dataframe(df_rl, use_container_width=True)
        else:
            st.info("暂无强化学习训练数据。请先进行查询并提供反馈以生成训练数据。")

def analyze_symptoms(user_input, question_type, task_type):
    """分析症状"""
    with st.spinner('正在分析，请稍候...'):
        start_time = time.time()
        
        try:
            result = st.session_state.processor.process_single_query(
                term=user_input,
                question_type=question_type,
                query_id=1,
                task_type=task_type
            )
            
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            
            # 保存到JSONL文件
            recorder_instance = get_query_recorder_instance()
            record_id = recorder_instance.save_query_record(user_input, question_type, task_type, result, processing_time)
            
            # 保存到数据库
            query_id = save_query_log(user_input, question_type, task_type, result, processing_time, record_id)
            
            st.session_state.current_query_id = query_id
            st.session_state.current_record_id = record_id
            
            # 显示结果
            st.success(f"✅ 分析完成！耗时：{processing_time:.2f}秒 (记录ID: {record_id[:12]}...)")
            display_result(result)
            
        except Exception as e:
            st.error(f"❌ 分析失败：{str(e)}")
            import traceback
            st.code(traceback.format_exc())

def display_result(result):
    """显示分析结果"""
    # 总体状态
    status_map = {
        "completed": "🟢 完全成功",
        "completed_with_meta_eval": "🟡 元评估完成", 
        "partially_completed": "🟠 部分完成",
        "failed": "🔴 失败"
    }
    
    overall_status = result.get('overall_status', 'unknown')
    st.markdown(f"**状态**: {status_map.get(overall_status, overall_status)}")
    
    # 症状提取结果（如果是症状提取任务）
    if result.get('task_type') == 'symptom_extraction':
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "症状一致性",
                result.get('is_symptom_consistent', 'N/A')
            )
        
        with col2:
            st.metric(
                "提取症状", 
                result.get('overall_pure_symptoms', 'N/A')
            )
        
        with col3:
            st.metric(
                "处理时间",
                f"{result.get('total_processing_time', 0):.2f}s"
            )
    
    # 详细结果
    st.subheader("详细结果")
    
    # 主知识库结果
    with st.expander("🏥 主知识库结果", expanded=True):
        main_answer = result.get('main_kb_answer', '无结果')
        st.markdown(main_answer)
        
        if result.get('main_kb_has_reference') and result.get('main_kb_use_kb', False):
            st.caption(f"检索到 {result.get('main_kb_final_retrieved_count', 0)} 个相关文档")
        else:
            st.caption("主知识库未处理（知识库未加载或配置）")
    
    # 症状知识库结果
    with st.expander("🩺 症状知识库结果"):
        symptom_answer = result.get('symptom_kb_answer', '无结果')
        st.markdown(symptom_answer)
        
        if result.get('symptom_kb_has_reference') and result.get('symptom_kb_use_kb', False):
            st.caption(f"检索到 {result.get('symptom_kb_final_retrieved_count', 0)} 个相关文档")
        else:
            st.caption("症状知识库未处理")
    
    # 最终整合结果
    with st.expander("📋 整合结果", expanded=True):
        overall_answer = result.get('overall_answer', '无整合结果')
        st.markdown(overall_answer)
    
    # 技术详情
    with st.expander("🔧 技术详情"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**主知识库**")
            st.write(f"- 使用知识库: {result.get('main_kb_use_kb', False)}")
            st.write(f"- 使用重排序: {result.get('main_kb_use_rerank', False)}")
            st.write(f"- 处理时间: {result.get('main_kb_processing_time', 0):.2f}s")
            
        with col2:
            st.write("**症状知识库**") 
            st.write(f"- 使用知识库: {result.get('symptom_kb_use_kb', False)}")
            st.write(f"- 使用重排序: {result.get('symptom_kb_use_rerank', False)}")
            st.write(f"- 处理时间: {result.get('symptom_kb_processing_time', 0):.2f}s")

if __name__ == "__main__":
    main()
