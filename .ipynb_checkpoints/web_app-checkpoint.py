import streamlit as st
import pandas as pd
import time
from config import Config
from medical_processor import MedicalProcessor

# 页面配置
st.set_page_config(
    page_title="医学症状提取系统",
    page_icon="🏥",
    layout="wide"
)

# 初始化session state
if 'processor' not in st.session_state:
    st.session_state.processor = None
    st.session_state.initialized = False

def initialize_system(config):
    """初始化医学处理系统"""
    if st.session_state.processor is None:
        with st.spinner('正在初始化系统...'):
            processor = MedicalProcessor(config)
            success = processor.initialize()
            if success:
                st.session_state.processor = processor
                st.session_state.initialized = True
                return True
            else:
                st.error("系统初始化失败")
                return False
    return True

def main():
    st.title("🏥 医学症状提取系统")
    st.markdown("---")
    
    # 侧边栏配置
    st.sidebar.header("系统配置")
    
    # API配置
    api_type = st.sidebar.selectbox(
        "选择API类型", 
        ["deepseek", "doubao", "qianwen", "ernie"]
    )
    
    api_key = st.sidebar.text_input(
        "API密钥", 
        type="password",
        help="输入您的API密钥"
    )
    
    # 路径配置
    vector_path = st.sidebar.text_input(
        "主向量库路径",
        value="/path/to/main/vector/db",
        help="主知识库的向量存储路径"
    )
    
    symptom_path = st.sidebar.text_input(
        "症状向量库路径",
        value="/path/to/symptom/vector/db", 
        help="症状知识库的向量存储路径"
    )
    
    # 高级选项
    with st.sidebar.expander("高级选项"):
        use_rerank = st.checkbox("启用重排序", value=True)
        initial_k = st.slider("初始检索数量", 10, 50, 20)
        final_k = st.slider("最终文档数量", 3, 15, 5)
        
        task_type = st.selectbox(
            "任务类型",
            ["symptom_extraction", "normal", "definition", "treatment"],
            help="选择处理任务的类型"
        )
    
    # 系统初始化
    if st.sidebar.button("初始化系统"):
        config = Config(
            api_type=api_type,
            api_key=api_key,
            vector_path=vector_path,
            symptom_path=symptom_path,
            use_rerank=use_rerank,
            initial_k=initial_k,
            final_k=final_k
        )
        
        if initialize_system(config):
            st.success("✅ 系统初始化成功！")
        else:
            st.error("❌ 系统初始化失败")
    
    # 主界面
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
        
        # 历史记录
        if 'history' not in st.session_state:
            st.session_state.history = []
        
        if st.session_state.history:
            st.header("查询历史")
            for i, record in enumerate(reversed(st.session_state.history[-5:])):
                with st.expander(f"查询 {len(st.session_state.history)-i}: {record['query'][:50]}..."):
                    display_result(record['result'])
    else:
        st.info("👈 请先在左侧配置并初始化系统")

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
            
            # 保存到历史记录
            st.session_state.history.append({
                'query': f"{user_input}的{question_type}",
                'result': result,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # 显示结果
            st.success(f"✅ 分析完成！耗时：{processing_time:.2f}秒")
            display_result(result)
            
        except Exception as e:
            st.error(f"❌ 分析失败：{str(e)}")

def display_result(result):
    """显示分析结果"""
    # 总体状态
    status_map = {
        "completed": "🟢 完全成功",
        "completed_with_meta_eval": "🟡 元评估完成", 
        "partially_completed": "🟠 部分完成",
        "failed": "🔴 失败"
    }
    
    status_color = {
        "completed": "success",
        "completed_with_meta_eval": "warning",
        "partially_completed": "info", 
        "failed": "error"
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
        
        if result.get('main_kb_has_reference'):
            st.caption(f"检索到 {result.get('main_kb_final_retrieved_count', 0)} 个相关文档")
    
    # 症状知识库结果
    with st.expander("🩺 症状知识库结果"):
        symptom_answer = result.get('symptom_kb_answer', '无结果')
        st.markdown(symptom_answer)
        
        if result.get('symptom_kb_has_reference'):
            st.caption(f"检索到 {result.get('symptom_kb_final_retrieved_count', 0)} 个相关文档")
    
    # 最终整合结果
    with st.expander("📋 整合结果", expanded=True):
        overall_answer = result.get('overall_answer', '无整合结果')
        st.markdown(overall_answer)
    
    # 技术详情
    with st.expander("🔧 技术详情"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**主知识库**")
            st.write(f"- 使用重排序: {result.get('main_kb_use_rerank', False)}")
            st.write(f"- 处理时间: {result.get('main_kb_processing_time', 0):.2f}s")
            
        with col2:
            st.write("**症状知识库**") 
            st.write(f"- 使用重排序: {result.get('symptom_kb_use_rerank', False)}")
            st.write(f"- 处理时间: {result.get('symptom_kb_processing_time', 0):.2f}s")

if __name__ == "__main__":
    main()
