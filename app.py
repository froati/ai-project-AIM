import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import os
from rag_engine import InvestmentRAGEngine

# --- Page Config ---
st.set_page_config(
    page_title="삼성전자 스마트 투자 보조 시스템",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize RAG Engine ---
@st.cache_resource
def load_rag_engine():
    pdf_files = [
        "[삼성전자]분기보고서(2026.05.15).pdf",
        "[삼성전자]사업보고서(2026.03.10).pdf"
    ]
    # Ensure paths are correct relative to where streamlit is run
    base_path = os.path.dirname(__file__)
    full_paths = [os.path.join(base_path, os.path.basename(f)) for f in pdf_files]
    return InvestmentRAGEngine(full_paths)

try:
    rag_engine = load_rag_engine()
    if rag_engine and rag_engine.vector_db:
        st.sidebar.success(f"✅ 문서 학습 완료 ({len(rag_engine.pdf_paths)}개 파일)")
        for p in rag_engine.pdf_paths:
            st.sidebar.caption(f"📄 {os.path.basename(p)}")
    else:
        st.sidebar.warning("⚠️ 학습된 문서가 없습니다. PDF 파일을 확인해주세요.")
except Exception as e:
    st.error(f"RAG 엔진 로드 실패: {e}")
    rag_engine = None

# --- Custom CSS (Notion Style & Layout) ---
st.markdown("""
<style>
    .main {
        background-color: #ffffff;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    .notion-text {
        font-family: 'Inter', sans-serif;
        line-height: 1.6;
        color: #37352f;
    }
    .source-accordion {
        background-color: #f7f6f3;
        border-radius: 5px;
        padding: 10px;
    }
    .badge-safe { background-color: #e2fceb; color: #216e39; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .badge-warning { background-color: #fff9db; color: #856404; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .badge-danger { background-color: #ffe3e3; color: #cf222e; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- Mock Data & Helpers ---
def get_financial_metrics():
    # 실제 시스템에서는 Vector DB나 API에서 가져올 데이터
    return {
        "PER": {"value": 15.2, "status": "Safe", "range": (0, 30)},
        "PBR": {"value": 1.4, "status": "Safe", "range": (0, 3)},
        "ROE": {"value": 12.5, "status": "Normal", "range": (0, 20)},
        "Debt_Ratio": {"value": 35.0, "status": "Safe", "range": (0, 100)}
    }

def create_gauge_chart(value, title, min_val, max_val):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title, 'font': {'size': 18}},
        gauge = {
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [min_val, (max_val-min_val)*0.4], 'color': "#e2fceb"},
                {'range': [(max_val-min_val)*0.4, (max_val-min_val)*0.7], 'color': "#fff9db"},
                {'range': [(max_val-min_val)*0.7, max_val], 'color': "#ffe3e3"}
            ],
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# --- Sidebar: Dashboard ---
with st.sidebar:
    st.header("📊 실시간 대시보드")
    metrics = get_financial_metrics()
    
    st.subheader("재무 건전성")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**PER**")
        st.markdown('<span class="badge-safe">안전</span>', unsafe_allow_html=True)
        st.plotly_chart(create_gauge_chart(metrics["PER"]["value"], "", 0, 30), width="stretch")
    with col2:
        st.markdown("**PBR**")
        st.markdown('<span class="badge-safe">안전</span>', unsafe_allow_html=True)
        st.plotly_chart(create_gauge_chart(metrics["PBR"]["value"], "", 0, 3), width="stretch")
        
    st.subheader("시장 심리 (Sentiment)")
    sentiment_data = pd.DataFrame({
        "Sentiment": ["긍정", "중립", "부정"],
        "Ratio": [65, 20, 15]
    })
    fig_donut = px.pie(sentiment_data, values='Ratio', names='Sentiment', hole=.4,
                 color_discrete_sequence=['#2ecc71', '#95a5a6', '#e74c3c'])
    fig_donut.update_layout(showlegend=False, height=250, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_donut, width="stretch")
    
    st.subheader("시장 동향 점수")
    trend_data = pd.DataFrame({
        "Date": pd.date_range(start="2026-05-01", periods=10),
        "Score": [70, 72, 68, 75, 80, 78, 82, 85, 83, 88]
    })
    fig_line = px.line(trend_data, x="Date", y="Score")
    fig_line.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_line, width="stretch")

# --- Main Area: AI Assistant ---
st.title("🤖 삼성전자 AI 투자 비서")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(f'<div class="notion-text">{message["content"]}</div>', unsafe_allow_html=True)
        if "sources" in message:
            with st.expander("📚 출처 확인하기 (Source Accordion)"):
                for source in message["sources"]:
                    st.markdown(f"- [{source['title']}]({source['link']})")

# User Input
if prompt := st.chat_input("삼성전자의 최근 배당 정책에 대해 알려줘"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Self-RAG Process Visualization
        status_text = st.status("🔍 정보를 분석 중입니다...", expanded=True)
        time.sleep(0.5)
        status_text.write("1. 관련 문서 검색 중 (Knowledge Base: 삼성전자 보고서)")
        
        if rag_engine:
            response_text, sources = rag_engine.process_query(prompt)
            status_text.write("2. 답변 생성 및 자가 검증 중 (Gemini Self-Reflection)")
            time.sleep(0.5)
            status_text.update(label="✅ 분석 완료!", state="complete", expanded=False)
            
            # Display response
            message_placeholder.markdown(f'<div class="notion-text">{response_text}</div>', unsafe_allow_html=True)
            
            # Display sources
            if sources:
                with st.expander("📚 출처 확인하기 (Source Accordion)"):
                    for src in sources:
                        filename = os.path.basename(src['title'])
                        st.markdown(f"- **{filename}** (p.{src['page']})")
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_text,
                "sources": [{"title": os.path.basename(s['title']), "link": "#"} for s in sources]
            })
        else:
            status_text.update(label="❌ RAG 엔진 미로드", state="error", expanded=True)
            st.error("RAG 엔진이 설정되지 않았습니다. API 키와 파일 경로를 확인해주세요.")
