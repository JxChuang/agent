import streamlit as st
from langchain_community.llms.sparkllm import SparkLLM
import os
import sys
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from dotenv import load_dotenv, find_dotenv

# ✅ 修复BUG1: 修正 sys.path，确保能找到同目录下的 zhipuai_embedding.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from zhipuai_embedding import ZhipuAIEmbeddings

# ✅ 修复BUG4: 使用新版 langchain_chroma 替代已废弃的 langchain_community.vectorstores.Chroma
from langchain_chroma import Chroma


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 页面配置（必须放在最前面）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="智能问答助手",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 自定义 CSS 样式
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
/* 整体背景 */
.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #e8edf5 100%);
}

/* 侧边栏样式 */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
}
[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

/* 主标题区域 */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    text-align: center;
    color: white;
}
.main-header h1 {
    margin: 0;
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.main-header p {
    margin: 0.5rem 0 0;
    font-size: 0.95rem;
    opacity: 0.85;
}

/* 聊天容器 */
[data-testid="stVerticalBlock"] > div:has([data-testid="stChatMessage"]) {
    background: #ffffff;
    border-radius: 16px;
    padding: 1rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
}

/* 用户消息气泡 */
[data-testid="stChatMessage"][data-role="human"] {
    background: linear-gradient(135deg, #667eea15, #764ba215);
    border-left: 3px solid #667eea;
    border-radius: 12px;
    margin: 0.4rem 0;
    padding: 0.5rem;
}

/* AI 消息气泡 */
[data-testid="stChatMessage"][data-role="ai"],
[data-testid="stChatMessage"][data-role="assistant"] {
    background: linear-gradient(135deg, #11998e10, #38ef7d10);
    border-left: 3px solid #11998e;
    border-radius: 12px;
    margin: 0.4rem 0;
    padding: 0.5rem;
}

/* 输入框样式 */
[data-testid="stChatInputContainer"] {
    background: #ffffff;
    border-radius: 16px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    padding: 0.5rem;
    border: 1px solid #e0e0e0;
}

/* 统计卡片 */
.stat-card {
    background: white;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    border: 1px solid #f0f0f0;
    margin-bottom: 0.5rem;
}
.stat-card .stat-number {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat-card .stat-label {
    font-size: 0.8rem;
    color: #888;
    margin-top: 0.2rem;
}

/* 清除按钮 */
.stButton > button {
    width: 100%;
    border-radius: 10px;
    border: none;
    background: linear-gradient(135deg, #ff6b6b, #ee5a24);
    color: white;
    font-weight: 600;
    padding: 0.5rem 1rem;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(238, 90, 36, 0.4);
}

/* 状态徽章 */
.status-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}
.status-online {
    background: #d4edda;
    color: #155724;
}
.status-loading {
    background: #fff3cd;
    color: #856404;
}
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 核心功能函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_retriever():
    """定义get_retriever函数，该函数返回一个检索器"""
    _ = load_dotenv(find_dotenv())
    ZHIPUAI_API_KEY = os.environ["ZHIPUAI_API_KEY"]
    embedding = ZhipuAIEmbeddings(api_key=ZHIPUAI_API_KEY)
    # ✅ 修复BUG3: 使用相对于当前脚本的绝对路径，避免工作目录不同导致路径错误
    base_dir = os.path.dirname(os.path.abspath(__file__))
    persist_directory = os.path.join(base_dir, '..', 'data_base', 'vector_db', 'chroma')
    persist_directory = os.path.normpath(persist_directory)
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    return vectordb.as_retriever()


def combine_docs(docs):
    """处理检索器返回的文本"""
    # Using chr(10) to safely join with newlines avoiding literal newline issues in LLM string generation
    return (chr(10) + chr(10)).join(doc.page_content for doc in docs["context"])


@st.cache_resource(show_spinner=False)
def get_qa_history_chain():
    """定义get_qa_history_chain函数，该函数可以返回一个检索问答链（缓存避免重复初始化）"""
    retriever = get_retriever()
    _ = load_dotenv(find_dotenv())
    IFLYTEK_SPARK_APP_ID = os.environ["IFLYTEK_SPARK_APP_ID"]
    IFLYTEK_SPARK_API_KEY = os.environ["IFLYTEK_SPARK_API_KEY"]
    IFLYTEK_SPARK_API_SECRET = os.environ["IFLYTEK_SPARK_API_SECRET"]

    # ✅ 修复BUG5提示: SparkLLM 在 langchain_community 中已提示废弃，但 spark-ai-python 原生调用正常
    llm = SparkLLM(
        model='Spark4.0 Ultra',
        app_id=IFLYTEK_SPARK_APP_ID,
        api_key=IFLYTEK_SPARK_API_KEY,
        api_secret=IFLYTEK_SPARK_API_SECRET,
        spark_api_url="wss://spark-api.xf-yun.com/v4.0/chat",
        spark_llm_domain="4.0Ultra"
    )
    condense_question_system_template = (
        "请根据聊天记录总结用户最近的问题,"
        "如果没有多余的聊天记录则返回用户的问题。"
    )
    condense_question_prompt = ChatPromptTemplate([
        ("system", condense_question_system_template),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])

    retrieve_docs = RunnableBranch(
        (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever),
        condense_question_prompt | llm | StrOutputParser() | retriever,
    )

    system_prompt = (
        "你是一个专业的知识库问答助手。" + chr(10) +
        "请你使用检索到的上下文片段回答这个问题。" + chr(10) +
        "如果你不知道答案就说不知道，不要编造内容。" + chr(10) +
        "请使用简洁、清晰的中文回答用户。" + chr(10) + chr(10) +
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    qa_chain = (
        RunnablePassthrough().assign(context=combine_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    qa_history_chain = RunnablePassthrough.assign(
        context=retrieve_docs,
    ).assign(answer=qa_chain)
    return qa_history_chain


def gen_response(chain, input, chat_history):
    """接受检索问答链、用户输入及聊天历史，以流式返回该链输出"""
    response = chain.stream({
        "input": input,
        "chat_history": chat_history
    })
    for res in response:
        if "answer" in res.keys():
            yield res["answer"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 侧边栏
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🦜 智能问答助手")
        st.markdown("---")

        # 系统状态
        st.markdown("### 📊 系统状态")
        user_count = sum(1 for m in st.session_state.get("messages", []) if m[0] == "human")
        ai_count = sum(1 for m in st.session_state.get("messages", []) if m[0] == "ai")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{user_count}</div>
                <div class="stat-label">我的提问</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{ai_count}</div>
                <div class="stat-label">AI 回复</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # 使用说明
        st.markdown("### 📖 使用说明")
        st.markdown("""
        1. 💬 在下方输入框输入问题
        2. ⏎ 按回车或点击发送
        3. 🤖 AI 将基于知识库回答
        4. 🔄 支持多轮上下文对话
        """)

        st.markdown("---")

        # 模型信息
        st.markdown("### ⚙️ 模型配置")
        st.markdown("""
        - **LLM**: 讯飞星火 Spark 4.0 Ultra
        - **Embedding**: 智谱 AI embedding-3
        - **向量库**: ChromaDB
        """)

        st.markdown("---")

        # 清除对话按钮
        if st.button("🗑️ 清除对话历史"):
            st.session_state.messages = []
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<small style='color:#aaa;text-align:center;display:block;'>Powered by LangChain · Streamlit</small>",
            unsafe_allow_html=True
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主界面
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    render_sidebar()

    # 主标题
    st.markdown("""
    <div class="main-header">
        <h1>🦜🔗 动手学大模型应用开发</h1>
        <p>基于 RAG 技术的智能知识库问答系统 · 支持多轮对话</p>
    </div>
    """, unsafe_allow_html=True)

    # ✅ Session state 初始化
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "qa_history_chain" not in st.session_state:
        with st.spinner("⏳ 正在初始化模型，请稍候..."):
            try:
                st.session_state.qa_history_chain = get_qa_history_chain()
                st.success("✅ 模型初始化成功！", icon="🎉")
            except Exception as e:
                st.error(f"❌ 模型初始化失败：{e}")
                st.stop()

    # 聊天消息容器
    messages = st.container(height=520)

    # 欢迎消息（无历史时显示）
    if not st.session_state.messages:
        with messages:
            st.markdown("""
            <div style='text-align:center; padding:3rem 1rem; color:#888;'>
                <div style='font-size:3rem;'>🤖</div>
                <div style='font-size:1.1rem; margin-top:0.5rem; font-weight:600;'>你好！我是你的 AI 知识库助手</div>
                <div style='font-size:0.9rem; margin-top:0.5rem;'>请在下方输入框中提问，我会基于知识库为你解答 ✨</div>
            </div>
            """, unsafe_allow_html=True)

    # ✅ 修复BUG2: 遍历对话历史时使用容器变量 messages，而非循环变量 message
    for message in st.session_state.messages:
        with messages.chat_message(message[0]):   # ← 关键修复：messages（容器），而非 message（循环变量）
            st.write(message[1])

    # 输入框
    if prompt := st.chat_input("💬 请输入您的问题..."):
        # 将用户输入添加到对话历史
        st.session_state.messages.append(("human", prompt))

        # 显示用户输入
        with messages.chat_message("human"):
            st.write(prompt)

        # 生成并流式输出回复
        with messages.chat_message("ai"):
            with st.spinner("🤔 正在思考..."):
                try:
                    answer = gen_response(
                        chain=st.session_state.qa_history_chain,
                        input=prompt,
                        chat_history=st.session_state.messages
                    )
                    output = st.write_stream(answer)
                    # 存入历史
                    st.session_state.messages.append(("ai", output))
                except Exception as e:
                    # 发生错误时，移除最后添加的用户消息，避免产生无效历史记录
                    if st.session_state.messages and st.session_state.messages[-1][0] == "human" and st.session_state.messages[-1][1] == prompt:
                        st.session_state.messages.pop()
                    st.error(f"❌ 思考失败：{e}")


# ✅ 修复BUG6: 确保 main() 在脚本入口被调用
if __name__ == "__main__":
    main()
else:
    # Streamlit 直接运行脚本，不走 __main__，所以也需要在模块级调用
    main()
