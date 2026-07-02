import streamlit as st
from langchain_community.llms.sparkllm import SparkLLM
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
import sys

sys.path.append("../streamlitapp")
from zhipuai_embedding import ZhipuAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import Chroma


# 定义get_retriever函数，该函数返回一个检索器
def get_retriever():
    # 定义Embedding
    embedding = ZhipuAIEmbeddings()
    # 向量数据库持久化路径
    persist_directory = '../data_base/vector_db/chroma'
    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    return vectordb.as_retriever()


# 定义combine_docs函数，该函数处理检索器返回的文本
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs["context"])


# 定义get_qa_history_chain函数，该函数可以返回一个检索问答链
def get_qa_history_chain():
    retriever = get_retriever()
    _ = load_dotenv(find_dotenv())
    IFLYTEK_SPARK_APP_ID = os.environ["IFLYTEK_SPARK_APP_ID"]
    IFLYTEK_SPARK_API_KEY = os.environ["IFLYTEK_SPARK_API_KEY"]
    IFLYTEK_SPARK_API_SECRET = os.environ["IFLYTEK_SPARK_API_SECRET"]
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
        "你是一个回答任务的助手。"
        "请你使用检索到的上下文片段回答这个问题。"
        "如果你不知道答案就说不知道。"
        "请使用简洁的话语回答用户。"
        "\n\n"
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


# 定义 gen_respondse函数，它接受检索问答链、用户输入及聊天历史，并以流式返回该链输出
def gen_response(chain, input, chat_history):
    response = chain.stream({
        "input": input,
        "chat_history": chat_history
    })
    for res in response:
        if "answer" in res.keys():
            yield res["answer"]


# 定义main函数，该函数制定显示效果与逻辑
def main():
    st.markdown("### 🦜🔗 动手学大模型应用开发")
    # st.session_state可以存储用户应用交互期间的状态与数据
    # 存储对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 存储问答检索链
    if "qa_history_chain" not in st.session_state:
        st.session_state.qa_history_chain = get_qa_history_chain()

    # 简历容器高度为500px
    messages = st.container(height=500)

    # 显示整个对话历史
    for message in st.session_state.messages:  # 遍历对话历史
        with message.chat_message(message[0]):  # messages指在容器下显示，chat_message显示用户及ai头像
            st.write(message[1])  # 打印内容
    if prompt := st.chat_input("Say somthing"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append(("human", prompt))
        # 显示当前用户输入
        with messages.chat_message("human"):
            st.write(prompt)
        # 生成回复
        answer = gen_response(
            chain=st.session_state.qa_history_chain,
            input=prompt,
            chat_history=st.session_state.messages
        )
        # 流式输出
        with messages.chat_message("ai"):
            output = st.write_stream(answer)

        # 将输出存入st.session_state.messages
        st.session_state.messages.append(("ai", output))


