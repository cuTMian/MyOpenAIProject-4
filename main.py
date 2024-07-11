import streamlit as st

from langchain.memory import ConversationBufferMemory

from utils import qa_agent

st.title("PDF问询器")
with st.sidebar:
    openai_api_key = st.text_input('请输入您的OpenAI API密钥', type='password')
    st.markdown("[获取OpenAI API密钥](https://platform.openai.com/api-keys)")
    openai_base_url = st.text_input('如果需要，您可以输入您的api base url')

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
            memory_key='chat_history',
            output_key='answer'
    )

uploaded_file = st.file_uploader("上传你的pdf", type="pdf")
msg = st.text_input("您的问题")
start = st.button("点击开始回答", disabled=not uploaded_file)

if uploaded_file and start:
    warning_msg = ""
    if not openai_api_key:
        warning_msg = "请输入您的OpenAI API密钥！"

    if warning_msg:
        st.info(warning_msg)
        st.stop()

    with st.spinner(("AI思考中，请稍等...")):
        response = qa_agent(msg, st.session_state["memory"], uploaded_file, openai_api_key, openai_base_url)
    st.write("### 答案")
    st.write(response["answer"])
    st.session_state["chat_history"] = response["chat_history"]

if "chat_history" in st.session_state:
    with st.expander("历史消息"):
        i = 0
        while i < len(st.session_state["chat_history"]):
            st.write(st.session_state["chat_history"][i].content)
            st.write(st.session_state["chat_history"][i+1].content)
            i += 2
            if i <= len(st.session_state["chat_history"]) - 2:
                st.divider()

