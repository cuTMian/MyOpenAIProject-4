import os

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

def qa_agent(msg, memory, uploaded_file, openai_api_key, openai_base_url):
    if openai_base_url:
        model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key, openai_api_base=openai_base_url)
    else:
        model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)

    file_content = uploaded_file.read()
    with open("./file.pdf", "wb") as f:
        f.write(file_content)
    loader = PyPDFLoader("./file.pdf")
    docs = loader.load()

    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", ""]
    )
    texts = text_spliter.split_documents(docs)

    embeddings_model = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings_model)
    retriever = db.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory
    )

    response = qa.invoke({"chat_history": memory, "question": msg})
    return response

if __name__ == '__main__':
    with open('./temp.pdf', 'rb') as f:
        memory = ConversationBufferMemory(
            return_messages=True,
            memory_key='chat_history',
            output_key='answer'
        )

        response = qa_agent("如果我想学习Transformer框架，我应该看哪篇论文呢", memory, f, os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_BASE_URL"))
        print(response)
