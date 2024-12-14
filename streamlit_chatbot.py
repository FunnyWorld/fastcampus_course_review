import os
import streamlit as st
from langchain_openai import (ChatOpenAI, OpenAIEmbeddings)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from operator import itemgetter


from typing import Final

k_user: Final[str] = 'user'
k_ai : Final[str] = 'assistant'
k_persist_directory: Final[str] = './chroma_db'

store = {}


@st.cache_resource
def load_and_split_document(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

@st.cache_resource
def create_vector_store(_original_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(_original_docs)
    
    vector_store = Chroma.from_documents(persist_directory=k_persist_directory, documents=docs, embedding=OpenAIEmbeddings(model='text-embedding-3-small'))
    return vector_store
    
@st.cache_resource
def get_vector_store(_docs):
    if os.path.exists(k_persist_directory):
        vector_store = Chroma(persist_directory=k_persist_directory, embedding_function=OpenAIEmbeddings(model='text-embedding-3-small'))
        return vector_store

    return create_vector_store(_docs)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


@st.cache_resource
def chaining():
    file_path = r"../../data/대한민국헌법(헌법)(제00010호)(19880225).pdf"
    pages = load_and_split_document(file_path)
    vector_store = get_vector_store(pages)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

#     prompt = PromptTemplate.from_template("""You are an helpful AI assistant for question-answering tasks. 
# Use the following pieces of retrieved context to answer the question. 
# If you don't know the answer, just say that you don't know.

# ## Context:
# {context}

# ## Chat History:
# {chat_history}

# ## Question:
# {question}

# ## Answer:
# """)
    
    system_template = """You are an helpful AI assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know.

## Context:
{context}
"""
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', system_template), 
            MessagesPlaceholder(variable_name='chat_history'), 
            ('human', '{question}')
        ]
    )
    
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.1)
    output_parser = StrOutputParser()

    chain = (
        {"context": itemgetter('question') | retriever | format_docs, "question": RunnablePassthrough(), "chat_history": itemgetter("chat_history")} 
        | prompt 
        | llm 
        | output_parser
    )

    chain_with_message_history = RunnableWithMessageHistory(
        chain, 
        get_session_history, 
        input_messages_key="question", 
        history_messages_key="chat_history"
    )
    
    return chain_with_message_history


os.environ["OPENAI_API_KEY"] = "..."

st.header("헌법 QA 챗봇 😜")


if 'messages' not in st.session_state:
    st.session_state['messages'] = [{'role': k_ai, 'content': "대한민국 헌법에 대해서 무엇이든 물어 보세요!"}]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])


rag_chain = chaining()


if prompt := st.chat_input("질문을 입력해 주세요!"):
    st.session_state.messages.append({'role': k_user, 'content': prompt})
    st.chat_message(k_user).write(prompt)
    
    with st.chat_message(k_ai):
        with st.spinner("답변 준비 중..."):
            response = rag_chain.invoke(
                input={"question": prompt},
                config={"configurable": {'session_id': 'flyh21c@gmail.com'}}
            )

            st.session_state.messages.append({'role': k_ai, 'content': response})
            st.write(response)



