import torch
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.llamacpp import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks import StreamlitCallbackHandler


class StreamHandler(BaseCallbackHandler): ## streamlit data streaming 클래스
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text=initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text+=token
        self.container.markdown(self.text)

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
@st.cache_resource(show_spinner = False)
def load_model(): ## llm 불러오기
    MODEL_PATH = "./models/komt-mistral-7b-v1-q4_0.gguf"
    llm = LlamaCpp(        
        model_path = MODEL_PATH,
        temperature = 0,
        max_tokens = 2046,
        n_ctx=2046,
        n_batch=512,
        n_gpu_layers=35,
        top_p = 1,
        callback_manager = callback_manager,
        verbose = True
    )
    return llm

@st.cache_resource(show_spinner = False)
def model_memory(): ## 프롬프트 템플릿
    template = """아래 주어진 문맥에 해당하는 내용을 친절하게 답변하시오.
    문맥 : {context}
    질문 : {question}
    답변 :"""
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)

    return prompt

@st.cache_resource(show_spinner = False)
def get_vectorstore(): ## 임베딩 데이터 불러오기(크로마db)
    model_name = "./models/ko-sbert-nli"
    EMBEDDINGS = HuggingFaceEmbeddings(model_name=model_name,
                                model_kwargs={'device':"cuda"},
                                encode_kwargs = {'normalize_embeddings' : True})
    PERSIST_DIRECTORY = "D:\python\FIND_INFO_IN_PDF\chroma_db"
    DB_Chroma = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=EMBEDDINGS)
    vectorstore = DB_Chroma.as_retriever(search_kwargs = {'k':1})
    return vectorstore

@st.cache_resource(show_spinner = False)
def qa_retriever(): ## qa retriever
    prompt = model_memory()
    QA = RetrievalQA.from_chain_type(
        llm=load_model(),
        chain_type="stuff",
        retriever=get_vectorstore(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return QA

def main():
    st.set_page_config(page_title="AI 챗봇에게 질문을 해보세요 ", page_icon=":books:") ## 사이드 콘텐츠 부분
    with st.sidebar:
        st.title("😃 사내 업무절차 검색 CHAT AI 시스템 입니다.💬")
        st.markdown("사내표준 등 업무와 관련된 문서의 내용을 사전 학습하여 AI가 사용자의 질문에 답변하는 CHAT MODEL 입니다.")
        st.write("Made by ❤️")
    st.title("AI 챗봇에게 질문을 해보세요 💬")
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "만나서 반가워요! 업무절차 검색 AI BOT 입니다."}]
    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
    def clear_chat_history(): ## 채팅 히스토리 삭제(사이드)
        st.session_state.messages = [{"role": "assistant", "content": "만나서 반가워요! 업무절차 검색 AI BOT 입니다."}]
    st.sidebar.button("채팅내용 지우기", on_click=clear_chat_history)

    if question := st.chat_input("궁금하신 업무에 대해 질문해주세요!"): ## 4-2 사용자 입력 받는 부분
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                qa = qa_retriever()
                chat_box = st.empty()
                stream_hander = StreamHandler(chat_box)
                response = qa(question, callbacks=[stream_hander])
                answer, docs = response["result"], response["source_documents"][0]
                docs_metadata = docs.metadata['source']
                message = {"role": "assistant", "content": answer}
                st.session_state.messages.append(message)

if __name__ == '__main__':
    main()
