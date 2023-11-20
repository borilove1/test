import torch
import streamlit as st
# from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
# from transformers import pipeline
# from langchain.llms import HuggingFacePipeline
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.llamacpp import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks import StreamlitCallbackHandler
from llama_cpp import llama
#Stream 받아 줄 Hander 만들기
from langchain.callbacks.base import BaseCallbackHandler
from langchain.llms import CTransformers

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text=initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text+=token
        self.container.markdown(self.text)

# @st.cache_resource(show_spinner = False)
# def load_model():
#     model_path = "D:\python\everything_ai\models\koRWKV-1.5B"
#     tokenizer = AutoTokenizer.from_pretrained(
#         pretrained_model_name_or_path = model_path)
    
#     tokenizer.pad_token_id = tokenizer.eos_token_id
#     model = AutoModelForCausalLM.from_pretrained(
#         pretrained_model_name_or_path = model_path,
#         torch_dtype=torch.bfloat16,
#         low_cpu_mem_usage=True)

#     pipe = pipeline(
#         'text-generation',
#         model = model,
#         tokenizer = tokenizer,
#         max_length = 500,
#         temperature = 0.01,
#         do_sample = True,
#         top_p = 0.99,
#         device = 'cuda:0'
#     )
#     llm = HuggingFacePipeline(pipeline=pipe)
#     return llm

# @st.cache_resource(show_spinner = False)
def load_model():
    # MODEL_PATH = "./models/ggml-model-q4_0.gguf"
#     config = {'max_new_tokens': 256, 'repetition_penalty': 1.1, 
#           'temperature': 0.1, 'stream': True}
#     llm = CTransformers(
#         model=MODEL_PATH,
#         model_type="llama",
#         **config
#     )
    MODEL_PATH = "./models/komt-mistral-7b-v1-q4_0.gguf"
    llm = LlamaCpp(        
        model_path = MODEL_PATH,
        temperature = 0,
        max_tokens = 2046,
        n_ctx=2046,
        n_batch=512,
        n_gpu_layers=35,
        top_p = 1,
        # callback_manager = callback_manager,
        verbose = True
    )
    return llm

# @st.cache_resource(show_spinner = False)
def model_memory():
    # Adding history to the model.
    template = """아래 주어진 문맥에 해당하는 내용을 친절하게 답변하시오.
    문맥 : {context}

 
    질문 : {question}
    답변 :"""

    # prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)
    # memory = ConversationBufferMemory(input_key="question", memory_key="history")
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)

    return prompt

# @st.cache_resource(show_spinner = False)
def get_vectorstore():
    model_name = "D://python//FIND_INFO_IN_PDF//models//ko-sbert-nli"
    EMBEDDINGS = HuggingFaceEmbeddings(model_name=model_name,
                                model_kwargs={'device':"cuda"},
                                encode_kwargs = {'normalize_embeddings' : True})
    PERSIST_DIRECTORY = "D:\python\FIND_INFO_IN_PDF\chroma_db"
    DB_Chroma = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=EMBEDDINGS)
    vectorstore = DB_Chroma.as_retriever(search_kwargs = {'k':1})
    return vectorstore

# @st.cache_resource(show_spinner = False)
def qa_retriever():
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

    # Sidebar contents
    st.set_page_config(page_title="AI 챗봇에게 질문을 해보세요 ",
                       page_icon=":books:")
    with st.sidebar:
        st.title("😃한전 사내 업무절차 검색 CHAT AI 시스템 입니다.💬")
        st.markdown(
            """
        사내표준 등 업무와 관련된 문서의 내용을 사전 학습하여 AI가 사용자의 질문에 답변하는 CHAT MODEL 입니다. 

        #### 현재까지 학습된 업무절차서 List
        - 0000
        - 1111
        - 2222
        """
        )
        st.write("Made by 부산울산본부 전자제어부❤️")

    st.title("AI 챗봇에게 질문을 해보세요 💬")

    #####################################################################################################################
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "만나서 반가워요! 업무절차 검색 AI BOT 입니다."}]

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "만나서 반가워요! 업무절차 검색 AI BOT 입니다."}]
    st.sidebar.button("채팅내용 지우기", on_click=clear_chat_history)

    ## 4-2 사용자 입력 받는 부분
    if question := st.chat_input("궁금하신 업무에 대해 질문해주세요!"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                qa = qa_retriever()
                chat_box = st.empty()
                # placeholder = st.empty()
                stream_hander = StreamHandler(chat_box)
                response = qa(question, callbacks=[stream_hander])
                response = qa(question)
                answer, docs = response["result"], response["source_documents"][0]
                docs_metadata = docs.metadata['source']
                # placeholder.markdown(answer)
                # message = {"role": "assistant", "content": f"{answer} \n\n{docs_metadata}"}
                message = {"role": "assistant", "content": answer}
                st.session_state.messages.append(message)

if __name__ == '__main__':
    main()