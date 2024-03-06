import streamlit as st
import os
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts.prompts import SimpleInputPrompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
from huggingface_hub import login
os.environ['HF_TOKEN']=os.getenv('HF_TOKEN')
login(token=os.getenv('HF_TOKEN'))

documents = SimpleDirectoryReader('data').load_data()
# Set up LLama2
system_prompt = """
You are a Q&A assistant. Your goal is to answer questions as
accurately as possible based on the question and context provided.
"""
query_wrapper_prompt = SimpleInputPrompt("{query_str}")

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 6.0, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",
)

embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embed_model
)

index = VectorStoreIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine()


# Streamlit app
st.title("HR Chatbot")


user_input = st.text_input("Ask a question:")
if st.button("Ask"):
    response = query_engine.query(user_input)
    st.write(response)
