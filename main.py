import os
import streamlit as st
import pickle
import time
from langchain_community.llms import Ollama
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


st.title("News Research Tool")

st.sidebar.title("News Articles URLS")


urls =[]
for i in range(3):
    url =st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked =st.sidebar.button("Process URLs")

file_path ="faiss_store.pkl"

main_placefolder =st.empty()

llm =Ollama(model = 'llama3')

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placefolder.text("Data Loading Started")
    data =loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size =1000,chunk_overlap =200)
    main_placefolder.text("Text Splitting Started")

    docs =text_splitter.split_documents(data)

    embeddings =OllamaEmbeddings(model="llama3")
    vectorstore =FAISS.from_documents(docs, embeddings)
    main_placefolder.text("Embedding Vector Started")
    time.sleep(2)


    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

query = main_placefolder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever =vectorstore.as_retriever())
            result =chain({"question": query},return_only_outputs=True)
            st.header("Answer")
            st.subheader(result['answer'])
