import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os
from typing import List

# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file) -> List[str]:
    import os
    name, extension = os.path.splitext(file)
    try:
        if extension == '.pdf':
            from langchain.document_loaders import PyPDFLoader
            print(f'Loading {file}')
            loader = PyPDFLoader(file)
        elif extension == '.docx':
            from langchain.document_loaders import Docx2txtLoader
            print(f'Loading {file}')
            loader = Docx2txtLoader(file)
        elif extension == '.txt':
            from langchain.document_loaders import TextLoader
            loader = TextLoader(file)
        else:
           # print('Document format is not supported!')
           # return None
            raise Exception ('Document format is not supported!')
    except Exception as e:
        print('Caught a document error: ' + repr(e)) 
    data = loader.load()
    return data



def chunk_data(data, chunk_size=256, chunk_overlap=20):
    '''Break down large pieces of data into smaller segments
    Helps optimizerelevance of the content from a vector database
    '''
    from langchain.text_splitter import RecursiveCharacterTextSplitter  
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap) 
    chunks = text_splitter.split_documents(data) 
    return chunks


def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)