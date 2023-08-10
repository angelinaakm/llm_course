import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
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
    return vector_store

def ask_and_get_answer(vector_store, q, k=3): #higher k is more expensive
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.2)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.run(q)
    return answer

def return_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens/1000 * 0.0004


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    #st.image('img.png')
    st.subheader('LLM Question-Answering Application')
    with st.sidebar:
        api_key=st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        uploaded_file = st.file_uploader('Upload a file ', type=['pdf', 'docs', 'txt'])
        chunk_size = st.number_input('Chunk size: ', min_value=100, max_value=2048, value=512)
        k = st.number_input('k', min_value=1, max_value=20, value=3)
        add_data = st.button ('Add Data')
    