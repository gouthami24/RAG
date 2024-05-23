import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain

# Set your OpenAI API key here
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

st.title('RAG Search')

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature = 0.5)
response = llm.invoke("What is LLM")
#print(response)
st.write("Before Search using RAG, RESPONSE : ", response.content)

def get_docs():
    loader = WebBaseLoader('https://www.techtarget.com/whatis/feature/Foundation-models-explained-Everything-you-need-to-know')
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )

    splitDocs = text_splitter.split_documents(docs)

    return splitDocs

def create_vector_store(docs):
    embedding = OpenAIEmbeddings(api_key=openai_api_key)
    vectorStore = FAISS.from_documents(docs, embedding=embedding)
    return vectorStore

def create_chain(vectorStore):
    model = ChatOpenAI(api_key=openai_api_key,
        temperature=0.4,
        model='gpt-3.5-turbo-1106'
    )

    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question.
    Context: {context}
    Question: {input}
    """)

    #print(prompt)

    # chain = prompt | model
    # We are creating the chain to add documents
    document_chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    # Retrieving the top 1 relevant document from the vector store , We can change k to 2 and get top 2 and so on
    retriever = vectorStore.as_retriever(search_kwargs={"k": 1})

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain

docs = get_docs()
vectorStore = create_vector_store(docs)
chain = create_chain(vectorStore)

response = chain.invoke({
    "input": "What is LLM?",
})

st.write("After RAG Search, RESPONSE : ", response['answer'])
#print(response)
