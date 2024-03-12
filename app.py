import os
import json
import uuid
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.tools import BraveSearch
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
import requests
import chromadb

# Load environment variables
load_dotenv()

# Constants
MODEL_NAME = "mixtral-8x7b-32768"
USER_PROMPT = "What are the latest advancements in electric cars?"
SYSTEM_REPHRASE_PROMPT = "You are a helpful search query builder assistant and always respond ONLY with a reworded version of the user input that should be given to a search engine API. Always be succint and use the same words as the input. ONLY RETURN THE REPHRASED VERSION OF THE USER INPUT WITH NO OTHER TEXT OR COMMENTARY"
HUMAN_REPHRASE_PROMPT = "INPUT TO REPHRASE:{text}"

def create_chat_pipeline():
    chat = ChatGroq(temperature=0, model_name=MODEL_NAME)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_REPHRASE_PROMPT),
        ("human", HUMAN_REPHRASE_PROMPT)
    ])
    return prompt | chat

def rephrase_query(chat_pipeline, text):
    rephrased_query = chat_pipeline.invoke({"text": text}).content
    rephrased_query = rephrased_query.strip('"').split("(")[0]
    return rephrased_query

def search_urls(api_key, rephrased_query):
    tool = BraveSearch.from_api_key(api_key)
    docs = tool.run(rephrased_query)
    docs_list = json.loads(docs)
    urls = [doc["link"] for doc in docs_list]
    return urls

def scrape_and_parse(urls):
    contents = []
    sources = []
    
    # Custom headers to mimic a browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36"
    }
    
    for url in urls:
        try:
            loader = WebBaseLoader(url, requests_kwargs={"headers": headers})
            document_list = loader.load()
            for document in document_list:
                soup = BeautifulSoup(document.page_content, 'html.parser')
                text_contents = soup.get_text()
                contents.append(text_contents)
                sources.append(url)
        except requests.exceptions.RequestException as e:
            print(f"Error occurred while scraping {url}: {str(e)}")
            continue
    
    return contents, sources

def embed_into_chroma_db(contents, sources, collection_name):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    persistent_client = chromadb.PersistentClient()
    collection = persistent_client.get_or_create_collection(collection_name)

    for content, source in zip(contents, sources):
        doc_id = str(uuid.uuid4())
        collection.add(ids=[doc_id], documents=[content], metadatas=[{"source": source}])

    print(f"Successfully embedded {len(contents)} documents into the Chroma DB in collection '{collection_name}'")

def query_chroma_db(query, collection_name, top_k=5):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    persistent_client = chromadb.PersistentClient()
    collection = persistent_client.get_collection(collection_name)

    results = collection.query(query_texts=[query], n_results=top_k, include=["documents", "metadatas"])
    relevant_docs = results["documents"][0]
    relevant_sources = [metadata["source"] for metadata in results["metadatas"][0]]

    return relevant_docs, relevant_sources

def generate_llm_response(relevant_docs, relevant_sources, query):
    context = "\n".join([f"Source: {source}\nContent: {doc}" for doc, source in zip(relevant_docs, relevant_sources)])
    llm = ChatGroq(temperature=0.5, model_name=MODEL_NAME)
    response = llm.invoke(f"Context:\n{context}\n\nQuery: {query}")
    return response

def main():
    chat_pipeline = create_chat_pipeline()
    rephrased_query = rephrase_query(chat_pipeline, USER_PROMPT)
    print("Rephrased Query:")
    print(rephrased_query)

    urls = search_urls(os.environ["BRAVE_API_KEY"], rephrased_query)
    page_contents, sources = scrape_and_parse(urls)

    embed_into_chroma_db(page_contents, sources, collection_name="chroma_collection")
    relevant_docs, relevant_sources = query_chroma_db(rephrased_query, collection_name="chroma_collection")

    llm_response = generate_llm_response(relevant_docs, relevant_sources, rephrased_query)

    print("LLM Response:")
    print(llm_response)
    print("Relevant Sources:")
    print(relevant_sources)
if __name__ == '__main__':
    main()