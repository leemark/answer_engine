import os
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.tools import BraveSearch
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
import chromadb
from langchain_openai import OpenAIEmbeddings
import uuid

# Load environment variables
load_dotenv()

# Constants
MODEL_NAME = "mixtral-8x7b-32768"
USER_PROMPT = "What is the latest news in open source LLMs"
SYSTEM_REPHRASE_PROMPT = "You are a helpful search query builder assistant and always respond ONLY with a reworded version of the user input that should be given to a search engine API. Always be succint and use the same words as the input. ONLY RETURN THE REPHRASED VERSION OF THE USER INPUT WITH NO OTHER TEXT OR COMMENTARY"
HUMAN_REPHRASE_PROMPT = "INPUT TO REPHRASE:{text}"

def create_chat_pipeline():
    chat = ChatGroq(temperature=0, model_name=MODEL_NAME)
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_REPHRASE_PROMPT), ("human", HUMAN_REPHRASE_PROMPT)])
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
    for url in urls:
        loader = WebBaseLoader(url)
        document_list = loader.load() 
        for document in document_list:
            soup = BeautifulSoup(document.page_content, 'html.parser')
            text_contents = soup.get_text()
            contents.append(text_contents)
    return contents

# Function to embed contents into a Chroma vector database
def embed_into_chroma_db(contents, collection_name):
    # Create the OpenAI Embedding function
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Initialize Chroma's persistent client and create/get a collection
    persistent_client = chromadb.PersistentClient()
    collection = persistent_client.get_or_create_collection(collection_name)
    
    # Iterate over the contents and add them to the database
    for content in contents:
        # Generate a unique ID for each content
        doc_id = str(uuid.uuid4())
        # Add the document to the Chroma collection
        collection.add(ids=[doc_id], documents=[content])

    print(f"Successfully embedded {len(contents)} documents into the Chroma DB in collection '{collection_name}'")

def query_chroma_db(query, collection_name, top_k=5):
    # Create the OpenAI Embedding function
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Initialize Chroma's persistent client and get the collection
    persistent_client = chromadb.PersistentClient()
    collection = persistent_client.get_collection(collection_name)

    # Query the collection and retrieve the top-k most similar documents
    results = collection.query(query_texts=[query], n_results=top_k)

    # Extract the relevant document contents from the results
    relevant_docs = [doc for doc, score in zip(results["documents"][0], results["distances"][0])]

    return relevant_docs

def generate_llm_response(relevant_docs, query):
    # Concatenate the relevant documents into a single string
    context = "\n".join(relevant_docs)
    # Create the LLM
    llm = ChatGroq(temperature=0.5, model_name=MODEL_NAME)
    # Generate the response using the LLM
    response = llm.invoke(f"Context: {context}\nQuery: {query}")
    return response

def main():
    chat_pipeline = create_chat_pipeline()
    input_text = USER_PROMPT
    rephrased_query = rephrase_query(chat_pipeline, input_text)
    
    # Search URLs
    urls = search_urls(os.environ["BRAVE_API_KEY"], rephrased_query)
    
    # Scrape and parse contents
    page_contents = scrape_and_parse(urls)
    
    # Load contents into vector DB using chroma
    embed_into_chroma_db(page_contents, collection_name="chroma_collection")

    # Get relevant results from vector db
    relevant_docs = query_chroma_db(rephrased_query, collection_name="chroma_collection")

    # Pass relevant documents to an LLM and generate response
    llm_response = generate_llm_response(relevant_docs, rephrased_query)

    print("LLM Response:")
    print(llm_response)

if __name__ == '__main__':
    main()
