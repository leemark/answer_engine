import os
# Import the dotenv module to load environment variables from a .env file
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# https://python.langchain.com/docs/integrations/text_embedding/openai
# import OpenAIEmbeddings to embed text
from langchain_openai import OpenAIEmbeddings
# create an instance of OpenAIEmbeddings  
embed_model = OpenAIEmbeddings(model="text-embedding-3-small")

# https://python.langchain.com/docs/integrations/chat/groq
# import ChatGroq and ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# create an instance of ChatGroq
chat = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
# create an instance of ChatPromptTemplate
system = "You are a helpful search query builder assistant and always respond ONLY with a reworded version of the user input that should be given to a search engine API. Always be succint and use the same words as the input.  ONLY RETURN THE REPHRASED VERSION OF THE USER INPUT WITH NO OTHER TEXT OR COMMENTARY"
human = "INPUT TO REPHRASE:{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | chat

rephrased_query = chain.invoke({"text": "What is the latest news in cancer treatments?"})
rephrased_query = rephrased_query.content

# remove any leading or trailing quotes from the rephrased query
rephrased_query = rephrased_query.strip('"')
# remove any parenthetical remarks from the rephrased query
rephrased_query = rephrased_query.split("(")[0]

# https://python.langchain.com/docs/integrations/tools/brave_search
# import BraveSearch and query the brave search api
from langchain_community.tools import BraveSearch
tool = BraveSearch.from_api_key(api_key= os.environ["BRAVE_API_KEY"])
docs = tool.run(rephrased_query)

# convert docs to json
import json
docs_list = json.loads(docs)

#for each doc in docs_list print the link
for doc in docs_list:
    url = doc["link"]
    print(url)
