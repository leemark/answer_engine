
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
system = "You are a helpful search query builder assistant and always respond ONLY with a reworded version of the user input that should be given to a search engine API. Always be succint and use the same words as the input.  ONLY RETURN THE REPHRASED VERSION OF THE USER INPUT WITH NO OTHER TEXT."
human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | chat
response = chain.invoke({"text": "Write a sonnet about the lord of the rings."})

print(response)