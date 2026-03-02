from database import get_qdrant_client
from sentence_transformers import SentenceTransformer
from chatbot import RAGChatbot
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda
import os

load_dotenv()

# Load Model
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim

# Load LLM (Groq llama3)
llm = ChatGroq(
    model="llama3-70b-8192",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

#Start DB connection
QDRANT_URL = "http://localhost:6333"

client = get_qdrant_client(QDRANT_URL)

if client is None:
    exit(1)
    
#End DB connection

COLLECTION_NAME = "messages"

#Call Chatbot
def invoke_chatbot(query: str):
    rag_chatbot = RAGChatbot(COLLECTION_NAME, model, llm=llm, client=client)
    result = rag_chatbot.answer(query)
    print("RESULT", result)
    return result

chatbot_chain = RunnableLambda(lambda x: invoke_chatbot(x["question"]))
#End call chatbot
