from database import get_qdrant_client
from documents_service import insert_data
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

text_model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#Start DB connection
QDRANT_URL = "http://localhost:6333"

client = get_qdrant_client(QDRANT_URL)

if client is None:
    exit(1)
    
#End DB connection

# Start Data Store
COLLECTION_NAME = "messages"
IMAGE_COLLECTION_NAME = "figures"

input_type = "TEXT"
file_path = "abc.pdf"

documents = [
    "Hello world",
    "Qdrant is a vector database",
    "Storing embeddings locally",
    "Hello There!",
]

inserted_data_result = insert_data(client, COLLECTION_NAME, IMAGE_COLLECTION_NAME, input_type, documents, file_path, text_model, clip_model, clip_processor)
print(inserted_data_result)
# End data store

#Search data
query_text = "Hello world"
query_vector = text_model.encode([query_text])[0].astype("float32").tolist()

results = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_vector,
    limit=3  # top 3 matches
)

for r in results:
    print(f"Found ID={r.id}, Score={r.score}, Text={r.payload['text']}")

#End search data
