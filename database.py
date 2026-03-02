from qdrant_client import QdrantClient

def get_qdrant_client(QDRANT_URL: str) -> QdrantClient:
    try:
        client = QdrantClient(QDRANT_URL)
        client.get_collections()  # simple health check
        print("✅ Connected to Qdrant at", QDRANT_URL)
        return client
    except Exception as e:
        print("❌ Cannot reach Qdrant:", e)
        return None