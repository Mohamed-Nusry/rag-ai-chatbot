from qdrant_client.http import models
from unstructured.partition.pdf import partition_pdf
import uuid
#from PIL import Image
import torch

def insert_data(client, COLLECTION_NAME: str, IMAGE_COLLECTION_NAME: str, input_type: str, documents, file_path, text_model, clip_model, clip_processor):
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        existing_names = [c.name for c in collections]

        if COLLECTION_NAME in existing_names:
            print(f"⚠️ Collection '{COLLECTION_NAME}' already exists")
        else:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={"size": 384, "distance": "Cosine"},
            )
            print(f"✅ Collection '{COLLECTION_NAME}' created successfully")

        #Create images collection
        if IMAGE_COLLECTION_NAME not in existing_names:
            client.recreate_collection(
                collection_name=IMAGE_COLLECTION_NAME,
                vectors_config={"size": 512, "distance": "Cosine"},
            )
            print(f"✅ Collection '{IMAGE_COLLECTION_NAME}' created successfully")

        if input_type == "TEXT":
            embeddings = text_model.encode(documents).astype("float32")
            print("Embedding shape:", embeddings.shape)

            points = [
                models.PointStruct(
                    id=i,  # unique ID
                    vector=embeddings[i].tolist(),
                    payload={"text": documents[i]},  # optional metadata
                )
                for i in range(len(documents))
            ]

            client.upsert(collection_name=COLLECTION_NAME, points=points)

            return f"✅ Inserted {len(points)} embeddings into collection '{COLLECTION_NAME}'"
        else:
            file_store(client, COLLECTION_NAME, IMAGE_COLLECTION_NAME, file_path, text_model, clip_model, clip_processor)
    except Exception as e:
        return e
    
def file_store(client, COLLECTION_NAME: str, IMAGE_COLLECTION_NAME: str, file_path, text_model, clip_model, clip_processor):

    FILE_PATH = file_path

    doc_id = str(uuid.uuid4())
    elements = partition_pdf(FILE_PATH, strategy="hi_res")

    points_text, points_figures = [], []
    last_chunk_id = None

    for i, e in enumerate(elements):
        if not e.text or not e.text.strip():
            continue

        element_id = f"chunk_{i}"

        if e.category in ["NarrativeText", "Title", "Table"]:
            embedding = text_model.encode(e.text).tolist()

            points_text.append(
                models.PointStruct(
                    id=element_id,
                    vector=embedding,
                    payload={
                        "doc_id": doc_id,
                        "chunk_id": element_id,
                        "type": e.category,
                        "text": e.text,
                        "source": FILE_PATH,
                    }
                )
            )
            last_chunk_id = element_id

        elif e.category == "Figure":
            related = []
            if last_chunk_id:
                related.append(last_chunk_id)

            # Look ahead for next text/table
            for j in range(i+1, len(elements)):
                if elements[j].category in ["NarrativeText", "Table"]:
                    related.append(f"chunk_{j}")
                    break

            # NOTE: if you have extracted image files for figures, load them here
            # (unstructured may give you a figure caption, but not always the actual image)
            # Example:
            # image_path = f"page_{e.metadata.page_number}_fig_{i}.png"
            # image = Image.open(image_path)
            # using PIL Image

            # For demo, we'll just embed the figure's caption using CLIP text encoder
            inputs = clip_processor(text=e.text, return_tensors="pt", padding=True)
            with torch.no_grad():
                clip_emb = clip_model.get_text_features(**inputs).squeeze().tolist()

            points_figures.append(
                models.PointStruct(
                    id=element_id,
                    vector=clip_emb,
                    payload={
                        "doc_id": doc_id,
                        "chunk_id": element_id,
                        "type": "Figure",
                        "caption": e.text,
                        "related_chunks": related,
                        "source": FILE_PATH,
                    }
                )
            )

        if points_text:
            client.upsert(collection_name=COLLECTION_NAME, points=points_text)

        if points_figures:
            client.upsert(collection_name=IMAGE_COLLECTION_NAME, points=points_figures)

        print(f"✅ Stored {len(points_text)} text chunks and {len(points_figures)} figures from {FILE_PATH}")
