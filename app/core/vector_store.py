import os
import uuid
from typing import List, Dict, Any
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

class QdrantDB:
    def __init__(self):
        # 1. Configuration via Environment Variables
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        print(f"Connecting to Qdrant at: {self.qdrant_url}")
        
        # Initialize Client
        self.client = QdrantClient(url=self.qdrant_url)
        
        # --- COLLECTIONS ---
        self.JOBS_COLLECTION = "jobs_collection"
        self.CANDIDATES_COLLECTION = "candidates_collection"
        self.PATIENTS_COLLECTION = "patient_profiles"  # <--- NEW

        print("Loading AI Models... (This happens only once)")
        self.dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5",cache_dir="./fast_embedding_cache")
        self.sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1",cache_dir="./fast_embedding_cache")
        
        self._init_collection(self.JOBS_COLLECTION)
        self._init_collection(self.CANDIDATES_COLLECTION)
        self._init_collection(self.PATIENTS_COLLECTION) # <--- NEW

    def _init_collection(self, collection_name: str):
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=384,
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=True)
                    )
                }
            )
            print(f"Created collection: {collection_name}")

    def _get_dense_embedding(self, text: str) -> List[float]:
        # fastembed returns a generator, we take the first item
        embedding = list(self.dense_model.embed([text]))[0]
        return embedding.tolist()

    def _get_sparse_embedding(self, text: str) -> models.SparseVector:
        sparse_vec = list(self.sparse_model.embed([text]))[0]
        return models.SparseVector(
            indices=sparse_vec.indices.tolist(),
            values=sparse_vec.values.tolist()
        )

    # --- GENERIC UPSERT/SEARCH HELPERS ---
    def _upsert(self, collection_name: str, point_id: str, text: str, payload: Dict[str, Any]):
        dense_vec = self._get_dense_embedding(text)
        sparse_vec = self._get_sparse_embedding(text)
        
        self.client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=point_id, 
                    vector={"dense": dense_vec, "sparse": sparse_vec}, 
                    payload=payload
                )
            ]
        )

    def _search(self, collection_name: str, query_text: str, limit: int) -> List[Dict[str, Any]]:
        dense_query = self._get_dense_embedding(query_text)
        sparse_query = self._get_sparse_embedding(query_text)
        
        results = self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(query=sparse_query, using="sparse", limit=limit * 2),
                models.Prefetch(query=dense_query, using="dense", limit=limit * 2),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit
        )

        matches: List[Dict[str, Any]] = []
        for hit in results.points:
            payload = hit.payload if hit.payload is not None else {}
            match_data = payload.copy()
            match_data["match_score"] = hit.score
            match_data["point_id"] = hit.id # Return ID so we can filter self out
            matches.append(match_data)
        return matches
    
    def _delete(self, collection_name: str, point_id: str):
        self.client.delete(collection_name=collection_name, points_selector=[point_id])


    # --- PATIENT METHODS (New) ---
    def upsert_patient(self, user_id: uuid.UUID, text_content: str, payload: Dict[str, Any]):
        """
        Stores a patient's psychological profile.
        text_content: Combined string of "Background + Chat Summary + Diagnosis"
        """
        # Ensure ID is string for Qdrant
        self._upsert(self.PATIENTS_COLLECTION, str(user_id), text_content, payload)

    def search_patients(self, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Finds patients with similar symptoms or backgrounds.
        """
        return self._search(self.PATIENTS_COLLECTION, query_text, limit)


# --- WRAPPERS (To be imported in API) ---
vector_db = QdrantDB()

def upsert_patient_embedding(user_id: uuid.UUID, text_content: str, payload: Dict[str, Any]):
    vector_db.upsert_patient(user_id, text_content, payload)

def search_similar_patients(query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
    return vector_db.search_patients(query_text, limit)