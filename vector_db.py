from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class QdrantVectorDB:
    def __init__(self, url="http://localhost:6333", collection="docs", dim=3072):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=self.collection, 
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )
    
    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        operation_info = self.client.upsert(self.collection, points)
        print(f"Upsert status: {operation_info}")
    
    def search(self, query_vector, top_k=5):
        search_results = self.client.query_points(
            collection_name= self.collection,
            query=query_vector,
            limit=top_k
        ).points
        contexts = []
        sources = set()

        for result in search_results:
            payload = result.payload
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}
