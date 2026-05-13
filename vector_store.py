import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import hashlib

class VectorStore:
    def __init__(self, persist_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_path)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.collection = self.client.get_or_create_collection("documents")
    
    def add_chunks(self, chunks: List[Dict]) -> None:
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode(texts).tolist()
        
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=[{"page": c["page"], "source": c["source"]} 
                      for c in chunks],
            ids=[
                f"{hashlib.md5(c['source'].encode()).hexdigest()}_{i}" 
                for i, c in enumerate(chunks)
            ]
        )
    
    def search(self, query: str, n_results: int = 3) -> List[Dict]:
        query_embedding = self.embedder.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        return [
            {
                "text": results["documents"][0][i],
                "page": results["metadatas"][0][i]["page"],
                "source": results["metadatas"][0][i]["source"]
            }
            for i in range(len(results["documents"][0]))
        ]