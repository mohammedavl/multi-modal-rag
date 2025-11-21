import os
import pickle
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class VectorStore:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        print(f"Loading embedding model: {model_name}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        self.persist_dir = "data/vector_store"
        os.makedirs(self.persist_dir, exist_ok=True)

        # ⭐ NEW ChromaDB API (correct for Chroma 0.5+)
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        self.collection = self.client.get_or_create_collection(
            name="rag_index",
            metadata={"hnsw:space": "cosine"}
        )

        self.chunks = []
        print("Successfully loaded\n")

    # -------------------------------------------------------
    def create_embeddings(self, chunks):
        self.chunks = chunks
        ids = []
        docs = []
        metas = []

        for i, chunk in enumerate(chunks):
            ids.append(str(i))
            docs.append(chunk["content"])
            metas.append({
                "page": chunk["page"],
                "type": chunk["type"],
                "source": chunk["source"]
            })

        print("Building Chroma index...")

        self.collection.add(
            ids=ids,
            documents=docs,
            metadatas=metas
        )

        print(f"Chroma index built with {len(ids)} vectors.\n")

    # -------------------------------------------------------
    def search(self, query, k=5):
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        formatted = []
        for rank, (doc, meta) in enumerate(zip(docs, metas)):
            formatted.append({
                "chunk": {
                    "content": doc,
                    "page": meta.get("page"),
                    "type": meta.get("type"),
                    "source": meta.get("source")
                },
                "score": 1.0,
                "rank": rank + 1
            })

        return formatted

    # -------------------------------------------------------
    def save(self):
        # Chroma persists automatically
        with open(f"{self.persist_dir}/chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        print("Chroma vector store saved.\n")

    # -------------------------------------------------------
    def load(self):
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_collection("rag_index")

        with open(f"{self.persist_dir}/chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

        print("Loaded vector store.\n")
