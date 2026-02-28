from langchain_community.vectorstores import FAISS

# FAISS-base vector store for document retreival
class VectorStore:

    def __init__(self, embeddings):
        self.embeddings=embeddings
        self.store=None

    # Build FAISS inde chunks
    def build(self, texts):
        self.store = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings.model
        )

    # Retrive tok-k relvant chunks
    def search(self, query : str, k : int = 3):
        if self.store is None:
            raise ValueError("Vector Store not initialized")
        
        docs = self.store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
        