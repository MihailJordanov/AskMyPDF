from langchain_community.vectorstores import FAISS

class VectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.store = None

    def build(self, texts):
        self.store = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings.model
        )

    # Returns: list[str] of page contents (filtered by threshold if provided).
    # Note: for FAISS, score meaning depends on distance metric. Lower is often better for L2 distance.
    # We'll treat threshold as 'max distance' if provided.
    def search(self, query: str, k: int = 3, score_threshold: float | None = None):

        if self.store is None:
            raise ValueError("Vector Store not initialized")

        results = self.store.similarity_search_with_score(query, k=k)
        # results: List[Tuple[Document, score]]

        if score_threshold is not None:
            results = [(doc, score) for (doc, score) in results if score <= score_threshold]

        for doc, score in results:
            print("SCORE:", score, "TEXT:", doc.page_content[:80])

        return [doc.page_content for (doc, score) in results]