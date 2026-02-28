from langchain_huggingface import HuggingFaceEmbeddings

# Wrapper around Sententrasformer embeddings
class EmbeddingModel:

    def __init__(self, model_name):
        self.model = HuggingFaceEmbeddings(
            model_name = model_name
        )
        
    def embed_document(self, texts):
        return self.model.embed_documents(texts)

    def embed_query(self, query : str):
        return self.model.embed_query(query)
