from app.core.config import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NALE, TOP_K
from app.rag.loader import PDFLoader
from app.rag.chunker import LangChainTextChunker
from app.rag.embeddings import EmbeddingModel
from app.rag.vectorstore import VectorStore

from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Sinleton-style RAG Engine
# Initialize once and serves all queries
class RAGEngine:

    def __init__(self):
        self.vector_store = None    
        self._initialize()

    def  _initialize(self):
        load_dotenv()

        text = PDFLoader(PDF_PATH).load()

        chunks = LangChainTextChunker(CHUNK_SIZE, CHUNK_OVERLAP).chunk(text)

        embeddings = EmbeddingModel(EMBEDDING_MODEL_NALE)
        
        self.vector_store = VectorStore(embeddings)
        self.vector_store.build(chunks)

        self.llm = ChatGroq(model="llama-3.3-70b-versatile")

    # Generate an answer using the vector store with a grounded prompt
    # Retrieve top -k chunks and pass  them  to llm with a strict prompt
    def generate_answer(self, question : str):

        contexts = self.vector_store.search(query=question, k=TOP_K)
        combined_text="\n\n".join(contexts)

        prompt_template = f"""
            You answer questions using ONLY the context below.

            Instructions:
            1) Read the context carefully.
            2) Answer the question using only information present in the context.
            3) If the answer is not in the context, respond exactly with: I don't know
            4) Keep the answer concise and to the point.

            Context:
            <<<
            {combined_text}
            >>>

            Question:
            {question}

            Answer:
            """

        agent = create_agent(
            model=self.llm,
            system_prompt="You are a helpful assistant"
        )

        result = agent.invoke({
            "messages":[
                {"role" : "user", "content" : prompt_template}
            ]
        })

        return result['messages'][-1].content