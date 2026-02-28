import re
from rapidfuzz import process, fuzz

from app.core.config import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NALE, TOP_K
from app.rag.loader import PDFLoader
from app.rag.chunker import LangChainTextChunker
from app.rag.embeddings import EmbeddingModel
from app.rag.vectorstore import VectorStore

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv


class RAGEngine:
    def __init__(self):
        self.vector_store = None
        self.terms = []
        self._initialize()

    def _initialize(self):
        load_dotenv()

        text = PDFLoader(PDF_PATH).load()
        chunks = LangChainTextChunker(CHUNK_SIZE, CHUNK_OVERLAP).chunk(text)

        embeddings = EmbeddingModel(EMBEDDING_MODEL_NALE)

        self.vector_store = VectorStore(embeddings)
        self.vector_store.build(chunks)

        # Build a simple term list from chunks (for typo correction)
        self.terms = self._build_term_list(chunks)

        self.llm = ChatGroq(model="llama-3.3-70b-versatile")


        
    # Build a vocabulary of candidate terms from your PDF chunks.
    # This is used to correct typos (e.g., overfittingg -> overfitting)
    def _build_term_list(self, chunks):
        
        vocab = set()
        for ch in chunks:
            # split on non-letters/digits to get words
            words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", ch)
            for w in words:
                w = w.lower()
                # ignore super short words; keep useful terms
                if len(w) >= 5:
                    vocab.add(w)
        return list(vocab)

    def _normalize(self, q: str) -> str:
        # lower + trim + collapse multiple spaces
        q = " ".join(q.strip().split()).lower()
        return q


    # Fuzzy-correct suspicious long words using the PDF vocabulary.
    # Only replaces when confidence is high, to avoid wrong corrections.
    def _fix_typos(self, q: str) -> str:
        if not self.terms:
            return q

        words = q.split()
        fixed = []

        for w in words:
            if len(w) >= 6:
                match = process.extractOne(w, self.terms, scorer=fuzz.WRatio)
                if match and match[1] >= 90:
                    fixed.append(match[0])
                else:
                    fixed.append(w)
            else:
                fixed.append(w)

        return " ".join(fixed)

    def _prepare_query(self, question: str) -> str:
        q = self._normalize(question)
        q = self._fix_typos(q)
        return q

    def generate_answer(self, question: str):

        query_for_search = self._prepare_query(question)
        contexts = self.vector_store.search(query=query_for_search, k=TOP_K, score_threshold=0.8)
        if not contexts:
            contexts = self.vector_store.search(query=query_for_search, k=TOP_K * 3, score_threshold=1.2)

        combined_text = "\n\n".join(contexts)

        prompt_template = f"""
            You answer questions using ONLY the context below.

            Instructions:
            1) Use only information present in the context.
            2) If the answer is not in the context, respond exactly with: I don't know
            3) Keep the answer concise.

            Context:
            <<<
            {combined_text}
            >>>

            Question: {question}

            Answer:
            """.strip()

        agent = create_agent(
            model=self.llm,
            system_prompt="You are a helpful assistant that follows the user instructions."
        )

        result = agent.invoke({
            "messages": [
                {"role": "user", "content": prompt_template}
            ]
        })

        return result["messages"][-1].content