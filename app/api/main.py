from fastapi import FastAPI, Query
from app.rag.engine import RAGEngine

app = FastAPI()

rag_engine = RAGEngine()

# uvicorn app.api.main:app
# Return an LLM generated answer, grounded using the PDF content
@app.get("/query")
def query(question : str = Query(..., description="User question!")):

    try:
        answer = rag_engine.generate_answer(question)
        return{
            "question" : question,
            "answer" : answer
        }
    except Exception as e:
        return {
            "question" : question,
            "answer" : None,
            "error" : str(e)
        }