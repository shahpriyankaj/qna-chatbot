from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from rag_pipeline import rag
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    global rag_obj
    rag_obj = rag.RAG()
    print("LLM Model & FAISS vector store loaded successfully!")
    yield

app = FastAPI(lifespan=lifespan)
class Question(BaseModel):
    question: str

@app.post("/chat")
async def chat(request: Question):
    """Streaming response from LLM"""
    return StreamingResponse(rag_obj.generate_response(request.question), media_type="application/json")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    