from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import time

app = FastAPI(
    title="Guarded RAG API",
    description="A hallucination-aware RAG pipeline with verifier agent, built over novel corp.",
    version="1.0.0",
)

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    use_verification: Optional[bool] = True

class ChunkReference(BaseModel):
    chunk_id: str
    book: str
    page_start: int
    page_end: int
    rerank_score: float

class QueryResponse(BaseModel):
    question: str
    answer: str
    final_verdict: str
    retrieved_chunks: list[ChunkReference]
    latency_seconds: float
    trace: list[dict]

class EvalRequest(BaseModel):
    id: str
    category: str
    query: str
    expected_answer_contains: Optional[list[str]] = []
    expected_behavior: Optional[str] = ""

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    chunks_loaded: int


# Endpoints

@app.get("/health", response_model=HealthResponse)
def health():
    try:
        n_chunks = len(retriever.chunks)
        model_ok = model is not None
    except Exception:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return HealthResponse(
        status="ok",
        model_loaded=model_ok,
        chunks_loaded=n_chunks,
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    start = time.time()
    try:
        if req.use_verification:
            result = answer_with_verification(
                req.question, retriever, top_k=req.top_k
            )
        else:
            draft = generate_answer_v2(req.question, retriever, top_k=req.top_k)
            result = {
                "query": req.question,
                "final_answer": draft["answer"],
                "final_verdict": "NO_VERIFICATION",
                "trace": [],
                "retrieved_chunks": draft["retrieved_chunks"],
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency = time.time() - start

    chunks_out = [
        ChunkReference(
            chunk_id=c["chunk_id"],
            book=c["book"],
            page_start=c["page_start"],
            page_end=c["page_end"],
            rerank_score=round(c.get("rerank_score", 0.0), 4),
        )
        for c in result["retrieved_chunks"]
    ]

    return QueryResponse(
        question=req.question,
        answer=result["final_answer"],
        final_verdict=result["final_verdict"],
        retrieved_chunks=chunks_out,
        latency_seconds=round(latency, 3),
        trace=result["trace"],
    )


@app.post("/eval")
def eval_single(req: EvalRequest):
    item = req.dict()
    result = evaluate_single(item, retriever, use_verification=True)
    return result