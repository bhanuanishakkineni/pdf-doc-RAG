import logging
from fastapi import FastAPI, Request, HTTPException
import boto3
from dotenv import load_dotenv
import os
import uuid
import datetime
from data_loader import load_and_chunk_pdf, embed_texts, client
from vector_db import QdrantVectorDB
from custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult, RAGQueryResult, RAGQuery

load_dotenv()

app = FastAPI()

def _load(pdf_path: str, source_id: str) -> RAGChunkAndSrc:
    # pdf_path = ctx.event.data["pdf_path"]
    # source_id = ctx.event.data.get("source_id", pdf_path)
    try:
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)
    except Exception as e:
        logging.error(f"Error loading and chunking PDF: {e}")
        raise Exception(f"Failed to load and chunk PDF: {str(e)}")
    
def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
    chunks = chunks_and_src.chunks
    source_id = chunks_and_src.source_id
    try:
        vectors = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}: {i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        # Qdrant client upsert is synchronous in this module
        QdrantVectorDB().upsert(ids, vectors, payloads)
        return RAGUpsertResult(ingested=len(chunks))
    except Exception as e:
        logging.error(f"Error upserting chunks: {e}")
        raise Exception(f"Failed to upsert chunks: {str(e)}")

def _search(question: str, top_k: int = 5) -> RAGSearchResult:
    try:
        query_vector = embed_texts([question])[0]
        store = QdrantVectorDB()
        # vector_db.search is synchronous; call directly
        search_result = store.search(query_vector, top_k)
        return RAGSearchResult(contexts=search_result["contexts"], source=search_result["sources"])
    except Exception as e:
        logging.error(f"Error searching chunks: {e}")
        raise Exception(f"Failed to search chunks: {str(e)}")


@app.post("/ingest")
async def ingest_pdf(request: Request):
    try:
        data = await request.json()
        pdf_path = data["pdf_path"]
        source_id = data.get("source_id", pdf_path)
        chunks_and_src = await _load(pdf_path, source_id)
        ingested = await _upsert(chunks_and_src)
        return {"ingested": ingested.ingested}
    except Exception as e:
        logging.error(f"Ingest endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_pdf(query: RAGQuery):
    try:
        search_result = _search(query.question, query.top_k)
        context_block = "\n\n".join(search_result.contexts)
        user_content = (
            "Use the following retrieved contexts to answer the question. \n\n"
            f"Context: \n{context_block}\n\n"
            f"Question: {query.question}\n"
            "Answer the question based on the retrieved contexts. If you don't know the answer, say you don't know."
        )
        try:
            model_response = client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[
                        {"role": "system", "content": "You answer questions only based on the provided contexts."},
                        {"role": "user", "content": user_content}
                    ],
                max_completion_tokens=1024,
                temperature=0.2
            )
        except Exception as e:
            logging.error(f"LLM request failed: {e}")
            raise HTTPException(status_code=502, detail=f"LLM error: {str(e)}")

        # Safely extract model output
        try:
            answer = model_response.choices[0].message.content
        except Exception:
            logging.error("Unexpected model response shape")
            raise HTTPException(status_code=500, detail="Unexpected model response shape")

        return {"answer": answer}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Query endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))