import logging
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import boto3
from botocore.config import Config
from dotenv import load_dotenv
import os
import uuid
import datetime
import requests
from data_loader import load_and_chunk_pdf, embed_texts, s3_client, client
from vector_db import QdrantVectorDB
from custom_types import RAGChunkAndSrc, RAGIngestRequest, RAGUpsertResult, RAGSearchResult, RAGQuery, S3PresignRequest, S3UploadRequest

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["localhost:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# s3_client = boto3.client("s3", region_name=os.getenv("AWS_REGION"), aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"), aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"), endpoint_url="https://s3.us-west-2.amazonaws.com", config=Config(signature_version="s3v4"))

# print("s3_client configured with bucket:", os.getenv("S3_BUCKET_NAME"))
# print("S3 client region:", s3_client.meta.region_name)

def _load(pdf_key: str, source_id: str) -> RAGChunkAndSrc:
    # pdf_key = ctx.event.data["pdf_key"]
    # source_id = ctx.event.data.get("source_id", pdf_key)
    try:
        chunks = load_and_chunk_pdf(pdf_key)
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

@app.post("/docs/presign", status_code=status.HTTP_200_OK)
async def presign_docs(S3PresignRequest: S3PresignRequest):
    key = f"pdfs/{uuid.uuid4()}.pdf"
    presigned_url = s3_client.generate_presigned_url("put_object", Params={"Bucket": os.getenv("S3_BUCKET_NAME"), "Key": key, "ContentType": "application/pdf"}, ExpiresIn=3600)
    return {"presigned_url": presigned_url, "key": key}

# Only for testing with local files - in production, the frontend should upload directly to S3 using the presigned URL
@app.post("/docs/upload", status_code=status.HTTP_200_OK)
async def upload_docs(s3Upload: S3UploadRequest):
    with open(s3Upload.file_path, "rb") as f:
        response = requests.put(s3Upload.presigned_url, headers={"Content-Type": "application/pdf"}, data=f)
        if response.status_code != 200:
            logging.error(f"S3 upload failed: {response.text}")
            raise HTTPException(status_code=500, detail="Failed to upload file to S3")
    return {"message": "Docs uploaded successfully"}


@app.post("/ingest")
async def ingest_pdf(ingest_request: RAGIngestRequest):
    try:
        pdf_key = ingest_request.pdf_key
        source_id = ingest_request.source_id or pdf_key
        chunks_and_src = _load(pdf_key, source_id)
        ingested = _upsert(chunks_and_src)
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

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "ok"}