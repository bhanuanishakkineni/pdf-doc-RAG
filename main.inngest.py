import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import os
import uuid
import datetime
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantVectorDB
from custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult, RAGQueryResult

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="qa-rag-project",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="qa-rag-project/ingest_pdf",
    trigger=inngest.TriggerEvent(event="qa-rag-project/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context) -> str:
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)
    
    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vectors = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}: {i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantVectorDB().upsert(ids, vectors, payloads)
        return RAGUpsertResult(ingested=len(chunks))


    chunks_and_src = await ctx.step.run("load_and_chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed_and_upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    # ctx.logger.info(f"Received event: {ctx.event.name} with data: {ctx.event.data}")
    # return "PDF ingested successfully"
    return ingested.model_dump()

@inngest_client.create_function(
    fn_id="qa-rag-project/query_pdf",
    trigger=inngest.TriggerEvent(event="qa-rag-project/query_pdf")
)
async def rag_query_pdf(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vector = embed_texts([question])[0]
        store = QdrantVectorDB()
        search_result = store.search(query_vector, top_k)
        return RAGSearchResult(contexts=search_result["contexts"], source=search_result["sources"])
    
    question = ctx.event.data["question"]
    top_k = ctx.event.data.get("top_k", 5)
    search_result = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    context_block = "\n\n".join(search_result.contexts)
    user_content = (
        "Use the following retrieved contexts to answer the question. \n\n"
        f"Context: \n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer the question based on the retrieved contexts. If you don't know the answer, say you don't know."
    )

    adapter = ai.openai.Adapter(
        auth_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-5.4-mini",
    )
    response = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_completion_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You answer questions only based on the provided contexts."},
                {"role": "user", "content": user_content}
            ]
        }
    )
    answer = response["choices"][0]["message"]["content"].strip()
    return {"answer": answer, "sources": search_result.source, "num_contexts": len(search_result.contexts)}



inngest_app = FastAPI()

inngest.fast_api.serve(inngest_app, inngest_client, [rag_ingest_pdf, rag_query_pdf])

