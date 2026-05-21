import pydantic

class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list[str]
    source_id: str = None

class RAGUpsertResult(pydantic.BaseModel):
    ingested: int

class RAGSearchResult(pydantic.BaseModel):
    contexts: list[str]
    source: list[str]

class RAGQuery(pydantic.BaseModel):
    question: str
    top_k: int = 5

class RAGIngestRequest(pydantic.BaseModel):
    pdf_key: str
    source_id: str = None

class S3UploadRequest(pydantic.BaseModel):
    presigned_url: str
    file_path: str
class S3PresignRequest(pydantic.BaseModel):
    pdf_name: str