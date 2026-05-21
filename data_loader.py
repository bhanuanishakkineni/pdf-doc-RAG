import os
import tempfile
from pathlib import Path
import boto3
from botocore.config import Config
from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072

s3_client = boto3.client(
    "s3",
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    endpoint_url="https://s3.us-west-2.amazonaws.com",
    config=Config(signature_version="s3v4"),
)

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)
def load_and_chunk_pdf(path_key: str):
    bucket = os.getenv("S3_BUCKET_NAME")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        s3_client.download_file(bucket, path_key, tmp_path)
        docs = PDFReader().load_data(file=Path(tmp_path))
    finally:
        os.unlink(tmp_path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        # dimensions=EMBED_DIM
    )
    return [item.embedding for item in response.data]

