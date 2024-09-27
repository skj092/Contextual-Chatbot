from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import uvicorn
from fastapi.responses import FileResponse
import time
from src.vectordb import create_chunks, store_chunks, semantic_search, generate_response
import logging
from src.utils import (log_async_execution_time, parse_document)
from config import chunk_size, retrival_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI()


class Query(BaseModel):
    text: str


@app.get("/")
@log_async_execution_time
async def read_index():
    return FileResponse("index.html")


@app.post("/upload")
@log_async_execution_time
async def upload_document(file: UploadFile = File(...)):
    text = parse_document(file)
    chunks = create_chunks(text, chunk_size)
    store_chunks(chunks)
    return {"message": "Document uploaded and processed successfully"}


@app.post("/query")
@log_async_execution_time
async def query_document(query: Query):
    tik = time.time()
    print(f"question from user: {query}")
    relevant_chunks = semantic_search(query.text)
    if not relevant_chunks:
        return {"response": "I don't know the answer to that question."}
    response = generate_response(query.text, relevant_chunks, retrival_model=retrival_model)
    tok = time.time()
    print(f"Total tokens {len(response.split())}, Time taken: {tok-tik}")
    print(f"Token per second: {len(response.split())/(tok-tik)}")
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
