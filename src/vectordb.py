from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, MilvusClient
from typing import List
from src.models import model, tokenizer, gpt_model
import torch
from src.utils import log_execution_time, device
import time
import logging
from config import embedding_dim, top_k
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize Milvus client
client = MilvusClient("data.db")


@log_execution_time
def create_chunks(text: str, chunk_size: int = 1000) -> List[str]:
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


@log_execution_time
def store_chunks(chunks: List[str]):
    embeddings = model.encode(chunks)
    entities = [{"id": i, "vector": embeddings[i], "text": chunks[i],
                 "subject": "tag"} for i in range(len(embeddings))]
    client.create_collection("document_chunks", dimension=embedding_dim)
    client.insert(collection_name="document_chunks", data=entities)


@log_execution_time
def semantic_search(query: str, top_k: int = top_k) -> List[str]:
    tik = time.time()
    query_embedding = model.encode([query])
    tok = time.time()
    logger.info(f"Time taken to encode query: {tok-tik}")
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    tik = time.time()
    results = client.search(collection_name="document_chunks",
                            data=query_embedding, limit=top_k, output_fields=["text"])
    tok = time.time()
    logger.info(f"Time taken to search: {tok-tik}")
    return [hit.get('entity').get('text') for hit in results[0]]


@log_execution_time
def generate_response(query: str, context: List[str]) -> str:
    prompt = f"Context: {' '.join(context)}\n\nQuestion: {query}\n\nAnswer:"
    tik = time.time()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    tok = time.time()
    logger.info(f"Time taken to encode prompt: {tok-tik}")
    attention_mask = torch.ones(input_ids.shape, device=device)

    tik = time.time()
    output = gpt_model.generate(
        input_ids, max_length=1279, num_return_sequences=1, attention_mask=attention_mask)
    tok = time.time()
    logger.info(f"Time taken to generate response: {tok-tik}")

    tik = time.time()
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    tok = time.time()
    logger.info(f"Time taken to decode response: {tok-tik}")
    return response.split("Answer:")[-1].strip()
