from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, MilvusClient
from typing import List
from src.models import model, tokenizer, gpt_model
import torch
from src.utils import log_execution_time, device



# Initialize Milvus client
client = MilvusClient("data.db")


@log_execution_time
def create_chunks(text: str, chunk_size: int = 1000) -> List[str]:
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


@log_execution_time
def store_chunks(chunks: List[str]):
    embeddings = model.encode(chunks)
    entities = [{"id": i, "vector": embeddings[i], "text": chunks[i], "subject": "tag"} for i in range(len(embeddings))]
    client.create_collection("document_chunks", dimension=384)
    client.insert(collection_name="document_chunks", data=entities)

@log_execution_time
def semantic_search(query: str, top_k: int = 3) -> List[str]:
    query_embedding = model.encode([query])
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = client.search(collection_name="document_chunks", data=query_embedding, limit=top_k, output_fields=["text"])
    return [hit.get('entity').get('text') for hit in results[0]]

@log_execution_time
def generate_response(query: str, context: List[str]) -> str:
    prompt = f"Context: {' '.join(context)}\n\nQuestion: {query}\n\nAnswer:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones(input_ids.shape, device=device)
    output = gpt_model.generate(input_ids, max_length=1279, num_return_sequences=1, attention_mask=attention_mask)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.split("Answer:")[-1].strip()

