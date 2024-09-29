from pymilvus import MilvusClient
from typing import List
from src.models import model, tokenizer, gpt_model, prompt
import torch
from src.utils import log_execution_time, device
import time
import logging
from langchain import hub
from langchain_openai import ChatOpenAI
import json
from langchain_ollama import ChatOllama

with open("config.json", "r") as f:
    config = json.load(f)
chunk_size = config["chunk_size"]
retrival_model = config["retrival_model"]
embedding_dim = config["embedding_dim"]
top_k = config["top_k"]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize Milvus client
client = MilvusClient("data.db")


@log_execution_time
def create_chunks(text: str, chunk_size: int = 1000) -> List[str]:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


@log_execution_time
def store_chunks(chunks: List[str]):
    embeddings = model.encode(chunks)
    entities = [
        {"id": i, "vector": embeddings[i], "text": chunks[i], "subject": "tag"}
        for i in range(len(embeddings))
    ]
    client.create_collection("document_chunks", dimension=embedding_dim)
    client.insert(collection_name="document_chunks", data=entities)


@log_execution_time
def semantic_search(query: str, top_k: int = top_k) -> List[str]:
    tik = time.time()
    query_embedding = model.encode([query])
    tok = time.time()
    logger.info(f"Time taken to encode query: {tok-tik}")
    tik = time.time()
    results = client.search(
        collection_name="document_chunks",
        data=query_embedding,
        limit=top_k,
        output_fields=["text"],
    )
    tok = time.time()
    logger.info(f"Time taken to search: {tok-tik}")
    return [hit.get("entity").get("text") for hit in results[0]]


@log_execution_time
def generate_response_gpt2(input_ids, attention_mask):
    """
    Generates a response using the GPT-2 model.
    """
    tik = time.time()
    output = gpt_model.generate(
        input_ids,
        max_length=1279,
        num_return_sequences=1,
        attention_mask=attention_mask,
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    tok = time.time()
    logger.info(f"Time taken to generate response with GPT-2: {tok - tik}")
    return response


@log_execution_time
def generate_response_llm(prompt: str, model_name: str):
    """
    Generates a response using the specified LLM (Ollama or OpenAI).
    """
    tik = time.time()
    if model_name == "ollama":
        llm = ChatOllama(model="llama3.2", temperature=0)
    elif model_name == "openai":
        llm = ChatOpenAI(model="gpt-4o-mini")
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    response = llm.invoke(prompt).content
    tok = time.time()
    logger.info(f"Time taken to generate response with {model_name}: {tok - tik}")
    return response


@log_execution_time
def format_prompt(context: List[str], query: str, model_name: str):
    """
    Formats the prompt based on the model name.
    """
    if model_name == "gpt-neo":
        return f"Context: {' '.join(context)}\n\nQuestion: {query}\n\nAnswer:"
    elif model_name == "ollama":
        return prompt.format(context=" ".join(context), question=query)
    elif model_name == "openai":
        return prompt.format(context=" ".join(context), question=query)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


@log_execution_time
def generate_response(query: str, context: List[str], retrival_model: str = "openai"):
    tik = time.time()
    prompt = hub.pull("rlm/rag-prompt")
    prompt = format_prompt(context, query, retrival_model)
    if retrival_model == "gpt-neo":
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        attention_mask = torch.ones(input_ids.shape, device=device)
        max_length = input_ids.shape[1] + 10
        output = gpt_model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            attention_mask=attention_mask,
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Response -1 : {response}")

    elif retrival_model in ["ollama", "openai"]:
        response = generate_response_llm(prompt, model_name=retrival_model)
    else:
        raise ValueError(f"Unknown model name: {retrival_model}")

    tok = time.time()
    logger.info(f"Time taken to generate response: {tok-tik}")
    return response.split("Answer:")[-1].strip()
