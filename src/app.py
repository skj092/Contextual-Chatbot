from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import PyPDF2
import docx
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from io import BytesIO
from fastapi.responses import FileResponse
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, MilvusClient

app = FastAPI()

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Milvus
#client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
client = MilvusClient("data.db")

# Initialize GPT-Neo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt_model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

class Query(BaseModel):
    text: str

def parse_document(file: UploadFile) -> str:
    content = file.file.read()
    if file.filename.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(BytesIO(content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif file.filename.endswith('.docx'):
        doc = docx.Document(BytesIO(content))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")
    return text

def create_chunks(text: str, chunk_size: int = 1000) -> List[str]:
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def store_chunks(chunks: List[str]):
    print(f"Storing {len(chunks)} chunks in Milvus")
    embeddings = model.encode(chunks)
    print("Embeddings generated successfully")
    print(f"Embedding shape: {embeddings.shape}")
    entities = [{"id": i, "vector" : embeddings[i], "text": chunks[i], "subject": "tag"} for i in range(len(embeddings))]
    client.create_collection("document_chunks", dimension=384)
    client.insert(collection_name="document_chunks", data=entities)
    print("Chunks stored successfully")

def semantic_search(query: str, top_k: int = 3) -> List[str]:
#    collection = Collection("document_chunks")
#    collection.load()

    query_embedding = model.encode([query])[0]
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    #results = collection.search([query_embedding], "embedding", search_params, limit=top_k)
    results = client.search(collection_name="document_chunks", data=[query_embedding], top_k=top_k)
    print(f"Search results: {results}")

    return [hit.get('entity').get('text') for hit in results[0]]

def generate_response(query: str, context: List[str]) -> str:
    prompt = f"Context: {' '.join(context)}\n\nQuestion: {query}\n\nAnswer:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones(input_ids.shape, device=device)

    output = gpt_model.generate(input_ids, max_length=100, num_return_sequences=1, attention_mask=attention_mask)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response.split("Answer:")[-1].strip()

@app.get("/")
async def read_index():
    return FileResponse("index.html")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    text = parse_document(file)
    chunks = create_chunks(text)
    store_chunks(chunks)
    return {"message": "Document uploaded and processed successfully"}

@app.post("/query")
async def query_document(query: Query):
    relevant_chunks = semantic_search(query.text)
    relevant_chunks = [chunk for chunk in relevant_chunks if chunk]
    print(f"Relevant chunks: {relevant_chunks}")
    if not relevant_chunks:
        return {"response": "I don't know the answer to that question."}

    response = generate_response(query.text, relevant_chunks)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
