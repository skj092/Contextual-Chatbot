from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
import PyPDF2
import docx
from io import BytesIO
from fastapi import HTTPException
from typing import List

client = MilvusClient("test.db")

model = SentenceTransformer('all-MiniLM-L6-v2')



def parse_document(file_path: str) -> str:
    # Determine the file format based on the file extension
    if file_path.endswith('.pdf'):
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(BytesIO(f.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
    elif file_path.endswith('.docx'):
        with open(file_path, 'rb') as f:
            doc = docx.Document(BytesIO(f.read()))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    else:
        raise Exception("Unsupported file format")

    #print(f"Document content: {text}")
    return text

def create_chunks(text: str, chunk_size: int = 1000) -> List[str]:
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


pdf = "/home/sonujha/rnd/qp-ai-assessment/static/sonu.pdf"
text = parse_document(pdf)
print(f"length of text: {len(text)}")
docs = create_chunks(text)
print(f"length of docs: {len(docs)}")
print(f"length of first doc: {len(docs[0])}")

vectors = model.encode(docs)
print("dimension:", len(vectors[0]))
print("shape of vectors:", vectors.shape)


data = [{"id": i, "vector": vectors[i], "text": docs[i], "subject": "tag"}
        for i in range(len(vectors))]

print("data has", len(data), "endities, each with fields:", data[0].keys())
print("vector dim", len(data[0]["vector"]))


# insert data
if client.has_collection("test"):
    client.drop_collection("test")
collection = client.create_collection("test", dimension=384)
res = client.insert(collection_name="test", data=data)
print(res)


# search
query_vector = model.encode(["whas is the education background of sonu"])
print(query_vector.shape)
res = client.search(collection_name="test", data=query_vector,
                    limit=2, output_fields=["text"])
print([hit['entity']["text"] for hit in res[0]])
