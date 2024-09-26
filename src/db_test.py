from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient

client = MilvusClient("test.db")

model = SentenceTransformer('all-MiniLM-L6-v2')


docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]

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
query_vector = model.encode(["When was the AI founded?"])
res = client.search(collection_name="test", data=query_vector, limit=2, output_fields=["text"])
print([hit['entity']["text"] for hit in res[0]])
