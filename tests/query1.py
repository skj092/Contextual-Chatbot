import requests
import time

url = "http://127.0.0.1:8000/query"

question = "who is sonu?"
# Your query text
query = {
    "text": question
}

tik = time.time()
response = requests.post(url, json=query)
tok = time.time()

print(f"total time taken {tok-tok}")
print(response.json())

