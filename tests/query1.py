import requests
import time

url = "http://0.0.0.0:52207/query"

question = "who is sonu?"
# Your query text
query = {
    "text": question
}

tik = time.time()
response = requests.post(url, json=query)
tok = time.time()

print(f"total time taken {tok-tik}")
print(response.json())

