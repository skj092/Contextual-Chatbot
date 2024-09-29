from sentence_transformers import SentenceTransformer
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch
from langchain import hub
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage

prompt = hub.pull("rlm/rag-prompt")


# Initialize SentenceTransformer model for embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Initialize GPT-Neo (retriever)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt_model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")


llm = ChatOllama(model="llama3.2", temperature=0)
