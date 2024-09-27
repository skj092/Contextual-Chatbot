from sentence_transformers import SentenceTransformer
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch



# Initialize SentenceTransformer model for embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')


# Initialize GPT-Neo (retriever)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt_model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

