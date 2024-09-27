import json

# Global variables
chunk_size = 1000
model_name = "all-MiniLM-L6-v2"
embedding_dim = 384
top_k = 3
retrival_model = "gpt2"
retrival_model = "ollama"
retrival_model = "openai"

# Create a list of variable names to include
var_names = [name for name in globals() if not name.startswith('__')
             and name != 'json']

# Create a dictionary from selected global variables
config = {name: globals()[name] for name in var_names}

# Write the configuration to a json file

with open('config.json', 'w') as file:
    json.dump(config, file, indent=4)


print("config file updated")
