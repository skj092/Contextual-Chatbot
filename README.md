# setup with docker
```bash
docker-compose up
```

# Setup without docker

1. Create a virtual environment

```bash
python -m venv .venv
```

2. activate the virtual environment

```bash
source .venv/bin/activate
```

3. Install the dependencies

```bash
pip install -r requirement.txt
```

4. Run the application

```bash
sh startup.sh
```

# Generate the test data

1. keep all the pdf in the /data/pdf folder
2. python evaluation/systhetic_data_generation.py
3. python evaluation/eval.py

python evaluation/synthetic_data_generate.py --pdf_directory="data/pdfs" --num_questions=12


```
{
    "chunk_size": 1000,
    "model_name": "all-MiniLM-L6-v2",
    "embedding_dim": 384,
    "top_k": 3,
    "retrival_model": "openai"
}
```
currently 3 retrival_models are available - 1. "gpt-neo", "ollama" and "openai"
To Use ollama need to setup ollama locally
To Use openai need to setup api key in the environment


# TODO
- [x] Flow: pdf -> Extract Text -> Chunking -> Embedding -> Save to DB -> query -> Embedding -> Search in DB -> Retrieve Chunks -> Generate Response
- [x] Build a boilerplate application
- [x] Refactor the code to seperate db and model
- [x] add logger and timer
- [x] Setup evaluation pipeline
- [x] Prepare question answer pairs for evaluation
- [ ] Setup mlops pipeline, versioning:
    - [x] Things to track (Store in DB)
        - [x] Chunk Size
        - [x] Embedding Model name
        - [x] Embedding dimension
    - [ ] Things to track (Retrieve):
        - [x] LLM model name
        - [ ] Latency
        - [ ] Accuracy
        - [ ] Cost
- [ ] Streaming response setup

------------
1. To Use Openai model set the openai key as an environment variable
2. To use ollama locally, install and run the ollama locally

-----------------------
- Unit Testing Response
- gpt x - accuracy not good, respons time ~ 30s, token/s=24.27
- ollama(llama 2.1) - accuracy good, response time - 30s, token/s = 1.9
- gpt4o-mini- accuracy good, response time 4.6s, token/s = 10.6


## Steps for new pdf
1. Run `python evaluation/systhetic_data_generation.py` to generate the synthetic test question-answer pairs.
2. Run the service and then `python evaluation/eval.py` to evaluate the service accuracy.


# Contextual Chatbot

This project implements a contextual chatbot that can answer questions based on uploaded documents. It uses FastAPI for the backend, Milvus for vector storage, and GPT-Neo for text generation.


## Usage

1. Upload a document:
   - Click on the "Choose File" button in the "Upload Document" section.
   - Select a PDF or DOCX file from your computer.
   - Click the "Upload" button to process and store the document.

2. Ask questions:
   - Enter your question in the text box in the "Ask a Question" section.
   - Click the "Submit" button to get a response from the chatbot.

3. View chat history:
   - The chat history will be displayed in the section below the query form.

## API Endpoints

- `POST /upload`: Upload and process a document (PDF or DOCX)

# ML Ops Cycle

New pdf -> Generate QA pairs -> Evaluate -> Update the config -> Evaluate
1. Generate QA pairs (evaluation/systhetic_data_generation.py)
2. Evaluate (evaluation/eval.py)
3. Check the score (score.csv) and update the config (config.py)
4. Repeat 2 and 3

# Files to Track using dvc
1. testset.csv -> evaluation/systhetic_data_generation.py -> Generate the question answer pairs (GT)
2. question_answer.csv -> Intermediary file
3. score.csv -> evaluation/eval.py -> predict and answer and evaluate the model by comparing with GT.
4. config.json -> config.py -> Store the configuration


# References:
- Vector DB: https://milvus.io/docs/quickstart.md
- RAGAS : https://arxiv.org/pdf/2309.15217)


