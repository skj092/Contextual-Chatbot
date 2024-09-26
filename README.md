# TODO
- [x] Flow: pdf -> Extract Text -> Chunking -> Embedding -> Save to DB -> query -> Embedding -> Search in DB -> Retrieve Chunks -> Generate Response
- [x] Build a boilerplate application
- [x] Refactor the code to seperate db and model
- [x] add logger and timer
- [x] Setup evaluation pipeline
- [x] Prepare question answer pairs for evaluation
- Setup mlops pipeline, versioning


["['sent']", ['sent']"]

## Using RAGAS ( https://arxiv.org/pdf/2309.15217)
https://docs.ragas.io/en/latest/concepts/metrics/index.html#different-types-of-metrics


## Steps for new pdf
1. Run `python evaluation/systhetic_data_generation.py` to generate the synthetic test question-answer pairs.
2. Run the service and then `python evaluation/eval.py` to evaluate the service accuracy.


Problem Statement - Contextual Chat Bot:

Mandatory Requirements:
Simple Contextual Chat Bot Document Parsing: Read and process a long PDF/Word document.

Contextual Chatbot:
Answer questions using the document, responding with "I don't know the answer" if no relevant answer is found.

Simple Interface: Provide a basic interface for document upload and querying.
Performance Evaluation: Propose a method to evaluate model performance in production.
FastAPI Integration: Develop a REST API with endpoints for document upload and querying using FastAPI.
MLOps Pipeline: Propose an end-to-end MLOps pipeline using draw.io, including model versioning, monitoring, and retraining.

Documentation: Include a README with setup instructions.

Advanced Challenge (Bonus Points):
Open Source LLMs: Use an open-source LLM instead of OpenAI (e.g., GPT-Neo, LLAMA).
Document Chunking: Split the document into chunks and store in a vector database (e.g., Milvus).
Semantic Search: Retrieve top 3 relevant chunks using semantic similarity search.
Performance Evaluation: Implement a method to evaluate chatbot performance (e.g., accuracy). Share the report as well
Pipeline management: Propose a method to create and manage data pipelines assuming you bring above use cases to production and have to inference it by sharing a document

How to submit the code?

Create a new GitHub repository with name : qp-ai-assessment
Once you are ready with the code, you can come back on this URL to submit the GitHub Repo Link.

----------------

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
- `POST /query`: Submit a query and get a response from the chatbot

## Performance Evaluation

The system logs various metrics for performance evaluation:
- Response time
- Query relevance (based on semantic similarity)
- User feedback (to be implemented)

To view the logs and metrics, check the console output of the FastAPI application.

## MLOps Pipeline

The project includes an end-to-end MLOps pipeline for model versioning, monitoring, and retraining. The pipeline consists of the following stages:

1. Data Collection
2. Data Preprocessing
3. Feature Engineering


# References:
- https://milvus.io/docs/quickstart.md
- https://claude.ai/chat/74f3d3be-3b5b-48a6-b62d-03bca7a6a2ff
