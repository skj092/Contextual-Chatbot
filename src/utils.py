import functools
import time
from fastapi import UploadFile, HTTPException
import docx
from io import BytesIO
import torch
import logging
import PyPDF2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


device = "cuda" if torch.cuda.is_available() else "cpu"


def log_execution_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(
            f"Function '{func.__name__}' executed in {execution_time:.4f} seconds"
        )
        return result

    return wrapper


def log_async_execution_time(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(
            f"Function '{func.__name__}' executed in {execution_time:.4f} seconds"
        )
        return result

    return wrapper


@log_execution_time
def parse_document(file: UploadFile) -> str:
    content = file.file.read()
    if file.filename.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(BytesIO(content))
        text = "".join([page.extract_text() for page in pdf_reader.pages])
    elif file.filename.endswith(".docx"):
        doc = docx.Document(BytesIO(content))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")
    return text
