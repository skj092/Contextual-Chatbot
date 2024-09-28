# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables to prevent Python from writing .pyc files and buffer output
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file to the container
COPY requirement.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirement.txt
RUN pip install --no-cache-dir gunicorn

# Create necessary directories and ensure correct permissions
RUN mkdir -p /app && chown -R $(whoami) /app

# Copy the rest of the application code to the container
COPY . .

# Add debugging information to check directory structure and permissions
RUN echo "Directory structure inside the container:" && ls -l /app

# Expose the port for the app to be accessible
EXPOSE 52207
# Run the application using Gunicorn with Uvicorn worker
CMD ["python", "-m", "gunicorn", "--bind=0.0.0.0:52207", "src.app:app", "-k", "uvicorn.workers.UvicornWorker", "--timeout", "180", "--workers", "1"]

