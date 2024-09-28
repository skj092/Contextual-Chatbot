python -m gunicorn --bind=0.0.0.0:52207 src.app:app -k uvicorn.workers.UvicornWorker --timeout 180 --workers 1
