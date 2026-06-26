FROM python:3.10-slim

# Don't buffer Python output (shows logs immediately)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy our code, models, and the test UI into the container
COPY app.py .
COPY index.html .
COPY movies_list.pkl .
COPY similarity.pkl .

# Tell Docker this container listens on port 8000
EXPOSE 8000

# Start the server when the container runs
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]