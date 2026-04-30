# Use lightweight Python image
FROM python:3.10-slim

# Set work directory inside container
WORKDIR /app

# Install system dependencies for Pillow
RUN apt-get update && apt-get install -y \
    build-essential \
    libjpeg62-turbo-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY ./app ./app
COPY ./model ./model
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
