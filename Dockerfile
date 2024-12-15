FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Set the working directory
WORKDIR /app

# Copy project files
COPY . .

# Install project dependencies
RUN pip install --no-cache-dir .

# Default command (can be overridden)
CMD ["python", "--version"]
CMD ["python", "train.py"]
