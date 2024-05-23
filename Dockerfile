# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variable to ensure python outputs logs in real time
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /ai-assistant-azure

# Install system-level dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    curl \
    vim \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the specified version
RUN pip install --upgrade pip==24.2

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dev tools 
RUN pip install --no-cache-dir jupyter pytest black flake8

# Copy the entire project directory into the container
COPY . .

# Optional: Expose Jupyter's default port (8888) if you intend to use Jupyter for development
# EXPOSE 8888

# Command to run for development 
CMD ["bash"]
