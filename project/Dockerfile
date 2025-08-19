# Use a lightweight official Python image as a base, with CUDA support
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the trainer directory and setup.py file into the container
COPY trainer /app/trainer
COPY setup.py /app/setup.py

# Install the dependencies from setup.py
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir .

# Set the entry point for the container, which runs your training script
# The -m flag tells Python to find the module and run it
ENTRYPOINT ["python3", "-m", "trainer.U-Net"]