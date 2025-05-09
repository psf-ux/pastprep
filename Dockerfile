# Use an official Python runtime as a base image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

ARG GEMINI_API
ARG MONGO_URI

ENV GEMINI_API=$GEMINI_API
ENV MONGO_URI=$MONGO_URI

# Copy the current directory contents into the container at /app
COPY . /app

COPY requirements.txt .

# Install any necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Flask will run on
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=main.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run the Flask app
CMD ["flask", "run"]
