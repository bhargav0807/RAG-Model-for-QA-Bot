# Use the official Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app code into the working directory
COPY . .

# Expose the port where the app will run
EXPOSE 8501

# Set environment variables to ensure Streamlit runs properly
ENV PINECONE_API_KEY=<your-pinecone-api-key>
ENV OPENAI_API_KEY=<your-openai-api-key>

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]
