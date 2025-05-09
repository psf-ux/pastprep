from flask import Flask, request, jsonify
import google.generativeai as genai
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import requests
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Access the variables
GEMINI_API= "AIzaSyByMXNJGawuAJadbGoLAG2XY_9vBfN9iNE"
MONGO_URI = "mongodb+srv://k214889:t6u.CLkfQhGhFPJ@cluster0.5rnig.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Flask app setup
app = Flask(__name__)

# MongoDB setup
uri = MONGO_URI 
client = MongoClient(uri, server_api=ServerApi('1'))
collection = client["rag_db"]["pastprepvectors"]
embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

# Define a function to generate embeddings
def get_embedding(data):
    """Generates vector embeddings for the given data."""
    embedding = embedding_model.encode(data)
    return embedding.tolist()

# Define a function to perform the vector query
def get_query_results(query, course, category):
    """Gets results from a vector search query."""
    query_embedding = get_embedding(query)
    # print("Embeddings: ",query_embedding)
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedding",
                "exact": True,
                "limit": 10
            }
        },
        {
            "$project": {
                "_id": 1,
                "text": 1,
                "year": 1,
                "Course": 1,
                "Paper": 1,
                "Category": 1
            }
        }
    ]

    results = collection.aggregate(pipeline)

    array_of_results = []
    for doc in results:
        if "Course" in doc and "Category" in doc:
            if doc["Course"] == course and doc["Category"] == category:
                array_of_results.append(doc)

    return array_of_results

# Gemini API Configuration
genai.configure(api_key=GEMINI_API)

def get_gemini_response(question, context):
    """Sends a request to the Gemini API to get the answer."""
    prompt = f"Respond only in plaintext\n Answer the following question based on the given context:\n\nQuestion: {question}\nContext: {context}"
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

# Flask endpoint
@app.route('/query', methods=['POST'])
def query_endpoint():
    """Handles the query and returns results."""
    data = request.json
    question = data.get("question")
    course = data.get("course")
    category = data.get("category")

    print(course, category, question)

    # Validate inputs
    if not question or not course or not category:
        return jsonify({"error": "Invalid input. 'question', 'course', and 'category' are required."}), 400

    # Get vector search results
    results = get_query_results(question, course, category)
    # print("Results: ",results)
    context = "\n".join([result["text"] for result in results])

    # Get answer from Gemini API
    answer = get_gemini_response(question, context)
    print("\n\n\nAnswer: \n\n",answer)
    return jsonify({"question": question, "answer": answer, "context": context})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)