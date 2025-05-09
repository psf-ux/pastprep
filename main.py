from flask import Flask, request, jsonify
import google.generativeai as genai
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
import numpy as np

import requests
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Access the variables
GEMINI_API = os.environ.get("GEMINI_API")
MONGO_URI = os.environ.get("MONGO_URI")

# Flask app setup
app = Flask(__name__)

# MongoDB setup
uri = MONGO_URI
client = MongoClient(uri, server_api=ServerApi("1"))
collection = client["rag_db"]["pastprepvectors"]
notes_collection = client["rag_db"]["notes_vectors"]
embedding_model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1", trust_remote_code=True
)
# Create a second embedding model for notes_vectors similarity calculation
notes_embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


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
                "limit": 10,
            }
        },
        {
            "$project": {
                "_id": 1,
                "text": 1,
                "year": 1,
                "Course": 1,
                "Paper": 1,
                "Category": 1,
            }
        },
    ]

    results = collection.aggregate(pipeline)

    array_of_results = []
    for doc in results:
        if "Course" in doc and "Category" in doc:
            if doc["Course"] == course and doc["Category"] == category:
                array_of_results.append(doc)

    return array_of_results


def retrieve_similar_docs(subject, question, top_k=3):
    subject_docs = list(notes_collection.find({"subject": subject}))

    if not subject_docs:
        return []

    # Use the notes_embedding_model instead for this function
    query_embedding = notes_embedding_model.encode(question).tolist()

    docs_with_scores = []
    for doc in subject_docs:
        if "embedding" in doc and isinstance(doc["embedding"], list):
            doc_embedding = doc["embedding"]

            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            docs_with_scores.append((doc, similarity))

    docs_with_scores.sort(key=lambda x: x[1], reverse=True)
    top_docs = docs_with_scores[:top_k]

    similar_docs = []
    for doc, score in top_docs:
        content = doc.get("text", "")

        metadata = {
            "subject": doc.get("subject", ""),
            "page": doc.get("page", 0),
            "source": doc.get("source", ""),
            "similarity_score": f"{score:.4f}",
        }

        similar_docs.append(Document(page_content=content, metadata=metadata))

    return similar_docs


# Gemini API Configuration
genai.configure(api_key=GEMINI_API)


def get_gemini_response(question, context):
    """Sends a request to the Gemini API to get the answer."""
    prompt = f"Respond only in plaintext\n Answer the following question based on the given context. If the context is not relevant to the question, don't mention that the context is irrelevant - simply answer the question using your knowledge while ignoring the context:\n\nQuestion: {question}\nContext: {context}"

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text


def get_knowledge_based_response(question, subject, level):
    prompt = f"You are an expert educational assistant specializing in {subject} at the {level} level. Answer the following question using your knowledge of the {level} syllabus for {subject}. Be accurate, educational, and provide appropriate examples or explanations. If the question is outside the scope of the {level} {subject} syllabus, mention this but still provide helpful information.\n\nQuestion: {question}\n\nPlease provide a comprehensive and educational answer suitable for a student at the {level} level."

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text


# Flask endpoint
@app.route("/query", methods=["POST"])
def query_endpoint():
    """Handles the query and returns results."""
    data = request.json
    question = data.get("question")
    course = data.get("course")
    category = data.get("category")

    print(course, category, question)

    # Validate inputs
    if not question or not course or not category:
        return (
            jsonify(
                {
                    "error": "Invalid input. 'question', 'course', and 'category' are required."
                }
            ),
            400,
        )

    # Get vector search results
    results = get_query_results(question, course, category)
    # print("Results: ",results)
    context = "\n".join([result["text"] for result in results])

    # Get answer from Gemini API
    answer = get_gemini_response(question, context)
    print("\n\n\nAnswer: \n\n", answer)
    return jsonify({"question": question, "answer": answer, "context": context})


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200


@app.route("/ask", methods=["POST"])
def ask_endpoint():
    data = request.json
    subject = data.get("subject")
    user_query = data.get("user_query")
    level = data.get("level")

    if not subject or not user_query or not level:
        return (
            jsonify(
                {
                    "error": "Invalid input. 'subject', 'user_query', and 'level' are required."
                }
            ),
            400,
        )

    try:
        similar_docs = retrieve_similar_docs(subject, user_query, top_k=3)

        if similar_docs and len(similar_docs) > 0:
            context = "\n".join([doc.page_content for doc in similar_docs])
            answer = get_gemini_response(user_query, context)

            return jsonify({"question": user_query, "answer": answer})
        else:
            answer = get_knowledge_based_response(user_query, subject, level)

            return jsonify({"question": user_query, "answer": answer})

    except Exception as e:
        print(f"Error in ask endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
