from flask import Flask, jsonify, request
from flask_swagger_ui import get_swaggerui_blueprint
from langchain_community.document_loaders import DirectoryLoader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss
import torch
import os

app = Flask(__name__)

SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "ContextualChat-AI"
    }
)

@app.route('/')
def home():
    return jsonify({"message": "API de chat con IA y RAG"})

if __name__ == '__main__':
    app.run(debug=True)
