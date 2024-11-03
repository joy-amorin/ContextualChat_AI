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

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

#Load the embendings model of Sentence Transformers
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

#Initializes the FAISS index for storing embeddings
dimension = 384
index = faiss.IndexFlatL2(dimension)

#Load DistilGPT-2 model and tokenizer
gpt_model_name = 'distilgpt2'
tokenizer = AutoTokenizer.from_pretrained(gpt_model_name)
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
gpt_model = AutoModelForCausalLM.from_pretrained(gpt_model_name, output_hidden_states=True,  return_dict_in_generate=True)

class ChatApi():
    def __init__(self):
        self.texts = []
api = ChatApi()

@app.route('/')
def home():
    return jsonify({"message": "API de chat con IA y RAG"})

#Endpoints for document loading
@app.route('/upload_documents', methods=['POST'])
def upload_documents():
    try:
        print(f"Buscando documentos en: {os.path.abspath('./documents')}")

        if not os.path.exists('./documents'):
            return jsonify({"error": "El directorio './documents' no existe."}), 404
        
        loader = DirectoryLoader('./documents') 
        documents = loader.load()

     # Verify that documents have been uploaded
        if not documents:
            return jsonify({"error": "No se encontraron documentos."}), 404
        
        api.texts = [doc.page_content for doc in documents]
        print(f"Documentos cargados: {len(api.texts)}") #print to test

        #Vectorization of documents
        embeddings = embedding_model.encode(api.texts, convert_to_tensor=True).cpu()

        #Verify the embeddings dimension
        if embeddings.shape[1] != dimension:
            return jsonify({"error": f"La dimensión de los embeddings ({embeddings.shape[1]}) no coincide con la dimensión del índice FAISS ({dimension})."}), 500
        
        index.add(embeddings.numpy().astype('float32')) #add emmbendings to index FAISS
        print(f"Cantidad de vectores en el índice FAISS: {index.ntotal}") #print to test

        if index.ntotal == 0:
            return jsonify({"message": f"Se cargaron {len(api.texts)} documentos y se añadieron al índice."})
        
        return jsonify({"message": f"Se cargaron {len(api.texts)} documentos y se añadieron al índice."})
    except Exception as e:
        print(f"Error al cargar documentos: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
