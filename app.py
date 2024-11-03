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
    
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('query')

    if not question:
        return jsonify({"error": "Invalid question provided"}), 400
    
    #generate embedding question
    question_embedding = embedding_model.encode(question, convert_to_tensor=True).cpu()

    #question embeddin 2D
    question_embedding = question_embedding.reshape(1, -1)

    #Search for embedding in the FAISS index
    distances, indices = index.search(question_embedding.numpy(), k=1)

    min_distance = distances[0][0]
    document_index = indices[0][0]
    print(f"Distancia mínima: {min_distance}") #print to test

    if document_index != -1 and min_distance < 200:
        print(f"Contexto encontrado: {api.texts[document_index]}") #print to test

        context = api.texts[document_index]
        # GPT-2 to generate an answer based on the context and the question.
        input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
        input_ids = tokenizer.encode(input_text, return_tensors='pt')

        #Set the attention mask
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        output = gpt_model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=100, num_return_sequences=1)
        #Convert output to token_ids
        token_ids = output.sequences[0].tolist()

        #Decode using token_ids
        response_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    else:
        response_text = "Lo siento, no encontré una respuesta relevante a tu pregunta."


    return jsonify({"response": response_text})

if __name__ == '__main__':
    app.run(debug=True)
