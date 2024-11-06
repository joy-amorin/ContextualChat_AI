from flask import Flask, jsonify, request
from flask_swagger_ui import get_swaggerui_blueprint
from langchain_community.document_loaders import DirectoryLoader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
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

# Load a summary pipeline if the context is very long
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

#Initializes the FAISS index for storing embeddings
dimension = 384
index = faiss.IndexFlatL2(dimension)

#Load DistilGPT-2 model and tokenizer
gpt_model_name = './results'
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
        filtered_context = find_relevant_paragraphs(context, question_embedding)

        # GPT-2 to generate an answer based on the context and the question.
        input_text = f"Contexto: {filtered_context}\n\nUsa el contexto anterior para responder a la siguiente pregunta: {question}\nRespuesta:"

        input_ids = tokenizer.encode(input_text, return_tensors='pt')

        #Set the attention mask
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        output = gpt_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100,
            no_repeat_ngram_size=2,
            temperature=0.3,
            do_sample=True
        )
        
        #Convert output to token_ids
        token_ids = output.sequences[0].tolist()

        #Decode using token_ids
        response_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    else:
        response_text = "Lo siento, no encontré una respuesta relevante a tu pregunta."

    return jsonify({"response": response_text})


def find_relevant_paragraphs(context, question_embedding, max_length=1024):
    paragraphs = context.split("\n\n")
    relevant_paragraphs = []
    
    # Generate embeddings for each paragraph
    paragraph_embeddings = embedding_model.encode(paragraphs, convert_to_tensor=True).cpu()

    # Calculate the similitari of each paragraph
    for i, paragraph in enumerate(paragraphs):
        paragraph_embedding = paragraph_embeddings[i].unsqueeze(0)

        similarity = cosine_similarity(question_embedding.numpy(), paragraph_embedding.numpy())

        if similarity.item() > 0.5:
            relevant_paragraphs.append(paragraph)

    if relevant_paragraphs:
        joined_context = "\n".join(relevant_paragraphs)
    else:
        joined_context = context

    if len(joined_context) > max_length:

        #Divide the text into smaller fragments
        fragments = [joined_context[i:i + max_length] for i in range(0, len(joined_context), max_length)]
        responses = []
        for fragment in fragments:
            summary = summarizer(fragment, max_length=150, min_length=30, do_sample=False)
            responses.append(summary[0]['summary_text'])

        joined_context = " ".join(responses)

    return joined_context

if __name__ == '__main__':
    app.run(debug=True)
