## ContextualChat_AI

This project implements a chat API that generates contextual responses using a pre-trained language model (DistilGPT-2) and a document search system based on FAISS. The API receives a question, searches for relevant context in a set of preprocessed texts, and uses that context to generate a coherent response.

### Table of Contents

-   Description
-   Technologies Used
-   Project Structure
-   Requirements
-   Functioning Details
-   Usage
-   Next Steps

### Description

The API follows these steps:

1.  **Question Reception**: The API receives a question from a user.
2.  **Context Search**: It uses **FAISS** to find the most relevant text fragments in a set of documents.
3.  **Response Generation**: Using the found context and the **DistilGPT-2** model, the API generates a suitable response to the question.

### Technologies Used

#### **Programming Languages**:

-   **Python**: The primary language used for the project's implementation.

#### **Libraries and Frameworks**:

-   **Flask**: A lightweight web framework used to build the RESTful API.
-   **LangChain**: A library designed for text processing and document loading.
-   **FAISS**: Facebook AI’s library used for vector search and efficient index construction.
-   **SentenceTransformers**: To create embeddings for questions and documents.
-   **Transformers (Hugging Face)**: To handle pre-trained NLP models like **DistilGPT-2**.
-   **Scikit-learn**: For computing similarities between embeddings.

#### **Language Models**:

-   **DistilGPT-2**: A lighter version of **GPT-2**, used to generate natural language responses based on the provided context.

### Requirements
This project requires the following libraries:

-   **transformers** (to load and use GPT-2)
-   **torch** (to run the GPT-2 model)
-   **faiss-cpu** (for efficient text search)
-   **sentence-transformers** (to create semantic embeddings)
-   **flask** (to implement the API)
-   **scikit-learn** (for similarity measurements)


### Functioning Details

This project follows a step-by-step process to answer contextual questions using Retrieval-Augmented Generation (RAG) technology. The API workflow is described below:

1.  **Document Loading and Preprocessing**: PDFs are loaded from the `./documents` directory, and the text is extracted using the **PyPDF2** library. The extracted texts are preprocessed to remove special characters and saved into a text file called **`fine_tuning_dataset.txt`**.
    
2.  **Tokenization**: We use the **DistilGPT-2** tokenizer to tokenize the texts extracted from the documents. These tokenized texts are used for fine-tuning the model.
    
3.  **Model Training**: The **DistilGPT-2** model is fine-tuned with the preprocessed and tokenized data using the **Trainer** library from **Hugging Face**. The fine-tuned model is saved in the **`./results`** directory.
    
4.  **Create Question Embeddings**: Upon receiving a user's question, it is converted into an embedding vector using the **Sentence-Transformer** model (paraphrase-MiniLM-L6-v2). This vector representation of the question allows measuring its similarity to the stored documents.
    
5.  **Search for Relevant Context**: We use **FAISS** to search an index of precomputed document embeddings. FAISS calculates the cosine similarity between the question’s embedding and the document embeddings, returning the most relevant fragment. The documents are loaded from the **./documents** directory, which can be updated via the **POST /upload_documents** endpoint.
    
6.  **Filter and Summarize Context (if necessary)**: If the context is too long (more than 1024 tokens), the relevant paragraphs are split into smaller fragments and summarized using the **DistilBART** model to ensure only the most relevant information is passed to **GPT-2**.
    
7.  **Response Generation**: Finally, the filtered context and the question are passed to the **GPT-2** model to generate a coherent response based on the retrieved context. The model uses text generation techniques like **top_p**, **top_k**, and **temperature** to control the randomness and quality of the generated response.

### Usage

1.  **Run the API**: To run the API locally, simply execute the following command:

`python app.py`

2.  **Send a Question**: The API listens on **`http://127.0.0.1:5000/chat`**. You can send **POST** requests to that URL with the following JSON format:

json
`{
    "query": "What is the circular economy?"
}` 

The API will respond with a generated answer based on the relevant context.

### Project Structure
`ContextualChat_AI/


├── app.py # Main code for the chat API

├── fine_tuning_dataset.txt # Text dataset for training

├── documents/ # Directory to upload documents

├── results/ # Directory to store the fine-tuned model

├── tokenizer.py # Code for tokenizing the dataset

└── train.py # Code for training the model with the dataset` |
### Next Steps

1.  **Optimization**: Improve the accuracy of the responses.
2.  **Context Search Improvement**: Refine the search model to consider more semantic aspects, such as synonyms or more complex relationships between concepts.
