# ContextualChat_AI

Este proyecto implementa una API de chat que genera respuestas contextuales utilizando un modelo de lenguaje preentrenado (GPT-2) y un sistema de búsqueda de documentos basado en FAISS. La API recibe una pregunta, busca el contexto relevante en un conjunto de textos preprocesados y utiliza ese contexto para generar una respuesta coherente.


## Tabla de contenidos

 - Descripción
 - Tecnologías utilizadas
 - Requisitos
 - Detallesde funcionamiento
 - Próximos pasos

### Descripción

Este proyecto crea un chatbot que responde preguntas basadas en un conjunto de documentos utilizando el modelo GPT-2 y el sistema de búsqueda FAISS para encontrar los fragmentos de texto más relevantes. La API implementa los siguientes pasos:

 1. **Recepción de una Pregunta**: La API recibe una pregunta de un usuario.
 2. **Búsqueda de Contexto**: Utiliza FAISS para encontrar los fragmentos de texto más relevantes en un conjunto de documentos.
 3. **Generación de Respuesta**: Usando el contexto encontrado y el modelo GPT-2, la API genera una respuesta adecuada a la pregunta.

### Tecnologías Utilizadas

Este proyecto se ha desarrollado utilizando una variedad de tecnologías y herramientas. A continuación se detallan las principales:

### **Lenguajes de Programación:**

-   **Python**: El lenguaje principal utilizado para la implementación del proyecto.

### **Bibliotecas y Frameworks:**

-   **Flask**: Framework web ligero utilizado para construir la API RESTful.
-   **LangChain**: Librería diseñada para el procesamiento de texto y carga de documentos.
-   **FAISS**: Biblioteca de Facebook AI utilizada para la búsqueda de vectores y la construcción de índices eficientes.
-   **SentenceTransformers**: Para crear embeddings de preguntas y documentos.
-   **Transformers (Hugging Face)**: Para manejar modelos preentrenados de NLP, como **DistilGPT-2**.

### **Modelos de Lenguaje:**

-   **DistilGPT-2**: Una versión más ligera de GPT-2 utilizada para generar respuestas en lenguaje natural basadas en el contexto proporcionado.
### Requisitos
Este proyecto necesita las siguientes bibliotecas:

-   **transformers** (para cargar y usar GPT-2)
-   **torch** (para ejecutar el modelo GPT-2)
-   **faiss-cpu** (para realizar búsquedas eficientes de texto)
-   **sentence-transformers** (para crear embeddings semánticos)
-   **flask** (para implementar la API)
-   **scikit-learn** (para medidas de similitud)

Instalar todos los requisitos utilizando el siguiente comando:

`pip install -r requirements.txt`

### Uso
1.  **Ejecutar la API**:
    
    Para ejecutar la API localmente, simplemente ejecuta el siguiente comando:
  
    
    `python app.py` 
    
2.  **Enviar una Pregunta**:
    
    La API escucha en `http://127.0.0.1:5000/chat`. Puedes enviar solicitudes POST a esa URL con el siguiente formato:
    
    json
    
    Copiar código
    
    `{
        "query": "¿Qué es la economía circular?"
    }` 
    
    La API responderá con una respuesta generada a partir del contexto relevante.
### Detalles del Funcionamiento

1.  **Cargar el Modelo de Lenguaje GPT-2**: Usamos el modelo GPT-2 de `transformers` para generar respuestas. El modelo es preentrenado y luego fine-tuned con datos relevantes, si es necesario.
    
2.  **Crear Embeddings de la Pregunta**: Utilizamos la librería `sentence-transformers` para convertir la pregunta del usuario en un vector de embedding.
    
3.  **Buscar el Contexto Relevante**: FAISS es utilizado para buscar el documento o párrafo más relevante para la pregunta, calculando la similitud de coseno entre los embeddings de la pregunta y los de los textos preprocesados.
    
4.  **Generación de Respuesta**: Después de encontrar el contexto más relevante, la pregunta y el contexto se pasan a GPT-2 para generar una respuesta coherente.

### Próximos Pasos

1.  **Optimización**: Mejorar la precisión de las respuestas aumentando la cantidad de datos.

3.  **Mejora de la Búsqueda de Contexto**: Refinar el modelo de búsqueda para considerar más aspectos semánticos, como sinónimos o relaciones más complejas entre conceptos.

