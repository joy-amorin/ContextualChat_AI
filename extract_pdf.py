import os
import PyPDF2
import re

pdf_directory = "/mnt/c/Users/5763/Documents/Desarrollo sostenible"

def count_pdfs_in_directory(directory):
    pdf_count = 0
    if not os.path.exists(directory):
        print(f"El directorio no existe: {directory}")
        return 0

    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            pdf_count += 1 

    return pdf_count

def extract_text_from_pdfs(directory):
    all_texts = []

    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(directory, filename) 
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ''
                    for page in reader.pages:
                        text += page.extract_text() + '\n'
                    all_texts.append(text)
            except Exception as e:
                print(f"No se pudo leer el archivo {filename}: {e}")

    return all_texts

pdf_count = count_pdfs_in_directory(pdf_directory)
print(f"Número de archivos PDF encontrados: {pdf_count}")

if pdf_count > 0:
    pdf_texts = extract_text_from_pdfs(pdf_directory)
    print(f"Número de textos extraídos: {len(pdf_texts)}")
else:
    print("No se encontraron archivos PDF para extraer texto.")

def preprocess_texts(texts):
    processed_texts = []
    
    for text in texts:
        #Removing extra blanks and special characters
        cleaned_text = re.sub(r'\s+', ' ', text) 
        cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
        cleaned_text = cleaned_text.lower() 
        
        processed_texts.append(cleaned_text)
    
    return processed_texts

def save_to_file(processed_texts, filename):
    with open(filename, 'w') as f:
        for text in processed_texts:
            f.write(text + '\n')

# Preprocess extracted texts and save them
if pdf_count > 0:
    processed_texts = preprocess_texts(pdf_texts)
    save_to_file(processed_texts, 'fine_tuning_dataset.txt')
    print(f"Se guardaron {len(processed_texts)} textos procesados en 'fine_tuning_dataset.txt'")
else:
    print("No se encontraron textos para procesar.")
