from datasets import load_dataset
from transformers import AutoTokenizer

#Load the dataset from text file
dataset = load_dataset('text', data_files='fine_tuning_dataset.txt')

#Load the tokenizer of the model (distilgpt2)
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')

# Establecer el token de relleno como el token de fin de secuencia
tokenizer.pad_token = tokenizer.eos_token

#Define the tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

#Apply tokenization to the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

print("Dataset tokenizado:")
print(tokenized_dataset)
