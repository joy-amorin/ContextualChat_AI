import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

#Load the pre-trained model and the tokenizer
model_name = "distilgpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#Add EOS token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

#Load tokenized dataset
dataset = load_dataset('text', data_files='fine_tuning_dataset.txt')

# Tokenize the dataset and make sure that the tags are equal to the input_ids
def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy() # Tags equal to input_ids
    return tokenized_inputs

#Apply tokenization to dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

print(tokenized_datasets) #verify that the dataset has the necessary columns

#Setting of training parameters
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=8e-5,      
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=10,         
    weight_decay=0.01,         
    logging_dir='./logs',
)

#Initialize the trainer with the default settings
trainer = Trainer(
    model=model,                 
    args=training_args,        
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['train'], 
)


trainer.train()

model.save_pretrained('./results')

tokenizer.save_pretrained('./results')

