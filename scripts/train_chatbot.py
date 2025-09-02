import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset, concatenate_datasets

# Directory for datasets
DATASETS_DIR = "./datasets"  # Path to your datasets

# Load datasets from the datasets directory
def load_datasets():
    datasets = []
    for filename in os.listdir(DATASETS_DIR):
        if filename.endswith(".json"):
            file_path = os.path.join(DATASETS_DIR, filename)
            with open(file_path, "r") as f:
                data = json.load(f)
                
                # Ensure the keys are accessed correctly by checking for case issues
                dataset = Dataset.from_dict({
                    "input_text": [entry.get("Input Text", "") for entry in data],
                    "response_text": [entry.get("Output Response", "") for entry in data],
                })
                datasets.append(dataset)
    return datasets

# Prepare data for fine-tuning
def preprocess_feedback_data(dataset, tokenizer):
    def preprocess_function(examples):
        combined_text = [
            f"User: {input_text}\nEaseBot: {response_text}"
            for input_text, response_text in zip(examples["input_text"], examples["response_text"])
        ]
        encodings = tokenizer(combined_text, truncation=True, padding="max_length", max_length=512)

        # Ensure 'labels' field is created
        encodings['labels'] = encodings['input_ids'].copy()
        return encodings

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    return tokenized_dataset

# Fine-tune model based on datasets
def fine_tune_model(model, tokenizer, datasets):
    # Tokenize the datasets
    tokenized_datasets = [preprocess_feedback_data(dataset, tokenizer) for dataset in datasets]
    
    # Concatenate all tokenized datasets into one
    combined_dataset = concatenate_datasets(tokenized_datasets)
    
    # Split the combined dataset into train and eval datasets
    train_size = int(0.8 * len(combined_dataset))
    train_dataset = combined_dataset.select(range(train_size))
    eval_dataset = combined_dataset.select(range(train_size, len(combined_dataset)))

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./finetuning_results",
        evaluation_strategy="epoch",  # Evaluation strategy
        logging_dir="./logs_feedback",
        num_train_epochs=1,  # Reduced epochs to speed up the training
        per_device_train_batch_size=1,  # Reduced batch size for smaller models
        per_device_eval_batch_size=1,
        logging_steps=5,  # Log every 5 steps
        save_steps=50,  # Save the model after every 50 steps
        save_total_limit=2,
        warmup_steps=50,
        learning_rate=5e-5,
        weight_decay=0.01,
    )

    # Define the data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train the model
    try:
        trainer.train()
    except Exception as e:
        print(f"Training failed: {e}")
    print("Fine-tuning complete. Model updated with datasets.")

# Main process
def train():
    print("Training the chatbot on the provided datasets...")

    # Load datasets
    datasets = load_datasets()
    if not datasets:
        print("No datasets found in the specified directory.")
        return

    print(f"Loaded {len(datasets)} datasets.")

    # Load distilGPT2 model and tokenizer
    model_name = "distilgpt2"  # Using distilgpt2 for faster training on limited hardware
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Set padding token to eos_token
    tokenizer.pad_token = tokenizer.eos_token

    # Fine-tune model on datasets
    fine_tune_model(model, tokenizer, datasets)

    # Save the updated model
    model.save_pretrained("./finetuned_model")
    tokenizer.save_pretrained("./finetuned_model")
    print("Finetuned model saved to ./finetuned_model.")

if __name__ == "__main__":
    train()
