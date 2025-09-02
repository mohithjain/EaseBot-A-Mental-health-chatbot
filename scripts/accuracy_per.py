import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

def calculate_perplexity(model, tokenizer, texts):
    """
    Calculate perplexity for a list of texts using the given model and tokenizer.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_words = 0

    for text in texts:
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        # Move inputs to GPU if available
        if torch.cuda.is_available():
            inputs = {key: val.to("cuda") for key, val in inputs.items()}
            model.to("cuda")

        # Compute the model outputs with labels as input_ids
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        
        # Add the loss for this text
        loss = outputs.loss
        total_loss += loss.item() * inputs.input_ids.size(1)  # Multiply by sequence length
        total_words += inputs.input_ids.size(1)  # Total number of words

    # Calculate average loss and perplexity
    avg_loss = total_loss / total_words
    perplexity = math.exp(avg_loss)
    return perplexity

def map_perplexity_to_percentage(perplexity, baseline=100):
    """
    Map perplexity to a percentage score based on the baseline perplexity.
    """
    if perplexity > baseline:
        return 0.0  # Clamp to 0 if perplexity exceeds baseline
    accuracy_percentage = 100 * (1 - (perplexity / baseline))
    return max(0, accuracy_percentage)

# Load your fine-tuned model and tokenizer
def load_model_and_tokenizer(model_path="./finetuned_model"):
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.config.pad_token_id = model.config.eos_token_id  # Ensure pad_token is correctly set
    print("Model and tokenizer loaded successfully!")
    return model, tokenizer

if __name__ == "__main__":
    # Example test dataset
    test_texts = [
        "I cant stop overthinking. How can I calm my mind?",
        "Im scared of commitment. How do I overcome it?",
        "I sometimes feel sad for no reason. Is that normal?",
        "How do I manage stress in my career?",
        "Im feeling anxious about the future. How can I manage it?"
        "How can I improve my sleep quality?"
        "I keep overthinking everything."
        "What can I do to boost my confidence?"
        "How do I manage career-related anxiety?"
        "How can I stop procrastinating and be more productive?"
        "How do I handle the loneliness of living alone in a new city?"
        "How do I know if Im on the right path in my career?"
        "Im feeling disconnected from my friends lately. How can I reconnect?"
        "How do I balance family expectations with my own goals?"
        "How do I know if I need therapy?"
        "How do I stop feeling like Im always anxious?"
        "How do I maintain good physical health during busy times?"
        "I feel pressured by societal expectations to succeed. How do I handle it?"
        "How can I stop stressing about things out of my control?"
        "How do I deal with exam anxiety?"
        
    ]

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Calculate perplexity
    perplexity = calculate_perplexity(model, tokenizer, test_texts)
    print(f"Model Perplexity: {perplexity:.4f}")

    # Map perplexity to percentage
    baseline_perplexity = 100  # Example baseline
    accuracy_percentage = map_perplexity_to_percentage(perplexity, baseline=baseline_perplexity)
    print(f"Model Accuracy (Percentage): {accuracy_percentage:.2f}%")
