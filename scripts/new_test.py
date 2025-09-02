from transformers import pipeline
import torch

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Sentiment analysis pipeline
def analyze_sentiment(text):
    sentiment_analyzer = pipeline("sentiment-analysis", device=0 if torch.cuda.is_available() else -1)  # Use GPU if available
    result = sentiment_analyzer(text)[0]
    return result["label"], result["score"]  # e.g., "POSITIVE", 0.95

# Detect uncertainty in user input
def detect_uncertainty(user_input):
    uncertain_phrases = ["I don't know", "maybe", "not sure", "I'm not sure", "not really"]
    return any(phrase in user_input.lower() for phrase in uncertain_phrases)

# Generate response from the model
def generate_response(model, tokenizer, user_input):
    model.eval()
    # Tokenize the user input with attention mask
    inputs = tokenizer.encode_plus(
        user_input,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Generate response using the model
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,  # Pass attention mask explicitly
        max_length=100,  # Limit response length
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response.strip()
