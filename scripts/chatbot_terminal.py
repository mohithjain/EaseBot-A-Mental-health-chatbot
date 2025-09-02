import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Load user data from user_info.json
def load_user_data():
    if os.path.exists("user_info.json"):
        with open("user_info.json", "r") as f:
            return json.load(f)
    return []

# Function to save user data
def save_user_data(data):
    with open("user_info.json", "w") as f:
        json.dump(data, f, indent=4)

# Function to collect user information
def collect_user_info():
    user_data = load_user_data()
    user_info = {}

    print("Chatbot: Before we begin, I'd like to know a little bit about you.")
    user_name = input("Chatbot: What is your name? ")

    # Check if the user already exists
    existing_user = next((user for user in user_data if user.get("name") == user_name), None)

    if existing_user:
        print(f"Chatbot: Welcome back, {user_name}! Let's dive into our conversation.")
        return existing_user
    else:
        user_info['name'] = user_name
        user_info['age'] = input("Chatbot: How old are you? ")
        user_info['occupation'] = input("Chatbot: What is your occupation? ")
        user_info['mood'] = input("Chatbot: How would you rate your current mood on a scale of 1 to 10 (1 being very sad, 10 being very happy)? ")

        print("Chatbot: Thank you for sharing that. Let's move on to some questions to understand how you're feeling.")

        user_info['stress'] = input("Chatbot: On a scale of 1 to 10, how stressed are you right now? ")
        user_info['sleep'] = input("Chatbot: How well have you been sleeping recently (1 to 10)? ")
        user_info['social'] = input("Chatbot: How would you rate your level of social interaction lately (1 to 10)? ")
        user_info['support'] = input("Chatbot: Do you feel you have enough support from friends/family right now (yes/no)? ")
        user_info['mood_trends'] = input("Chatbot: Have you noticed any changes in your mood recently? (yes/no) ")

        print("\nChatbot: Thanks for sharing! I’ll now remember this information as we chat.")

        # Add new user to the data
        user_data.append(user_info)
        save_user_data(user_data)

        return user_info


# Load the fine-tuned model and tokenizer
def load_model_and_tokenizer(model_path="./finetuned_model"):
    try:
        print("Loading fine-tuned model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.config.pad_token_id = model.config.eos_token_id  # Keep pad_token same as eos_token
        print("Model and tokenizer loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

# Sentiment analysis pipeline with error handling
def get_sentiment(text):
    try:
        # Specify the model explicitly to avoid default model warning
        sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", tokenizer="distilbert-base-uncased-finetuned-sst-2-english", device=0 if torch.cuda.is_available() else -1)
        sentiment = sentiment_pipeline(text)[0]
        return sentiment['label']
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return "NEUTRAL"

# Preprocess input for model
def preprocess_input(input_text, tokenizer, chat_history_ids):
    # Tokenize the input and generate attention_mask
    encoding = tokenizer(input_text + tokenizer.eos_token, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    if chat_history_ids is not None:
        # Concatenate the previous chat history with new input
        bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1)
        bot_attention_mask = torch.cat([chat_history_ids.new_ones(chat_history_ids.shape), attention_mask], dim=-1)
    else:
        bot_input_ids = input_ids
        bot_attention_mask = attention_mask
    
    return bot_input_ids, bot_attention_mask

# Generate model response
def generate_response(model, tokenizer, user_input, chat_history_ids):
    bot_input_ids, bot_attention_mask = preprocess_input(user_input, tokenizer, chat_history_ids)
    
    with torch.no_grad():
        output_ids = model.generate(
            bot_input_ids,
            attention_mask=bot_attention_mask,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    chat_history_ids = output_ids
    response = tokenizer.decode(output_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids

# Chat function
def chat():
    # Loading the fine-tuned model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    print("Welcome to the chatbot! Let's start chatting. Type 'exit' to stop.")
    
    # Collect user's name and greet or ask for preliminary info
    user_info = collect_user_info()
    name = user_info['name']
    
    chat_history_ids = None
    while True:
        # Get user input
        user_input = input(f"You ({name}): ").strip()
        if user_input.lower() == "exit":
            print(f"Goodbye {name}! Take care.")
            break

        # Sentiment analysis (for better interaction)
        sentiment = get_sentiment(user_input)

        # Generate response
        response, chat_history_ids = generate_response(model, tokenizer, user_input, chat_history_ids)

        # Modify response based on sentiment
        if sentiment == "NEGATIVE":
            print(f"Bot: {response} ...I'm here if you need someone to talk to.")
        elif sentiment == "POSITIVE":
            print(f"Bot: {response} That's great to hear! 😊")
        else:
            print(f"Bot: {response}")

if __name__ == "__main__":
    chat()
