import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import torch

# Load the pre-trained model and tokenizer
def load_model_and_tokenizer(model_name="microsoft/DialoGPT-medium"):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.config.pad_token_id = model.config.eos_token_id  # Set pad_token_id to eos_token_id
    return model, tokenizer

# Sentiment analysis pipeline to analyze user's mood dynamically
def get_sentiment(text):
    sentiment_pipeline = pipeline("sentiment-analysis", device=0 if torch.cuda.is_available() else -1)  # Specify device for GPU if available
    sentiment = sentiment_pipeline(text)[0]
    return sentiment['label'], sentiment['score']

# Function to collect user information (name, age, and mood)
def collect_user_info():
    print("Welcome! Let's get to know you a little better.")
    name = input("What's your name? ")
    age = input("How old are you? ")
    mood = input("How are you feeling today? (e.g., happy, sad, anxious) ")

    # Return the collected information
    return name, age, mood

# Function to preprocess input text, preparing the input for the model
def preprocess_input(input_text, tokenizer, chat_history_ids=None):
    new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')
    
    if chat_history_ids is not None:
        # Concatenate new input with past conversation history (limited to 1024 tokens)
        bot_input_ids = torch.cat([chat_history_ids[:, -1024:], new_user_input_ids], dim=-1)
    else:
        bot_input_ids = new_user_input_ids
    
    # Create attention mask to focus on relevant tokens
    attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long)
    
    return bot_input_ids, attention_mask

# Function to generate a response based on the user's input
def generate_response(model, tokenizer, input_text, chat_history_ids=None, user_info=None):
    input_ids, attention_mask = preprocess_input(input_text, tokenizer, chat_history_ids)

    # Generate response using the model with do_sample=True to avoid warnings
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=1000,
            num_beams=5,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            do_sample=True,  # Set to True to avoid warnings
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=True,
        )

    # Get the updated chat history and decode the response
    chat_history_ids = output
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Sentiment analysis to dynamically adapt the response
    sentiment_label, sentiment_score = get_sentiment(input_text)
    
    # Use sentiment to guide the response dynamically
    if sentiment_label == 'NEGATIVE':
        # More empathetic response for negative sentiment
        response = f"Vivek, I'm really sorry to hear you're feeling down. It’s okay to feel like this, and it’s important to take care of your mental health. You might want to try some relaxation techniques, like deep breathing or mindfulness. If you're comfortable, reaching out to a therapist or talking to a friend might also help you feel supported. I'm here to listen if you want to talk more."
    elif sentiment_label == 'POSITIVE':
        # Encouraging response for positive mood
        response = f"That's awesome, Vivek! It sounds like you're feeling great. Keep up the good work, and make sure to keep engaging in activities that make you happy and keep your mood up! Don’t forget to stay connected with loved ones and keep up with healthy habits like exercise and rest!"
    else:
        # Neutral mood response
        response = f"Vivek, I can sense that you're not feeling extremely happy or sad, but that’s okay. Sometimes we go through neutral moods. Taking care of yourself through regular exercise, healthy eating, and enough rest can really help improve your mood over time. If you feel like chatting more, I’m always here!"

    return response, chat_history_ids

# Main chat function to interact with the user
def chat():
    model, tokenizer = load_model_and_tokenizer()

    # Collect user information (name, age, mood)
    name, age, mood = collect_user_info()
    user_info = (name, age, mood)

    chat_history_ids = None  # Initialize chat history

    print("Bot is ready to chat! Type 'exit' to end the conversation.")
    while True:
        user_input = input(f"You ({name}): ")
        
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        # Generate a response from the model based on the user's input and the collected info
        response, chat_history_ids = generate_response(model, tokenizer, user_input, chat_history_ids, user_info)
        
        # Print the bot's response
        print(f"Bot: {response}")

# Run the chatbot interaction
if __name__ == "_main_":
    chat()