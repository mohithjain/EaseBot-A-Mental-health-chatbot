import json
import os
from datetime import datetime
from new_test import generate_response, analyze_sentiment, detect_uncertainty
from load_models import load_gpt_j_model

# Paths for user data
USER_DATA_DIR = "./user_data"

# Load or initialize user profile
def load_user_profile(user_name):
    user_file = os.path.join(USER_DATA_DIR, f"{user_name}.json")
    if os.path.exists(user_file):
        with open(user_file, "r") as f:
            return json.load(f)
    else:
        return {
            "name": user_name,
            "age": None,
            "occupation": None,
            "preferences": {"advice_style": "friendly", "tone": "supportive"},
            "history": [],
            "mood_trends": [],
            "current_goal": None
        }

# Save user profile
def save_user_profile(user_name, profile_data):
    os.makedirs(USER_DATA_DIR, exist_ok=True)
    user_file = os.path.join(USER_DATA_DIR, f"{user_name}.json")
    with open(user_file, "w") as f:
        json.dump(profile_data, f, indent=4)

# Onboarding: Collect initial user details
def collect_user_details(user_profile):
    print("Welcome to EaseBot! Let's get to know you better.")
    if not user_profile["age"]:
        user_profile["age"] = input("What's your age? ").strip()
    if not user_profile["occupation"]:
        user_profile["occupation"] = input("What's your occupation? ").strip()
    advice_style = input("Do you prefer professional or friendly advice? ").strip().lower()
    tone = input("What tone do you prefer (supportive, motivational, humorous, neutral)? ").strip().lower()
    user_profile["preferences"]["advice_style"] = advice_style
    user_profile["preferences"]["tone"] = tone
    return user_profile

# Chat session
def chat_session(user_profile, model, tokenizer):
    print("Type 'exit' to end the chat.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Thank you! See you next time.")
            break
        
        # Detect uncertainty and adjust responses
        if detect_uncertainty(user_input):
            bot_response = "Could you clarify that for me?"
        else:
            bot_response = generate_response(model, tokenizer, user_input)

        # Sentiment analysis
        sentiment_label, sentiment_score = analyze_sentiment(user_input)

        # Update mood trends
        user_profile["mood_trends"].append({
            "time": str(datetime.now()),
            "input": user_input,
            "sentiment": sentiment_label,
            "score": sentiment_score
        })

        # Save chat history
        user_profile["history"].append({
            "time": str(datetime.now()),
            "user_input": user_input,
            "bot_response": bot_response
        })

        print(f"EaseBot ({sentiment_label}, {sentiment_score:.2f}): {bot_response}")
        save_user_profile(user_profile["name"], user_profile)

# Main
if __name__ == "__main__":
    print("Welcome to EaseBot!")
    user_name = input("What's your name? ").strip()
    user_profile = load_user_profile(user_name)

    if not user_profile.get("age"):
        user_profile = collect_user_details(user_profile)
        save_user_profile(user_name, user_profile)

    # Load GPT-J model
    tokenizer, model = load_gpt_j_model()

    # Display previous chats
    print("Here are your last chats:")
    for entry in user_profile.get("history", [])[-3:]:
        print(f"You: {entry['user_input']}")
        print(f"EaseBot: {entry['bot_response']}")
    print("-----")

    # Start chat session
    chat_session(user_profile, model, tokenizer)
