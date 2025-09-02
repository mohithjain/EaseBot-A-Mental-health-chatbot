import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load GPT-J model
def load_gpt_j_model():
    model_name = "EleutherAI/gpt-j-6B"

    # Print status for tokenizer initialization
    print("Initializing the tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise

    # Print status for model loading
    print("Loading the GPT-J model. This may take a while...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        print("GPT-J model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    # Print device information
    print(f"Model is loaded on device: {device.upper()}")

    return tokenizer, model

# Test the function if run as a standalone script
if __name__ == "__main__":
    print("Checking system and loading model components...")
    try:
        tokenizer, model = load_gpt_j_model()
        print("Model and tokenizer are ready for use!")
    except Exception as e:
        print(f"An error occurred during loading: {e}")
