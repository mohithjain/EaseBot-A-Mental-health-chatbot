# 🤖 EaseBot: A Mental Health Companion Chatbot  

EaseBot is a **terminal-based mental health companion chatbot** designed for **young adults (ages 19–25)**.  
It acts as a **friend and listener**, helping users express emotions, reflect on their mental state, and receive **empathetic, non-judgmental guidance** during difficult times.  

Unlike generic bots, EaseBot is **trained on age-specific and emotion-focused datasets**, making it **empathetic, context-aware, and tailored** to the unique challenges faced by young adults.  

⚠️ *Note: This project runs directly in the terminal (no GUI).*  

---

## ✨ Key Features  

- 🧑‍🤝‍🧑 **Age-based customization** – optimized for the 19–25 age group.  
- 💬 **Emotion-aware conversations** – powered by curated mental health datasets.  
- 📊 **Feedback loop** – captures user feedback to continuously improve responses.  
- ⚡ **Fine-tuned with Hugging Face Transformers** for natural, empathetic dialogue.  
- 🖥 **Terminal-based interaction** – lightweight, distraction-free, and easy to run anywhere.  

---

## 📂 Repository Structure  

EaseBot-A-Mental-health-chatbot/
│
├── datasets/ # JSON datasets categorized by age & emotions
├── feedback_data/ # Stores user feedback data
├── finetuned_model/ # Trained model files (stored via Git LFS)
├── finetuning_results/ # Training checkpoints & results (via Git LFS)
├── scripts/ # Helper scripts (training, evaluation, visualization)
│
├── chatbot_terminal.py # Main entry point to run the chatbot
├── user_info.json # Stores user-specific details
├── .gitattributes # Git LFS tracking configuration
├── requirements.txt # Python dependencies


---

## 🚀 Getting Started  

### 1️⃣ Clone the Repository  

> ⚠️ This repo uses **Git LFS** for large model files. Install Git LFS before cloning.  

```bash
# Install Git LFS (if not already installed)
git lfs install

# Clone the repository
git clone https://github.com/mohithjain/EaseBot-A-Mental-health-chatbot.git

# Move into the project directory
cd EaseBot-A-Mental-health-chatbot

# Pull large model files tracked by LFS
git lfs pull

2️⃣ Set Up the Environment

Make sure you have Python 3.8+ installed. Then install dependencies:
pip install -r requirements.txt

3️⃣ Run the Chatbot

Launch EaseBot in your terminal:

python chatbot_terminal.py


The bot will:

Ask for your age and details (stored in user_info.json).

Provide customized, empathetic responses for the 19–25 age group.

🧩 Tech Stack in One Line

Python | Hugging Face Transformers | JSON Datasets | Git LFS

⚠ Disclaimer

EaseBot is designed as a companion and not a replacement for professional mental health help.
If you are experiencing severe mental health issues, please consult a certified mental health professional.



