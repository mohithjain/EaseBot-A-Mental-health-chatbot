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

- 📂 **EaseBot-A-Mental-health-chatbot/**
  - 📂 **datasets/** — JSON datasets categorized by age & emotions
  - 📂 **feedback_data/** — Stores user feedback data
  - 📂 **finetuned_model/** — Trained model files (stored via Git LFS)
  - 📂 **finetuning_results/** — Training checkpoints & results (via Git LFS)
  - 📂 **scripts/** — Helper scripts (training, evaluation, visualization)
  - 🐍 **chatbot_terminal.py** — Main entry point to run the chatbot
  - 📝 **user_info.json** — Stores user-specific details
  - ⚙️ **.gitattributes** — Git LFS tracking configuration
  - 📄 **requirements.txt** — Python dependencies


---

## 🚀 Getting Started  

### 1️⃣ Clone the Repository  

> ⚠ This repo uses **Git LFS** for large model files.  
> Please install Git LFS before cloning.  

```bash
# Install Git LFS (if not already installed)
git lfs install

# Clone the repository
git clone https://github.com/mohithjain/EaseBot-A-Mental-health-chatbot.git

# Move into the project directory
cd EaseBot-A-Mental-health-chatbot

# Pull large model files tracked by LFS
git lfs pull
```
### 2️⃣ Set Up the Environment

Make sure you have Python 3.8+ installed. Then install the dependencies:
```bash
#pip install -r requirements.txt
```
### 3️⃣ Run the Chatbot

To launch the chatbot in the terminal:
```bash
python chatbot_terminal.py
```
The bot will then start interacting with you.
It uses user_info.json for storing age and user details, and tailors responses for the 19–25 age group.

## 💡 Features

🧑‍🤝‍🧑 Age-based customization (19–25 age group)

🤖 Emotion-aware responses trained on mental health datasets

💬 Terminal-based interaction (no GUI required)

📊 Feedback loop to capture user feedback and improve responses

⚡ Fine-tuned model with Hugging Face Transformers

## ⚠ Disclaimer

This chatbot is designed only as a companion and not a replacement for professional help.
If you are experiencing severe mental health issues, please consult a certified mental health professional.

## 👨‍💻 Author

Manvi Sharma
Mohith Jain
