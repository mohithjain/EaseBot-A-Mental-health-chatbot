# EaseBot: A Mental Health Companion Chatbot

EaseBot is a **terminal-based mental health companion chatbot** designed for users aged **19–25**.  
It acts as a **friend and companion**, helping users express emotions, reflect on their mental state, and receive gentle guidance during low times.  

The chatbot is trained on various age-specific and emotion-based datasets, making it more empathetic and tailored to young adults.  
This project does **not** have a GUI — it runs directly in the terminal.

---

## 📂 Repository Structure

EaseBot-A-Mental-health-chatbot/
│
├── datasets/ # JSON datasets categorized by age & emotions
├── feedback_data/ # Stores user feedback data
├── finetuned_model/ # Trained model files (stored via Git LFS)
├── finetuning_results/ # Training checkpoints & results (stored via Git LFS)
├── scripts/ # Helper scripts (training, evaluation, visualization)
│
├── chatbot_terminal.py # Main entry point to run the chatbot
├── user_info.json # Stores user-specific details
├── .gitattributes # Git LFS tracking configuration


---

## 🚀 Getting Started

### 1. Clone the repository

> ⚠️ This repo uses **Git LFS** for large model files.  
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

### 2. Set up the environment
> Make sure you have Python 3.8+ installed. Then install the dependencies:
```bash
pip install -r requirements.txt
