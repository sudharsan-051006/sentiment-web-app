🎬 Movie Review Sentiment Analyzer

A full-stack AI-powered web application that analyzes movie reviews from CSV files and provides sentiment insights using a fine-tuned Transformer model.

🔗 Live App:
👉 https://sudharsan051006-Movie-Sentiment.hf.space

🚀 Features

📂 Upload CSV files containing movie reviews

🧠 Sentiment analysis using a fine-tuned DistilBERT model

📊 Visual analytics:

Positive / Negative / Uncertain percentages

Progress bars

Pie chart

⬇️ Download processed CSV with sentiment labels

🌐 Deployed as a public web app (no login required)

🧠 Model Details

Base Model: distilbert-base-uncased

Fine-tuned on: IMDb movie reviews dataset

Classes:

POSITIVE

NEGATIVE

UNCERTAIN (low confidence predictions)

Framework: Hugging Face Transformers + PyTorch

🏗️ Project Structure
├── app.py                 # FastAPI backend
├── Dockerfile             # Docker configuration
├── requirements.txt       # Python dependencies
├── sentiment_finetuned_v2 # Fine-tuned model
├── templates/
│   └── index.html         # Frontend UI
├── README.md              # Project documentation


#Input File Format

reviews
first-review
second-review
.......

