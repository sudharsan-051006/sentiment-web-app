🎬 Movie Review Sentiment Analyzer
A full-stack AI-powered web application that analyzes movie reviews from CSV files and provides clear sentiment insights using a fine-tuned Transformer model.
🔗 Live App
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


Fine-tuned on: IMDb Movie Reviews Dataset


Classes:


POSITIVE


NEGATIVE


UNCERTAIN (low-confidence predictions)




Frameworks Used:


Hugging Face Transformers


PyTorch





🏗️ Project Structure
├── app.py                  # FastAPI backend
├── Dockerfile              # Docker configuration
├── requirements.txt        # Python dependencies
├── sentiment_finetuned_v2/ # Fine-tuned transformer model
├── templates/
│   └── index.html          # Frontend UI
├── README.md               # Project documentation


📄 Input File Format
The application accepts a CSV file with a single column named reviews.
Example: movie_reviews.csv
reviews
This movie was absolutely amazing!
The plot was boring and predictable.
Decent acting but poor screenplay.

⚠️ Important Notes


The column name must be exactly reviews


Each row should contain one review


No empty rows are recommended



📊 Output
After processing:


The app displays sentiment distribution visually


A downloadable CSV is generated with an added sentiment label for each review



🛠️ Tech Stack


Backend: FastAPI, Python


Frontend: HTML, CSS, JavaScript


ML Model: DistilBERT (Hugging Face)


Deployment: Docker + Hugging Face Spaces



▶️ Run Locally
git clone https://github.com/sudharsan-051006/Movie-Review-Sentiment-Analyzer.git
cd Movie-Review-Sentiment-Analyzer
pip install -r requirements.txt
python app.py

Then open:
👉 http://localhost:8000

📌 Use Cases


Movie review analysis


NLP sentiment classification demos


AI/ML portfolio project


Learning full-stack ML deployment



👤 Author
🔗 GitHub: https://github.com/sudharsan-051006

⭐ If you like this project, don’t forget to star the repository!
