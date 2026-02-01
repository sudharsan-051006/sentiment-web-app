from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from transformers import pipeline
import pandas as pd
from io import StringIO


templates = Jinja2Templates(directory="templates")


app = FastAPI(title="Batch Sentiment Analysis API")

# Load model once
sentiment_model = pipeline(
    task="sentiment-analysis",
    model="sentiment_finetuned_v3",
    tokenizer="sentiment_finetuned_v3"
)

LABEL_MAP = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "POSITIVE"
}

@app.post("/predict-file-download")
async def predict_file_download(file: UploadFile = File(...)):

    # 1️⃣ Validate file
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    content = await file.read()
    df = pd.read_csv(StringIO(content.decode("utf-8")))

    if "reviews" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="CSV must contain a column named 'reviews'"
        )

    sentiments = []

    # 2️⃣ Sentiment prediction per review
    for text in df["reviews"]:
        if not isinstance(text, str) or len(text.split()) < 5:
            sentiments.append("UNCERTAIN")
            continue

        result = sentiment_model(text)[0]
        sentiments.append(LABEL_MAP[result["label"]])

    # 3️⃣ Add sentiment column
    df["sentiment"] = sentiments

    # 4️⃣ Calculate percentages
    total = len(df)
    positive_pct = round((sentiments.count("POSITIVE") / total) * 100, 2)
    negative_pct = round((sentiments.count("NEGATIVE") / total) * 100, 2)
    uncertain_pct = round((sentiments.count("UNCERTAIN") / total) * 100, 2)

    # 5️⃣ Convert DataFrame to CSV
    output = StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    # 6️⃣ Return CSV + summary headers
    response = StreamingResponse(
        output,
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=reviews_with_sentiment.csv",
            "X-Positive-Percentage": str(positive_pct),
            "X-Negative-Percentage": str(negative_pct),
            "X-Uncertain-Percentage": str(uncertain_pct),
        }
    )

    return response

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
