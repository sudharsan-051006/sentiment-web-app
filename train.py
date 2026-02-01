from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# --------------------------------------------------
# 1. Load IMDb dataset
# --------------------------------------------------
raw_dataset = load_dataset("imdb")

# --------------------------------------------------
# 2. Create LARGE BALANCED dataset (IMPORTANT)
# --------------------------------------------------
positive = (
    raw_dataset["train"]
    .filter(lambda x: x["label"] == 1)
    .shuffle(seed=42)
    .select(range(5000))   # 🔥 more data
)

negative = (
    raw_dataset["train"]
    .filter(lambda x: x["label"] == 0)
    .shuffle(seed=42)
    .select(range(5000))   # 🔥 more data
)

dataset = concatenate_datasets([positive, negative])
dataset = dataset.shuffle(seed=42)
dataset = dataset.train_test_split(test_size=0.2)

# --------------------------------------------------
# 3. Load tokenizer FROM BASE MODEL
# --------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

dataset = dataset.map(tokenize, batched=True)

# --------------------------------------------------
# 4. Prepare dataset for Trainer
# --------------------------------------------------
dataset = dataset.rename_column("label", "labels")
dataset = dataset.remove_columns(["text"])
dataset.set_format("torch")

# --------------------------------------------------
# 5. Load BASE model (CLEAN START)
# --------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# --------------------------------------------------
# 6. Training configuration (STRONGER)
# --------------------------------------------------
training_args = TrainingArguments(
    output_dir="./sentiment_model_v3",
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,        # 🔥 train longer
    logging_steps=100,
    save_strategy="no",        # Windows-safe
    report_to="none"
)

# --------------------------------------------------
# 7. Trainer
# --------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

# --------------------------------------------------
# 8. Train
# --------------------------------------------------
trainer.train()

# --------------------------------------------------
# 9. MANUAL SAVE (Windows-safe)
# --------------------------------------------------
model.save_pretrained("sentiment_finetuned_v3")
tokenizer.save_pretrained("sentiment_finetuned_v3")
    