from datasets import load_dataset, Value
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Yelp Review Full dataset
dataset = load_dataset("yelp_review_full")

# Cast label column to float for regression
dataset["train"] = dataset["train"].cast_column("label", Value("float32"))
dataset["test"]  = dataset["test"].cast_column("label",  Value("float32"))

# Split the original training set into train + validation
train_valid = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = train_valid["train"]
valid_dataset = train_valid["test"]
test_dataset = dataset["test"]

# Load DistilBERT tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Preprocessing / tokenization function
def preprocess(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,   # dynamic padding in the data collator
        max_length=256,
    )

# Apply preprocessing to train, validation, and test splits
encoded_train = train_dataset.map(preprocess, batched=True)
encoded_valid = valid_dataset.map(preprocess, batched=True)
encoded_test  = test_dataset.map(preprocess,  batched=True)

# Data collator for dynamic padding within each batch
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load DistilBERT with a single-output regression head
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=1,         # single scalar output → regression
)
model.config.problem_type = "regression"

# Define regression metrics: MSE, MAE, and rounded accuracy on validation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.squeeze(logits)
    labels = labels.astype(np.float32)      # ensure float for regression

    mse = np.mean((preds - labels) ** 2)
    mae = np.mean(np.abs(preds - labels))

    # Round predictions to nearest integer in [0, 4]
    rounded = np.clip(np.rint(preds), 0, 4)
    star_acc = np.mean(rounded == labels)

    return {
        "mse": mse,
        "mae": mae,
        "star_accuracy": star_acc,
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./yelp_distilbert_reg",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="mse",
    greater_is_better=False,  # lower MSE is better
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_train,
    eval_dataset=encoded_valid,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate on validation set
print("Validation performance (regression):")
print(trainer.evaluate())

# Predict on the held-out test set
test_outputs = trainer.predict(encoded_test)
test_logits = test_outputs.predictions
test_labels = test_outputs.label_ids.astype(np.float32)  # true labels 0–4 as floats

# Continuous predictions
test_preds_continuous = np.squeeze(test_logits)

# Round continuous predictions to the nearest integer in [0, 4]
label_ids = [0, 1, 2, 3, 4]
test_preds_rounded = np.clip(np.rint(test_preds_continuous), 0, 4)

# Confusion matrix
cm_reg = confusion_matrix(test_labels, test_preds_rounded, labels=label_ids)

print("Confusion matrix on the test set (rows = true labels, columns = rounded predicted labels):")
print(cm_reg)

# Plot heatmap
plt.figure(figsize=(6, 5))
star_labels = [str(l + 1) for l in label_ids]  # display 1–5 instead of 0–4
sns.heatmap(
    cm_reg,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=star_labels,
    yticklabels=star_labels,
)
plt.xlabel("Rounded predicted rating")
plt.ylabel("True rating")
plt.title("Confusion Matrix (DistilBERT Regression, Test Set)")
plt.tight_layout()
plt.show()

