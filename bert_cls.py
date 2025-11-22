from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
import evaluate
from sklearn.metrics import confusion_matrix, classification_report 
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Yelp Review Full dataset
dataset = load_dataset("yelp_review_full")

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
    # Tokenize the raw text; we keep labels as they are (0–4)
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,   # dynamic padding will be applied by the data collator
        max_length=256,  # limit max sequence length
    )

# Apply preprocessing to train, validation, and test splits
encoded_train = train_dataset.map(preprocess, batched=True)
encoded_valid = valid_dataset.map(preprocess, batched=True)
encoded_test  = test_dataset.map(preprocess,  batched=True)

# Data collator for dynamic padding within each batch
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load DistilBERT model for 5-class classification
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=5,  # 5 discrete rating classes (0–4)
)

# Define metrics for validation (accuracy and macro-F1)
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
    }

# Training arguments (using validation set for evaluation)
training_args = TrainingArguments(
    output_dir="./yelp_distilbert_cls",
    eval_strategy="epoch",           # evaluate at the end of each epoch on validation
    save_strategy="epoch",           # save a checkpoint at each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_steps=200,
    load_best_model_at_end=True,     # reload the best checkpoint according to eval metric
    metric_for_best_model="f1_macro",
    greater_is_better=True,
)

# Create Trainer with train = train split, eval = validation split
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

# Evaluate on the validation set (optional, for logging)
print("Validation performance:")
print(trainer.evaluate())

# Predict on the held-out test set
test_outputs = trainer.predict(encoded_test)
test_logits = test_outputs.predictions
test_labels = test_outputs.label_ids

# Convert logits to predicted class indices (0–4)
test_preds = np.argmax(test_logits, axis=-1)

target_names = ["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"]
print("\nFinal Test Results:")
print(classification_report(test_labels, test_preds, target_names=target_names, digits=2))

# Compute confusion matrix on the test set
label_ids = [0, 1, 2, 3, 4]
cm = confusion_matrix(test_labels, test_preds, labels=label_ids)

print("Confusion matrix on the test set (rows = true labels, columns = predicted labels):")
print(cm)

# Plot confusion matrix as a heatmap and SAVE as PNG
plt.figure(figsize=(6, 5))
star_labels = [str(l + 1) for l in label_ids]  # display 1–5 instead of 0–4
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=star_labels,
    yticklabels=star_labels,
)
plt.xlabel("Predicted rating")
plt.ylabel("True rating")
plt.title("Confusion Matrix (DistilBERT Classification, Test Set)")
plt.tight_layout()

# Save the figure
plt.savefig("confusion_matrix_distilbert_cls.png", dpi=300)
plt.close()
