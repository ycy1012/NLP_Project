import numpy as np
import torch
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, Value

# 1. Load the dataset
dataset = load_dataset("yelp_review_full")

# --- CHANGE 1: Cast labels to float for Regression ---
# This is critical: Regression requires continuous targets (float), not integers.
dataset["train"] = dataset["train"].cast_column("label", Value("float32"))
dataset["test"] = dataset["test"].cast_column("label", Value("float32"))

# 2. Split training set: 90% Train, 10% Validation (Same as your original code)
train_val_split = dataset['train'].train_test_split(test_size=0.1, seed=42)

dataset_split = {
    'train': train_val_split['train'],
    'validation': train_val_split['test'],
    'test': dataset['test']
}

print(f"Data Split Sizes: Train={len(dataset_split['train'])}, Val={len(dataset_split['validation'])}, Test={len(dataset_split['test'])}")

# 3. Tokenization using RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def tokenize_function(examples):
    # Keeping your original 512 length for fairness against your classification model
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = {}
for split in dataset_split:
    tokenized_datasets[split] = dataset_split[split].map(tokenize_function, batched=True)
    # We keep labels as float32 here
    tokenized_datasets[split].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# --- CHANGE 2: Regression Metrics ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Squeeze ensures dimensions match: (Batch_Size, 1) -> (Batch_Size,)
    preds = np.squeeze(logits)
    
    # Calculate MSE (standard regression metric)
    mse = mean_squared_error(labels, preds)
    
    # Calculate "Rounded Accuracy" to compare with Classification models
    # We round the float prediction to the nearest integer (e.g., 3.6 -> 4.0)
    # and clip it to ensure it stays between 0 and 4.
    rounded_preds = np.clip(np.rint(preds), 0, 4)
    acc = accuracy_score(labels, rounded_preds)
    
    return {
        'mse': mse,
        'accuracy': acc  # This is the "Star Accuracy"
    }

# --- CHANGE 3: Model Configured for Regression ---
print("Initializing RoBERTa Regression Model...")
model = RobertaForSequenceClassification.from_pretrained(
    'roberta-base', 
    num_labels=1  # Single output neuron
)
# Explicitly tell Hugging Face to use MSELoss
model.config.problem_type = "regression"

# --- CHANGE 4: Training Arguments for Regression ---
training_args = TrainingArguments(
    output_dir='./results_roberta_regression',
    learning_rate=2e-5,
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    
    eval_strategy="epoch", 
    save_strategy="epoch",       
    load_best_model_at_end=True, 
    
    # For regression, we watch MSE. 
    # IMPORTANT: Lower MSE is better, so greater_is_better=False
    metric_for_best_model="mse",
    greater_is_better=False,
    
    logging_dir='./logs',
    logging_steps=200,
    fp16=True, 
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Start Training
print("Starting Training...")
trainer.train()

# Final Evaluation on Test Set
print("Evaluating on Test Set...")
test_output = trainer.predict(tokenized_datasets['test'])

# Get continuous predictions
test_preds_continuous = np.squeeze(test_output.predictions)
test_labels = test_output.label_ids

# Round predictions for Visualization (Confusion Matrix)
# This effectively converts the regression output back to classification buckets
test_preds_rounded = np.clip(np.rint(test_preds_continuous), 0, 4)

# Print MSE and Accuracy
final_mse = mean_squared_error(test_labels, test_preds_continuous)
final_acc = accuracy_score(test_labels, test_preds_rounded)
print(f"\nFinal Test Results (Regression):")
print(f"MSE: {final_mse:.4f}")
print(f"Rounded Accuracy: {final_acc:.4f}")

# Plot Confusion Matrix
# We use the rounded predictions to generate the heatmap
conf_mat = confusion_matrix(test_labels, test_preds_rounded)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars'], 
            yticklabels=['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars'])
plt.xlabel('Predicted Label (Rounded)')
plt.ylabel('True Label')
plt.title('Confusion Matrix - RoBERTa Regression Model')

# Save the figure
plt.savefig("confusion_matrix_roberta_reg.png", dpi=300)
plt.close()