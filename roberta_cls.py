import numpy as np
import torch
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("Yelp/yelp_review_full")

# Split training set: 90% Train, 10% Validation
train_val_split = dataset['train'].train_test_split(test_size=0.1, seed=42)

dataset_split = {
    'train': train_val_split['train'],
    'validation': train_val_split['test'],
    'test': dataset['test']
}

print(f"Data Split Sizes: Train={len(dataset_split['train'])}, Val={len(dataset_split['validation'])}, Test={len(dataset_split['test'])}")

# Tokenization using RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = {}
for split in dataset_split:
    tokenized_datasets[split] = dataset_split[split].map(tokenize_function, batched=True)
    tokenized_datasets[split].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, predictions)
    
    f1 = f1_score(labels, predictions, average='macro')
    
    return {
        'accuracy': acc,
        'f1_macro': f1  
    }

# Initialize the RoBERTa model
model = RobertaForSequenceClassification.from_pretrained(
    'roberta-base', 
    num_labels=5
)

training_args = TrainingArguments(
    output_dir='./results_roberta',
    learning_rate=2e-5,
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    
    eval_strategy="epoch", 
    save_strategy="epoch",       
    load_best_model_at_end=True, 
    
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    
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
predictions_output = trainer.predict(tokenized_datasets['test'])
y_preds = np.argmax(predictions_output.predictions, axis=1)
y_true = tokenized_datasets['test']['label'].numpy()

# Print Classification Report
print("\nFinal Test Results:")
print(classification_report(y_true, y_preds, target_names=['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars']))

# Plot Confusion Matrix
conf_mat = confusion_matrix(y_true, y_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars'], 
            yticklabels=['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - RoBERTa Model')

# Save the figure
plt.savefig("confusion_matrix_roberta_cls.png", dpi=300)
plt.close()
