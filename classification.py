import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv(r"C:\Users\joelc\Desktop\InlegalBert\Indian-Legal-Text-Classification-Mini-Project\LegalData.csv")  # Replace with the actual path to your dataset

# Convert categorical labels to numeric values using LabelEncoder
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Split the dataset into training and validation sets
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

# Configure model with the correct number of labels
num_labels = 7
model = AutoModelForSequenceClassification.from_pretrained("law-ai/InLegalBERT", num_labels=num_labels)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")

# Define a PyTorch Dataset
class CustomDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]
        return {"text": text, "label": label}

# Create instances of the CustomDataset for training and validation
train_dataset = CustomDataset(texts=train_df["text"], labels=train_df["label"])
valid_dataset = CustomDataset(texts=valid_df["text"], labels=valid_df["label"])

# Tokenize the datasets
train_encodings = tokenizer(train_dataset.texts.tolist(), truncation=True, padding=True, return_tensors="pt")
valid_encodings = tokenizer(valid_dataset.texts.tolist(), truncation=True, padding=True, return_tensors="pt")

# Attach labels to the tokenized tensors
train_labels = torch.tensor(train_dataset.labels.tolist())
valid_labels = torch.tensor(valid_dataset.labels.tolist())

train_dataset = torch.utils.data.TensorDataset(train_encodings["input_ids"], train_encodings["attention_mask"], train_labels)
valid_dataset = torch.utils.data.TensorDataset(valid_encodings["input_ids"], valid_encodings["attention_mask"], valid_labels)

# Define the DataLoader for training and validation
train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=8)

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
num_epochs = 8  # You can adjust the number of epochs
best_valid_loss = float('inf')
patience = 2
#early_stopping_counter = 0

# Lists to store accuracy values for plotting
train_accuracy_list = []
valid_accuracy_list = []

# Manually specify label_names
#label_names = ['Argument', 'Facts', 'Precedent', 'Ratio of the decision', 'Ruling by Lower Court', 'Ruling by Present Court', 'Statute']

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    all_train_preds = []  
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        #print(f"Batch loss: {loss.item()}")

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).tolist()
        all_train_preds.extend(preds)

    train_accuracy_list.append(accuracy_score(train_df["label"], all_train_preds))

    # Evaluation loop
    model.eval()
    valid_loss = 0.0
    all_preds = []
    with torch.no_grad():
        for batch in valid_loader:
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
            outputs = model(**inputs)
            loss = outputs.loss
            valid_loss += loss.item()
        
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).tolist()
            all_preds.extend(preds)

    valid_loss /= len(valid_loader)
    valid_accuracy = accuracy_score(valid_df["label"], all_preds)

    valid_accuracy_list.append(valid_accuracy)

    print(f"Epoch {epoch + 1}/{num_epochs}, Valid Loss: {valid_loss}, Valid Accuracy: {valid_accuracy}")
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(valid_df["label"], all_preds, target_names=label_encoder.classes_))


    # Check for early stopping
    #if valid_loss < best_valid_loss:
       # best_valid_loss = valid_loss
        #early_stopping_counter = 0
    #else:
        #early_stopping_counter += 1
        #print(f"Early stopping counter: {early_stopping_counter}/{patience}")

        #if early_stopping_counter >= patience:
            #print(f"Early stopping! No improvement in validation loss for {patience} epochs.")
            #break

# Plotting accuracy
plt.plot(range(1, num_epochs + 1), train_accuracy_list, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), valid_accuracy_list, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(r"C:\Users\joelc\Desktop\InlegalBert\Indian-Legal-Text-Classification-Mini-Project/accuracy_plot.png")

# Save the trained model, tokenizer, and label encoder using torch.save
output_dir = r"C:\Users\joelc\Desktop\InlegalBert\Indian-Legal-Text-Classification-Mini-Project"
os.makedirs(output_dir, exist_ok=True)

model_path = os.path.join(output_dir, 'model.pth')
tokenizer_path = os.path.join(output_dir, 'tokenizer.pth')
label_encoder_path = os.path.join(output_dir, 'label_encoder.pth')

torch.save(model.state_dict(), model_path)
tokenizer.save_pretrained(tokenizer_path)
torch.save(label_encoder, label_encoder_path)