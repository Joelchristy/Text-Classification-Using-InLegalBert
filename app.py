import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import pandas as pd
import os

# Load the saved model, tokenizer, and label encoder
output_dir =r"C:\Users\joelc\Desktop\InlegalBert\Indian-Legal-Text-Classification-Mini-Project"
model_path = os.path.join(output_dir, 'model.pth')
tokenizer_path = os.path.join(output_dir, 'tokenizer.pth')
label_encoder_path = os.path.join(output_dir, 'label_encoder.pth')

# Load the& model without the classifier layer
model = AutoModelForSequenceClassification.from_pretrained("law-ai/InLegalBERT")
state_dict = torch.load(model_path)

# Exclude 'classifier' keys from the state dictionary
state_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k}

# Load the state dictionary into the model
model.load_state_dict(state_dict, strict=False)

tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
label_encoder = torch.load(label_encoder_path)

# Create a function to make predictions
def predict(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()
    predicted_category = label_encoder.inverse_transform([predicted_label])[0]
    return predicted_category

# Streamlit app
st.title("Legal Text Classification App")

# Input text box for user to enter legal text
text_input = st.text_area("Enter legal text:", "")

# Make predictions when the user clicks the "Predict" button
if st.button("Predict"):
    if text_input:
        prediction = predict(text_input)
        st.success(f"Predicted Category: {prediction}")
    else:
        st.warning("Please enter legal text for prediction.")
