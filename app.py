import streamlit as st
import torch
import torch.nn as nn
import pickle

# -----------------------------
# Define Model Architecture
# -----------------------------
class SentimentModel(nn.Module):
    def __init__(self, input_size):
        super(SentimentModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# -----------------------------
# Load Vectorizer
# -----------------------------
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -----------------------------
# Load Model
# -----------------------------
input_size = len(vectorizer.get_feature_names_out())

model = SentimentModel(input_size)
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🧠 Sentiment Analysis (PyTorch)")

text = st.text_area("Enter your text here")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        # preprocess
        vec = vectorizer.transform([text]).toarray()
        tensor_input = torch.tensor(vec, dtype=torch.float32)

        # prediction
        with torch.no_grad():
            output = model(tensor_input)
            prediction = (output > 0.5).int()

        if prediction.item() == 1:
            st.success("Positive 😊")
        else:
            st.error("Negative 😠")