import streamlit as st
import torch
import torch.nn as nn
import pickle
import re

# -----------------------------
# Define Model Architecture
# -----------------------------
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# -----------------------------
# Load Vectorizer & Model
# -----------------------------
@st.cache_resource
def load_model():
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    input_size = len(vectorizer.get_feature_names_out())
    model = RNN(input_size)
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return vectorizer, model

vectorizer, model = load_model()

# -----------------------------
# Preprocessing (match training)
# -----------------------------
def preprocess(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = text.lower().strip()
    return text

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🧠 Sentiment Analysis (PyTorch)")

text = st.text_area("Enter your text here")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned = preprocess(text)

        vec = vectorizer.transform([cleaned]).toarray()
        tensor_input = torch.tensor(vec, dtype=torch.float32).unsqueeze(1)  # (1, 1, 5000)

        with torch.no_grad():
            output = model(tensor_input)
            prob = torch.sigmoid(output).item()
            prediction = 1 if prob > 0.5 else 0

        st.write("---")
        if prediction == 1:
            st.success(f"✅ Positive 😊")
            st.metric("Confidence", f"{prob:.1%}")
        else:
            st.error(f"❌ Negative 😠")
            st.metric("Confidence", f"{1 - prob:.1%}")

        # Debug info (remove later if you want)
        st.caption(f"Raw sigmoid output: {prob:.4f}")