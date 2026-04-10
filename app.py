import streamlit as st
import torch
import torch.nn as nn
import pickle

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

        # ✅ FIX: add seq_len dimension → shape (1, 1, 5000)
        tensor_input = tensor_input.unsqueeze(1)

        # prediction
        with torch.no_grad():
            output = model(tensor_input)
            # ✅ FIX: apply sigmoid before thresholding (model has no sigmoid in forward())
        