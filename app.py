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

        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # fully connected layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # optional => shape (num of layers, batch size, hidden size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.rnn(x, h0) 
        # 1st value = hidden state of all the timesteps => (batch, seq_len, hidden size)
        # 2nd value = final hidden state of last timestep

        out = self.fc(out[:, -1, :])
        return out

# -----------------------------
# Load Vectorizer
# -----------------------------
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -----------------------------
# Load Model
# -----------------------------
input_size = len(vectorizer.get_feature_names_out())

model = RNN(input_size)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
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