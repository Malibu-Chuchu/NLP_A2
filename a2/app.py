from flask import Flask, request, jsonify, render_template
import torch
from torchtext.data.utils import get_tokenizer
from model.lstm import LSTMLanguageModel
import pickle
import platform

app = Flask(__name__)

# Function to load vocabulary from the pickle file
def load_vocabulary(vocab_file):
    with open(vocab_file, "rb") as file:
        return pickle.load(file)

vocab = load_vocabulary("./model/vocab.pkl")
print(f"Vocabulary size: {len(vocab)}")

# Model hyperparameters
embedding_dim = 1024  # Embedding dimension (as per model design)
hidden_dim = 1024     # Hidden state dimension (as per model design)
num_layers = 2        # Number of LSTM layers
dropout = 0.65        # Dropout rate to avoid overfitting
vocab_size = len(vocab)
tokenizer = get_tokenizer('basic_english')
max_sequence_length = 30
sampling_temperature = 0.5
random_seed = 0

# Determine the device (use 'mps' for M1/M2 chips, 'cuda' for GPU, else 'cpu')
device = torch.device("mps" if platform.system() == "Darwin" and torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

# Load the trained LSTM model
def load_trained_model(model_path):
    model = LSTMLanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers, dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

model = load_trained_model('./model/best-val-lstm_lm.pt')

# Function to generate text given a prompt
def generate_text_from_prompt(prompt, model, tokenizer, vocab, max_len, temp, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    tokens = tokenizer(prompt)
    token_indices = [vocab[t] for t in tokens]
    hidden_state = model.init_hidden(1, device)

    with torch.no_grad():
        for _ in range(max_len):
            input_tensor = torch.LongTensor([token_indices]).to(device)
            predictions, hidden_state = model(input_tensor, hidden_state)
            probabilities = torch.softmax(predictions[:, -1] / temp, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1).item()

            # Avoid <unk> token
            while next_token == vocab['<unk>']:
                next_token = torch.multinomial(probabilities, num_samples=1).item()

            # Stop if <eos> token is reached
            if next_token == vocab['<eos>']:
                break

            token_indices.append(next_token)

    idx_to_token = vocab.get_itos()
    return [idx_to_token[idx] for idx in token_indices]

# Main page route
@app.route("/")
def home():
    return render_template("index.html")

# Text generation route (POST request)
@app.route("/generate", methods=["POST"])
def generate_from_prompt():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")

        if not prompt:
            return jsonify({"error": "A prompt is required"}), 400

        # Generate text
        generated_tokens = generate_text_from_prompt(prompt, model, tokenizer, vocab, max_sequence_length, sampling_temperature, device, random_seed)
        generated_text = " ".join(generated_tokens)

        return jsonify({"generated_text": generated_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
