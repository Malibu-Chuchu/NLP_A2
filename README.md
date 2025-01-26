# NLP Assignment2 Language Model 
## st124952 Patsachon Pattakulpong

## How to Preprocessing Star Wars work! 
- Step 1 Load dataset: (dataset = datasets.load_dataset('myamjechal/star-wars-dataset'))
- Step 2 Preprocessing
        - 2.1 Tokenizing: Which is Simply tokenize the given text to tokens by splitting sentences into tokens using get_tokenizer("basic_english").
        - 2.2 Numericalization: We’ll configure torchtext to add any word that appears at least three times in the dataset to the vocabulary. This helps keep the vocabulary size manageable. Additionally, we’ll ensure that the special tokens `unk` (unknown) and `eos` (end of sequence) are included in the vocabulary.
- step 3 Preparing a batch loader: Which on my assignment i choose batch_size = 128
- step 4 Modeling: We create `LSTMLanguageModel` class to defines an LSTM-based language model. It consists of an embedding layer, an LSTM layer, a dropout layer, and a fully connected layer for predicting the next word in a sequence. The model initializes its weights using uniform distributions and defines methods to handle hidden states. The `forward` method processes input sequences through the embedding, LSTM, and fully connected layers, applying dropout for regularization and returning predictions along with updated hidden states.
- step 5 Training: The process follows a straightforward approach. One thing to keep in mind is that some of the input sequences fed into the model might contain parts from different sequences in the original dataset or be a subset of a single sequence, depending on the decoding length. Because of this, I reset the hidden state at the start of each epoch. This approach assumes that the next batch of sequences is likely a continuation of the previous ones in the original dataset.
  - Model Parameters tune for training process
      - vocab_size = 5752
      - emb_dim = 1024
      - hid_dim = 1024           
      - num_layers = 2
      - dropout_rate = 0.65
      - lr = 1e-3
   - The model has 28,579,448 trainable parameters

- Training process : Train the model to predict the next token in a sequence using Cross-Entropy Loss (nn.CrossEntropyLoss()).
    - Forward : Get prediction (probabilities over the vocabulary) and updated hidden states.
    - Compute Loss : Use Cross-Entropy Loss to compare predictions with the validation tokens.
    - Backward : Compute gradients via backpropagation
    - Training with 50 epochs

## How to run the application:
- Step1 : Download model files from this link : https://drive.google.com/file/d/1I5EZ0uuKDJZ9kwOyPVj12bX5haoavnqN/view?usp=sharing
- Step2 : Move best-val-lstm.pt into Model/
- Step3 : python app.py

## Demo application for Star Wars Text Generated web
![App Screenshot](assets/a2-screenshot.jpg)

![App Screenshot](assets/a2-screenshot2.jpg)

## Citation: https://huggingface.co/datasets/myamjechal/star-wars-dataset

