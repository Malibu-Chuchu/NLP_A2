# NLP Assignment2: Language Model 
## by st124952 Patsachon Pattakulpong

## "Preprocessing Star Wars Text Data: A Guide to Crafting Galactic Narratives"
- **Step 1 Load dataset**: (dataset = datasets.load_dataset('myamjechal/star-wars-dataset'))
  (You can download via this link! : https://huggingface.co/datasets/myamjechal/star-wars-dataset)
- **Step 2 Preprocessing**
        - *2.1 Tokenizing:* Which is Simply tokenize the given text to tokens by splitting sentences into tokens using get_tokenizer("basic_english").
        - *2.2 Numericalization:* We’ll configure torchtext to add any word that appears at least three times in the dataset to the vocabulary. This helps keep the vocabulary size manageable. Additionally, we’ll ensure that the special tokens `unk` (unknown) and `eos` (end of sequence) are included in the vocabulary.
- **Step 3 Preparing a batch loader:** Which on my assignment i choose batch_size = 128
- **Step 4 Modeling:** We create `LSTMLanguageModel` class to defines an LSTM-based language model. It consists of an embedding layer, an LSTM layer, a dropout layer, and a fully connected layer for predicting the next word in a sequence. The model initializes its weights using uniform distributions and defines methods to handle hidden states. The `forward` method processes input sequences through the embedding, LSTM, and fully connected layers, applying dropout for regularization and returning predictions along with updated hidden states.
- **Step 5: Model Training:** The training process follows a standard approach, with careful consideration given to the structure of input sequences. Depending on the specified decoding length, these sequences may consist of segments from different sequences in the original dataset or represent subsets of a single sequence. To address this, the hidden state of the model is reset at the start of each epoch. This strategy assumes that subsequent batches are likely continuations of the preceding sequences from the original dataset. The model is designed to predict the next token in a sequence by employing Cross-Entropy Loss. This training process generates output predictions, which are probability distributions over the vocabulary, and simultaneously updates its hidden states, after that we measuring the difference between the predicted probability distributions and the true target tokens, on the last but not least process, using backpropagation to calculate gradient, and the model’s parameters are optimized to reduce the loss.

**Model Configuration:**  
The following hyperparameters were tuned for the training process
- Vocabulary size (`vocab_size`): 5,752  
- Embedding dimension (`emb_dim`): 1,024  
- Hidden dimension (`hid_dim`): 1,024  
- Number of layers (`num_layers`): 2  
- Dropout rate (`dropout_rate`): 0.65  
- Learning rate (`lr`): 1 × 10⁻³  

My model architecture consists of 28,579,448 trainable parameters. Training is conducted over 50 epochs to ensure adequate convergence.
- **Step 6 Testing:** My test perplexity result is 78.050, which indicates that the model exhibits some level of ambiguity in its predictions. This suggests that the model is moderately uncertain when predicting the next token in the sequence.

## Getting Started: Launching the Application
- Step1 : Download 'best-val-lstm_lm.pt' from this link : https://drive.google.com/file/d/1sLretpcS-GiktkmeyPRI_f336Bu4tvME/view?usp=share_link
- Step2 : Move this file into model folder
- Step3 : You can run the application by using **python app.py**

## Demo application for Star Wars Text Generated web
<img width="1440" alt="Screenshot 2568-01-27 at 1 49 28 AM" src="https://github.com/user-attachments/assets/6dd8db54-b0a0-410b-af76-d3139c14823f" />
<img width="1440" alt="Screenshot 2568-01-27 at 1 48 42 AM" src="https://github.com/user-attachments/assets/8cb35b7c-7c89-4ffd-baec-57689eb5dffd" />
<img width="1440" alt="Screenshot 2568-01-27 at 1 47 14 AM" src="https://github.com/user-attachments/assets/7d45da05-2b55-468e-be9a-220370a9efd4" />
<img width="1440" alt="Screenshot 2568-01-27 at 1 47 04 AM" src="https://github.com/user-attachments/assets/5c4f8894-e6b4-4d96-9489-fed710c11d72" />


## Citation: https://huggingface.co/datasets/myamjechal/star-wars-dataset
### FYI: The .pt file cannot attache to Github because this file is too large for Github can handle.
