# Ran in 3 minutes on Google
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# 300 unsupervised reviews
_URL = 'https://codehs.com/uploads/60f4bd4ef2560a07c70af52b10c3ecfa'
_dir = tf.keras.utils.get_file('reviews.csv', origin=_URL, extract=True)

_dir_base = os.path.dirname(_dir)

# Read in with Pandas
data_path = os.path.join(_dir_base, 'reviews.csv')
dataset = pd.read_csv(data_path)
sentences_full = dataset['text'].tolist()


# Try increasing this from 5 if it processes quickly
num_records = 5
sentences = []

for i in range(num_records):
  sentences.append(sentences_full[i])

# Print first review as sample
print(sentences[0])

vocab_size = 1000

# Create the tokenizer
tokenizer = Tokenizer(vocab_size)

# Fit the training sentences to the tokenizer
tokenizer.fit_on_texts(sentences)

sequences = []
padding_type = "pre"

# Loop through all sentences
for line in sentences:
	# Tokenize teach sentence
	token_list = tokenizer.texts_to_sequences([line])[0]

	#Loop through the token list and create N-grams to add to the sequences list
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		sequences.append(n_gram_sequence)

print("Total input sequences: " + str(len(sequences)))
# Pad sequences to the longest sequence
max_length = max([len(seq) for seq in sequences])
padded = pad_sequences(sequences,maxlen=max_length, padding=padding_type)

# Split sequences between the "input" sequence and "output" predicted word
input_sequences, labels = padded[:,:-1], padded[:,-1]
review_labels = tf.keras.utils.to_categorical(labels, num_classes=vocab_size)

embedding_dim = 32

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length-1),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(input_sequences, review_labels, epochs=200, verbose=1)

seed_text = "A classic film"
next_words = 50

for i in range(next_words):
  # Tokenize and pad the seed text
  token_list = tokenizer.texts_to_sequences([seed_text])[0]
  token_list_padded = pad_sequences([token_list], maxlen=max_length-1, padding='pre')
  # Use the model to predict probabilities of token dictionary
  predicted_probs = model.predict(token_list_padded)[0]
  # Predict a word index based on probabilities
  predicted = np.random.choice([x for x in range(len(predicted_probs))],
                               p=predicted_probs)
  # Look up the index to find the word
  output_word = ""
  for word, index in tokenizer.word_index.items():
    if index == predicted:
      output_word = word
      break
  # Add to the seed text
  seed_text += " " + output_word

# Print results
print(seed_text)