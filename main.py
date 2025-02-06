# main.py

import numpy as np
from nlp import TextProcessor
from helpers import (print_aligned_vocabulary_and_array,
                     print_aligned_vocabulary_and_array_combo,
                     return_aligned_vocabulary_and_array)
from nn.neuralbase import NeuralNetwork


# Input text with explicit spaces between sentences
input_text = (
    "ema ma maso. "
    "mama ma misu. "
    "ema je holka. "
    "mama je starsi holka. "
    "ema je dcera mamy. "
    "mama je matka emy. "
    "mama ma misu. "
    "misa je modra. "
    "ema nema rada maso. "
    "ema ma mamu. "
)

# Create TextProcessor instance
processor = TextProcessor(input_text)

# Create continuous attention arrays
ca_attention_span_length = 3
ca_attention_weight = 0.1
continuous_attention = processor.create_continuous_attention(
    ca_attention_span_length=ca_attention_span_length,
    ca_attention_weight=ca_attention_weight
)

# Print results
for sentence_idx, sentence_ca in enumerate(continuous_attention):
    print(f"\nSentence {sentence_idx + 1}:")
    original_sentence = processor.si_to_sentences()[sentence_idx]
    print(f"Original: {original_sentence}")

    for word_idx, (ca_words, ca_weights) in enumerate(sentence_ca):
        print(f"\nWord position {word_idx}:")
        print(f"CA Words:   {ca_words}")
        print(f"CA Weights: [{', '.join([f'{w:.1f}' for w in ca_weights])}]")

# nastaveí neuronky
vocabulary_size = len(processor.vocabulary_itw)  # vcetne rezervovanych tokenu
input_layer_count = vocabulary_size
hidden_layer_counts = [round(vocabulary_size / 7), round(vocabulary_size)]
output_layer_count = vocabulary_size
save_dir = "./model_checkpoints"

# inicializace neuronky
nn = NeuralNetwork(input_layer_count, hidden_layer_counts, output_layer_count, save_dir)

# Seznamy pro uložení vstupních a výstupních dat
X = []
y = []

# Iterace přes všechny záznamy v continuous_attention
for ca_sample in continuous_attention:
    # Vytvoření nlm_index_array pro vstupní a výstupní data
    nlm_input_index_array = processor.create_nlm_index(
        continuous_attention_sample=([ca_sample[0][0][ca_attention_span_length]], [0.9]))
    nlm_output_index_array = processor.create_nlm_index(
        continuous_attention_sample=ca_sample[0])

    # Přidání do seznamů
    X.append(nlm_input_index_array)
    y.append(nlm_output_index_array)

# Konverze seznamů na numpy array
X = np.array(X)
y = np.array(y)

# Tisk zarovnaného slovníku a nlm_index_array pro první záznam
print_aligned_vocabulary_and_array_combo(processor.vocabulary_itw, X[0], y[0])

print("X: ", X)
print("y: ", y)

# Trénování modelu
#nn.train(X, y, batch_size=1, epochs=200)
#nn.summary()

# Provádění predikce na základě vzorku
sample_to_test = 1
sample = X[sample_to_test].reshape(1, -1)
sample_prediction = nn.predict(sample)

# Tisk zarovnaného slovníku pro zadani a vystup nn
print_aligned_vocabulary_and_array_combo(
    processor.vocabulary_itw, sample.flatten(), sample_prediction.flatten())

print("\n----------------------------")
print(return_aligned_vocabulary_and_array(processor.vocabulary_itw, sample.flatten()))
print(return_aligned_vocabulary_and_array(processor.vocabulary_itw, sample_prediction.flatten()))
