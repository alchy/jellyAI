# main.py

from nlp import TextProcessor
from nlptensors import NLPTensorProcessor

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


###

# Vytvoření instance NLPTensorProcessor
tensor_processor = NLPTensorProcessor(processor)

# Trénování modelu
tensor_processor.train(num_epochs=100, sequence_length=3)

# Příklad predikce
test_sequence = processor.si[0][:2]  # Prvních x slov z X věty
predicted_word = tensor_processor.predict_next_word(test_sequence)
print(f"\nTest sequence: {[processor.itw(idx) for idx in test_sequence]}")
print(f"Predicted next word: {predicted_word}")

# Příklad predikce
test_sequence = [processor.wti('ema'), processor.wti('je')]  # Prvních x slov z X věty
predicted_word = tensor_processor.predict_next_word(test_sequence)
print(f"\nTest sequence: {[processor.itw(idx) for idx in test_sequence]}")
print(f"Predicted next word: {predicted_word}")

# Příklad predikce
test_sequence = [processor.wti('ema'), processor.wti('rada')]  # Prvních x slov z X věty
predicted_word = tensor_processor.predict_next_word(test_sequence)
print(f"\nTest sequence: {[processor.itw(idx) for idx in test_sequence]}")
print(f"Predicted next word: {predicted_word}")

# Příklad predikce
test_sequence = [processor.wti('maso'), processor.wti('ma')]  # Prvních x slov z X věty
predicted_word = tensor_processor.predict_next_word(test_sequence)
print(f"\nTest sequence: {[processor.itw(idx) for idx in test_sequence]}")
print(f"Predicted next word: {predicted_word}")
