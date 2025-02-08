# imports
import numpy as np
from nlp import TextProcessor
from helpers import (print_aligned_vocabulary_and_array,
                     print_aligned_vocabulary_and_array_combo,
                     return_aligned_vocabulary_and_array)
from nn.neuralbase import NeuralNetwork
from api.jellyAPI import JellyAPI


# program behavior
API_RUN = True
TRAINING = True


# Provádění predikce na základě vzorku
def nn_prediction(word):
    sample = text_processor.create_nlm_index(
        text_attention_sample=([[text_processor.wti(word)], [np.float64(0.9)]]))
    sample=np.array(sample).reshape(1, -1)
    return sample, nn.predict(sample)


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

# create instance of text_processor with input text
text_processor = TextProcessor(input_text)

# create continuous attention arrays from input text with parameters below
text_attention_span_length = 3
text_attention_weight = 0.1
text_attention = text_processor.create_text_attention(
    text_attention_span_length=text_attention_span_length,
    text_attention_weight=text_attention_weight
)

# setup feed forward nn
vocabulary_size = len(text_processor.vocabulary_itw)  # vcetne rezervovanych tokenu
input_layer_count = vocabulary_size  # velikost vstupni vstvy odpovida poctu slov
hidden_layer_counts = [round(vocabulary_size / 7), round(vocabulary_size)] 
output_layer_count = vocabulary_size # velikos vystupni vrstvy odpovida poctu slov
save_dir = "./model_checkpoints"

# inicializace neuronky
nn = NeuralNetwork(input_layer_count, hidden_layer_counts, output_layer_count, save_dir)

# Seznamy pro uložení vstupních a výstupních dat
X = []
y = []

"""
Iterace přes všechny záznamy v text_attention[0][0][0]:
  
    jeden ze zaznamu v text_attention pro vzorovou:
        [
            ( [0, 0, 0, 1, 2, 3, 2], [0.0, 0.0, 0.0, 0.0, 0.76, 0.71, 0.66] ),
            ( [0, 0, 1, 2, 3, 2, 4], [0.0, 0.0, -0.09, 0.0, 0.76, 0.71, 0.66] ), 
            ( [0, 1, 2, 3, 2, 4, 2], [0.0, -0.19, -0.09, 0.0, 0.76, 0.71, 0.66] ), 
            ( [1, 2, 3, 2, 4, 2, 5], [-0.29, -0.19, -0.09, 0.0, 0.76, 0.71, 0.66]), 
            ( [2, 3, 2, 4, 2, 5, 0], [-0.29, -0.19, -0.09, 0.0, 0.76, 0.71, 0.0 ] ), 
            ( [3, 2, 4, 2, 5, 0, 0], [-0.29, -0.19, -0.09, 0.0, 0.76, 0.0, 0.0] ), 
            ( [2, 4, 2, 5, 0, 0, 0], [-0.29, -0.19, -0.09, 0.0, 0.0, 0.0, 0.0] )
        ]
"""
# iterace pres vsechny zaznamy v text attention
for text_attention_sample in text_attention:
    for text_attention_sample_variant in text_attention_sample:
        # vytvoření nlm_index_array pro vstupní data - v tomto pripade slovo kolem ktereho je budovana text attention
        # do create_nlm_index se posle vzorek a vrati se cele jednorozmerne pole hodnot pro cely slovnik
        # zde se predave (cislo/index slova, hodnota/vaha), zbytek se dopocivava, slovo o ktere jde je vzdy 
        # na pozici text_attention_span_lenght
        nlm_input_index_array = text_processor.create_nlm_index(
            text_attention_sample=([text_attention_sample_variant[0][text_attention_span_length]], [np.float64(0.9)]))
        # vytvoření nlm_index_array pro vystupni data - v tomto pripade jak ma pro dane slovo vypadat vystup
        # zde se preda cely radek ze strutury odpovidajici 
        # ([cislo/index slova, cislo/index slova, ...],[hodnota/vaha, hodnota/vaha, ..])
        nlm_output_index_array = text_processor.create_nlm_index(
            text_attention_sample=text_attention_sample_variant)

        # Přidání do seznamů
        X.append(nlm_input_index_array)
        y.append(nlm_output_index_array)

# Tisk zarovnaného slovníku a nlm_index_array pro první záznam
print_aligned_vocabulary_and_array_combo(text_processor.vocabulary_itw, X[0], y[0])

# Trénování modelu
if TRAINING: 
    nn.train(X, y, batch_size=3, epochs=1024)
    nn.summary()

# main loop
if API_RUN:
    api = JellyAPI()
    api.clear_data()

running = True

while running:
    user_input = input("Enter a word (enter '.' to stop): ")
    if user_input == '.':
            print("Stopping the loop.")
            running = False
    else:
        # Prediction
        sample, nn_result = nn_prediction(user_input)

        # Tisk zarovnaného slovníku pro zadani a vystup nn
        print_aligned_vocabulary_and_array_combo(
            text_processor.vocabulary_itw, sample[0], nn_result.flatten())

        
        # Přidání nových dat do API
        # user_input
        api_post_data = return_aligned_vocabulary_and_array(text_processor.vocabulary_itw, sample[0], threshold=0.01)
        print("[d] API post data: ", api_post_data)
        if API_RUN:
            print("[i] API response sample[0]: ", api.add_data(return_aligned_vocabulary_and_array(text_processor.vocabulary_itw, sample[0], threshold=0.01)))
        # prediction
        api_post_data = return_aligned_vocabulary_and_array(text_processor.vocabulary_itw, nn_result[0], threshold=0.01)
        print("[d] API post data: ", api_post_data)
        if API_RUN:
            print("[i] API response nn_result[0]: ", api.add_data(return_aligned_vocabulary_and_array(text_processor.vocabulary_itw, nn_result[0], threshold=0.01)))
