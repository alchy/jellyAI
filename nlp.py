# nlp.py
import numpy as np
import re


DEBUG = True


class TextProcessor:
    def __init__(self, input_text, reserved_tokens_count=1):
        self.reserved_tokens_count = reserved_tokens_count
        
        self.input_text = (
            input_text.replace('.', '</p>')
            .replace(',', '</comma>')
            .replace(';', '</semicolon>')
            .replace(':', '</colon>')
            .replace('!', '</exclamation-mark>')
            .replace('(','</left-bracket>')
            .replace(')', '</right-bracket>')
            .replace('?', '</question-mark>')
            .replace('"', '</quote>')
            .replace('--', '</double-minuses>')
            .replace('\n', '</new-line>')
        )
        self.input_text = re.sub(r'\s+', '</space>', self.input_text)

        self.vocabulary_itw = [f"<reserved-{i}>" for i in range(self.reserved_tokens_count)]
        self.vocabulary_wti = {}  # Dictionary mapping word -> index        
        self.create_vocabularies() # Vytvoř slovníky

        self.sentence_index = []  # List of sentences (sentence index), where each sentence is a list of word indices - words are number tokens
        self.split_text_into_sentence_index()

    def create_vocabularies(self):
        # rozdel text, oddelovace <> zustanou soucasti slovniku
        words = re.split(r'(</?[^>]+>)', self.input_text)
        words = [word for word in words if word.strip()]

        # vytvor slovniky
        index = self.reserved_tokens_count
        for word in words:
            if word not in self.vocabulary_wti:
                self.vocabulary_wti[word] = index
                self.vocabulary_itw.append(word)
                index += 1

        if DEBUG:
            print("[d] create_vocabularies: ", self.vocabulary_wti)

    # vstup je slovo, vystup je index (pozice ve slovniku) 
    def wti(self, word):
        word_to_index = self.vocabulary_wti.get(word, -1)
        if word_to_index == -1:
            print("[!] word to index failure")
            exit()

        return word_to_index

    # vstup je index (pozice ve slovniku, vystup je slovo)
    def itw(self, index):
        if 0 <= index < len(self.vocabulary_itw):
            return self.vocabulary_itw[index]
        return ""

    def split_text_into_sentence_index(self):
        """
        split all input_text to separate sentences in index:

        sentences are then in class variable:
            self.sentences[ [s1], [s2], [s3]..]
        """
        if DEBUG:
            print("[d] entering function split_text_into_sentence_index")

        # Exclude all instances of the delimiter "</space>" from sentence
        exclude = "</space>"

        sentences = self.input_text.split("</p>") # split input text to sentence strings
        for sentence in sentences: # for each sentence
            if sentence.strip():
                # split sentence to array of words
                words = re.split(r'(</?[^>]+>)', sentence)
                print("[d] words1: ", words)
                # eliminate empty words
                words = [field for field in words if field]
                print("[d] words2: ", words)
                # eliminate delimiter
                words = [field for field in words if field != exclude]
                print("[d] words3: ", words)
                # create index of words
                sentence_indices = [self.wti(word) for word in words]

                self.sentence_index.append(sentence_indices)

    def create_text_attention(self, text_attention_span_length=3, text_attention_weight=0.1):
        """
        creates text attention arrays - every array consists of all words in sentence.

        Args:
            text_attention_span_length (int): The span length on each side of the focus word
            text_attention_weight (float): The weight decay factor for attention

        Returns:
            list: List of sentences, where each sentence contains a list of tuples.
                 Each tuple contains (text_attention_words, text_attention_weights) for each word position.
        """
        text_attention_sentences_local = []
        for sentence in self.sentence_index:
            # define text_attention_sentence_local placeholder
            text_attention_sentence_local = []

            # count lengths
            text_attention_sentence_length = len(sentence)

            # For each word position in the sentence
            for pos in range(text_attention_sentence_length):
                # initialize arrays with zeros
                # text attention array for 3 elements has 7 items
                # w1    w2   w3 <-- w4 --> w5    w6    w7 
                # words to left <att.word> words to right
                text_attention_words = [0] * (2 * text_attention_span_length + 1)
                text_attention_weights = [np.float64(0.0)] * (2 * text_attention_span_length + 1)

                # Fill the arrays
                for offset in range(-text_attention_span_length, text_attention_span_length + 1):
                    array_pos = offset + text_attention_span_length
                    sentence_pos = pos + offset

                    # Check if the position is within sentence bounds
                    if 0 <= sentence_pos < text_attention_sentence_length:
                        text_attention_words[array_pos] = sentence[sentence_pos]

                        # Calculate weight based on offset
                        if offset > 0:  # Words to the right
                            text_attention_weights[array_pos] = np.tanh(1.1 - (offset * text_attention_weight))
                        elif offset < 0:  # Words to the left
                            text_attention_weights[array_pos] = np.tanh(offset * text_attention_weight)
                        # When offset == 0, weight remains 0

                        if DEBUG:
                            print("[d] create_text_attention -> text attention words: ", text_attention_words, " text attention weights: ", text_attention_weights)
                        
                text_attention_sentence_local.append((text_attention_words, text_attention_weights))

            text_attention_sentences_local.append(text_attention_sentence_local)

        return text_attention_sentences_local

    def create_nlm_index(self, text_attention_sample):
        """
        Creates a one-dimensional array with nlm_index_count elements,
        initializes it with 0.0, and fills it with values from nlm_index_values
        at positions corresponding to nlm_index_positions.

        Args:
            text_attention (list): The continuous attention array
            text_attention_sample

        Returns:
            list: One-dimensional array filled with appropriate values
        """
        # Získání počtu prvků ve slovníku
        nlm_index_count = len(self.vocabulary_itw)

        # Získání pozic a hodnot z text_attention
        nlm_index_positions, nlm_index_values = text_attention_sample

        # Vytvoření jednorozměrného pole s délkou nlm_index_count a naplnění hodnotami 0.0
        nlm_index_array = [np.float64(0.0)] * nlm_index_count

        # Naplnění pole hodnotami na odpovídající pozice
        for position, value in zip(nlm_index_positions, nlm_index_values):
            if position < nlm_index_count:
                nlm_index_array[position] = value

        return nlm_index_array


# Příklad použití třídy TextProcessor
if __name__ == "__main__":
    input_text = "This is a sample. Sample input text. Text which contains several words. This text is used for testing."
    text_processor = TextProcessor(input_text)

    """
    # Vytvoření text_attention
    [
        [
            ( [0, 0, 0, 1, 2, 3, 2], [0.0, 0.0, 0.0, 0.0, 0.76, 0.71, 0.66] ),
            ( [0, 0, 1, 2, 3, 2, 4], [0.0, 0.0, -0.09, 0.0, 0.76, 0.71, 0.66] ), 
            ( [0, 1, 2, 3, 2, 4, 2], [0.0, -0.19, -0.09, 0.0, 0.76, 0.71, 0.66] ), 
            ( [1, 2, 3, 2, 4, 2, 5], [-0.29, -0.19, -0.09, 0.0, 0.76, 0.71, 0.66]), 
            ( [2, 3, 2, 4, 2, 5, 0], [-0.29, -0.19, -0.09, 0.0, 0.76, 0.71, 0.0 ] ), 
            ( [3, 2, 4, 2, 5, 0, 0], [-0.29, -0.19, -0.09, 0.0, 0.76, 0.0, 0.0] ), 
            ( [2, 4, 2, 5, 0, 0, 0], [-0.29, -0.19, -0.09, 0.0, 0.0, 0.0, 0.0] )
        ], 
        [
            ( [0, 0, 0, -1, 2, 7, 2], [0.0, 0.0, 0.0, 0.0, 0.76, 0.71, 0.66] ), 
            ( [0, 0, -1, 2, 7, 2, 8], [0.0, 0.0, -0.09, 0.0, 0.76, 0.71, 0.66] ), 
            ( [0, -1, 2, 7, 2, 8, 2], [0.0, -0.19, -0.09, 0.0, 0.76, 0.71, 0.66] ), 
            ( [-1, 2, 7, 2, 8, 2, 9], [-0.29, -0.19, -0.09, 0.0, 0.76, 0.71, 0.66] ), 
            ( [2, 7, 2, 8, 2, 9, 0], [-0.29, -0.19, -0.09, 0.0, 0.76, 0.71, 0.0] ), 
            ( [7, 2, 8, 2, 9, 0, 0], [-0.29, -0.19, -0.09, 0.0, 0.76, 0.0, 0.0] ), 
            ( [2, 8, 2, 9, 0, 0, 0], [-0.29, -0.19, -0.09, 0.0, 0.0, 0.0, 0.0] )
        ], 
        [
            ( [0, 0, 0, -1, 2, 10, 2], [0.0, 0.0, 0.0, 0.0, 0.76, 0.71, 0.66] ), 
            ( [0, 0, -1, 2, 10, 2, 11], [0.0, 0.0, -0.09, 0.0, 0.76, 0.71, 0.66] ), 
            ( [0, -1, 2, 10, 2, 11, 2], [0.0, -0.19, -0.09, 0.0, 0.76, 0.71, 0.66] ), 
            ( [-1, 2, 10, 2, 11, 2, 12], [-0.29, -0.19, -0.09, 0.0, 0.76, 0.71, 0.66] ), 
            ( [2, 10, 2, 11, 2, 12, 2], [-0.29, -0.19, -0.09, 0.0, 0.76, 0.71, 0.66] ), 
            ( [10, 2, 11, 2, 12, 2, 13], [-0.29, -0.19, -0.09, 0.0, 0.76, 0.71, 0.66] ), 
            ( [2, 11, 2, 12, 2, 13, 2], [-0.29, -0.19, -0.09, 0.0, 0.76, 0.71, 0.66] ), 
            ( [11, 2, 12, 2, 13, 2, 14], [-0.29, -0.19, -0.09, 0.0, 0.76, 0.71, 0.66] ), 
            ( [2, 12, 2, 13, 2, 14, 0], [-0.29, -0.19, -0.09, 0.0, 0.76, 0.71, 0.0] ), 
            ( [12, 2, 13, 2, 14, 0, 0], [-0.29, -0.19, -0.09, 0.0, 0.76, 0.0, 0.0] ), 
            ( [2, 13, 2, 14, 0, 0, 0], [-0.29, -0.19, -0.09, 0.0, 0.0, 0.0, 0.0]) 
        ], 
        [..]
    ]
    """
    text_attention_span_length = 3
    text_attention_weight = 0.1
    text_attention = text_processor.create_text_attention(
        text_attention_span_length=text_attention_span_length, text_attention_weight=text_attention_weight)
    print("\n[d] text attention index: ", text_attention)

    # Vytvoření a naplnění nlm_index_array
    text_attention_sentence = 0
    text_attention_sentence_position = 0  # position is between 0 <---> text_attention_span_lengt * 2 (+1) 
    nlm_index_array = text_processor.create_nlm_index(text_attention[text_attention_sentence][text_attention_span_length])
    print(nlm_index_array)
