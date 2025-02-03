# nlp.py

class TextProcessor:
    def __init__(self, input_text, reserved_tokens_count=4):
        self.input_text = input_text
        self.reserved_tokens_count = reserved_tokens_count
        self.vocabulary_wti = {}  # Dictionary mapping word -> index
        self.vocabulary_itw = [f"<reserved-{i}>" for i in range(self.reserved_tokens_count)]
        self.si = []  # List of sentences (sentence index), where each sentence is a list of word indices
        self.create_vocabularies()
        self.split_into_sentences()

    def create_vocabularies(self):
        words = self.input_text.replace(".", "").replace(",", "").split()
        index = self.reserved_tokens_count
        for word in words:
            if word not in self.vocabulary_wti:
                self.vocabulary_wti[word] = index
                self.vocabulary_itw.append(word)
                index += 1

    def wti(self, word):
        return self.vocabulary_wti.get(word, -1)

    def itw(self, index):
        if 0 <= index < len(self.vocabulary_itw):
            return self.vocabulary_itw[index]
        return ""

    def split_into_sentences(self):
        sentences = self.input_text.split(".")
        for sentence in sentences:
            if sentence.strip():
                words = sentence.replace(",", "").split()
                sentence_indices = [self.wti(word) for word in words]
                self.si.append(sentence_indices)

    def si_to_sentences(self):
        # si - sentence index
        sentences = []
        for sentence_indices in self.si:
            sentence_words = [self.itw(index) for index in sentence_indices]
            sentence = " ".join(sentence_words)
            sentences.append(sentence)
        return sentences

    def create_continuous_attention(self, ca_attention_span_length=3, ca_attention_weight=0.1):
        """
        Creates continuous attention arrays for each word in each sentence.

        Args:
            ca_attention_span_length (int): The span length on each side of the focus word
            ca_attention_weight (float): The weight decay factor for attention

        Returns:
            list: List of sentences, where each sentence contains a list of tuples.
                 Each tuple contains (ca_words, ca_weights) for each word position.
        """
        continuous_attention = []

        for sentence in self.si:
            sentence_ca = []
            sentence_length = len(sentence)

            # For each word position in the sentence
            for pos in range(sentence_length):
                # Initialize arrays with zeros
                ca_words = [0] * (2 * ca_attention_span_length + 1)
                ca_weights = [0.0] * (2 * ca_attention_span_length + 1)

                # Fill the arrays
                for offset in range(-ca_attention_span_length, ca_attention_span_length + 1):
                    array_pos = offset + ca_attention_span_length
                    sentence_pos = pos + offset

                    # Check if the position is within sentence bounds
                    if 0 <= sentence_pos < sentence_length:
                        ca_words[array_pos] = sentence[sentence_pos]

                        # Calculate weight based on offset
                        if offset > 0:  # Words to the right
                            ca_weights[array_pos] = 1.0 - (offset * ca_attention_weight)
                        elif offset < 0:  # Words to the left
                            ca_weights[array_pos] = ((-1) ** abs(offset)) * (abs(offset) * ca_attention_weight)
                        # When offset == 0, weight remains 0

                sentence_ca.append((ca_words, ca_weights))

            continuous_attention.append(sentence_ca)

        return continuous_attention

    def create_nlm_index(self, continuous_attention_sample):
        """
        Creates a one-dimensional array with nlm_index_count elements,
        initializes it with 0.0, and fills it with values from nlm_index_values
        at positions corresponding to nlm_index_positions.

        Args:
            continuous_attention (list): The continuous attention array
            continuous_attention_sample

        Returns:
            list: One-dimensional array filled with appropriate values
        """
        # Získání počtu prvků ve slovníku
        nlm_index_count = len(self.vocabulary_itw)

        # Získání pozic a hodnot z continuous_attention
        nlm_index_positions, nlm_index_values = continuous_attention_sample

        # Vytvoření jednorozměrného pole s délkou nlm_index_count a naplnění hodnotami 0.0
        nlm_index_array = [0.0] * nlm_index_count

        # Naplnění pole hodnotami na odpovídající pozice
        for position, value in zip(nlm_index_positions, nlm_index_values):
            if position < nlm_index_count:
                nlm_index_array[position] = value

        return nlm_index_array


# Příklad použití třídy TextProcessor
if __name__ == "__main__":
    input_text = "This is a sample input text, which contains several words. This text is used for testing."
    processor = TextProcessor(input_text)

    # Vytvoření continuous_attention
    continuous_attention = processor.create_continuous_attention()

    # Vytvoření a naplnění nlm_index_array
    nlm_index_array = processor.create_nlm_index(continuous_attention)
    print(nlm_index_array)
