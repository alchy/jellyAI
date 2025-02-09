import numpy as np
import re
from collections import Counter

DEBUG = True


class TextProcessor:
    """
    Třída pro zpracování textu s podporou attention mechanismu a vážení slov 
    podle jejich distribuce v textu.
    """

    def __init__(self, input_text, reserved_tokens_count=16, gaussian_range=1.0):
        """
        Inicializace TextProcessoru.

        Args:
            input_text (str): Vstupní text k zpracování
            reserved_tokens_count (int): Počet rezervovaných tokenů na začátku slovníku
            gaussian_range (float): Rozsah standardních odchylek pro úpravu vah slov
        """
        self.reserved_tokens_count = reserved_tokens_count
        self.gaussian_range = gaussian_range

        # Nahrazení speciálních znaků za tokenizované verze
        self.input_text = (
            input_text.replace('.', '</p>')
            .replace(',', '</comma>')
            .replace(';', '</semicolon>')
            .replace(':', '</colon>')
            .replace('!', '</exclamation-mark>')
            .replace('(', '</left-bracket>')
            .replace(')', '</right-bracket>')
            .replace('?', '</question-mark>')
            .replace('\"', '</quote>')
            .replace('--', '</double-minuses>')
            .replace('\n', '</new-line>')
        )
        # Nahrazení bílých znaků za token mezery
        self.input_text = re.sub(r'\s+', '</space>', self.input_text)

        # Inicializace slovníků pro převod mezi slovy a indexy
        self.vocabulary_itw = [f"<reserved-{i}>" for i in range(self.reserved_tokens_count)]  # index->word
        self.vocabulary_wti = {}  # word->index

        # Vytvoření slovníků
        self.create_vocabularies()

        # List vět, kde každá věta je seznam indexů slov
        self.sentence_index = []
        self.split_text_into_sentence_index()

        # Výpočet vah slov na základě jejich distribuce v textu
        self.word_weights = self.calculate_word_weights()

        if DEBUG:
            print("[d] Word weights distribution:", self.word_weights)

    def create_vocabularies(self):
        """
        Vytvoří slovníky pro převod mezi slovy a jejich indexy.
        Rozdělí text na slova a přiřadí jim unikátní indexy.
        """
        # Rozdělení textu na slova, zachování oddělovačů v <>
        words = re.split(r'(</?[^>]+>)', self.input_text)
        words = [word for word in words if word.strip()]

        # Vytvoření slovníků word->index a index->word
        index = self.reserved_tokens_count
        for word in words:
            if word not in self.vocabulary_wti:
                self.vocabulary_wti[word] = index
                self.vocabulary_itw.append(word)
                index += 1

        if DEBUG:
            print("[d] create_vocabularies: ", self.vocabulary_wti)

    def wti(self, word):
        """
        Převede slovo na jeho index ve slovníku.

        Args:
            word (str): Slovo k převodu

        Returns:
            int: Index slova ve slovníku
        """
        word_to_index = self.vocabulary_wti.get(word, -1)
        if word_to_index == -1:
            print("[!] word to index failure")
            exit()
        return word_to_index

    def itw(self, index):
        """
        Převede index na odpovídající slovo.

        Args:
            index (int): Index slova

        Returns:
            str: Slovo odpovídající indexu
        """
        if 0 <= index < len(self.vocabulary_itw):
            return self.vocabulary_itw[index]
        return ""

    def split_text_into_sentence_index(self):
        """
        Rozdělí vstupní text na věty a převede je na posloupnosti indexů slov.
        Výsledek uloží do self.sentence_index.
        """
        if DEBUG:
            print("[d] entering function split_text_into_sentence_index")

        # Vyloučení mezery z tokenů
        exclude = "</space>"

        # Rozdělení textu na věty podle značky </p>
        sentences = self.input_text.split("</p>")
        for sentence in sentences:
            if sentence.strip():
                # Rozdělení věty na slova
                words = re.split(r'(</?[^>]+>)', sentence)
                if DEBUG:
                    print("[d] words1: ", words)
                # Odstranění prázdných slov
                words = [field for field in words if field]
                if DEBUG:
                    print("[d] words2: ", words)
                # Odstranění mezer
                words = [field for field in words if field != exclude]
                if DEBUG:
                    print("[d] words3: ", words)
                # Převod slov na indexy
                sentence_indices = [self.wti(word) for word in words]
                self.sentence_index.append(sentence_indices)

    def calculate_word_weights(self):
        """
        Vypočítá váhy slov na základě jejich četnosti v textu.
        Používá Gaussovské rozložení pro identifikaci běžných slov.

        Returns:
            dict: Slovník {index_slova: váha}
        """
        # Spočítání četnosti slov
        word_frequencies = Counter()
        for sentence in self.sentence_index:
            word_frequencies.update(sentence)

        # Výpočet statistických hodnot
        frequencies = list(word_frequencies.values())
        if not frequencies:
            return {i: 1.0 for i in range(len(self.vocabulary_itw))}

        mean_freq = np.mean(frequencies)
        std_freq = np.std(frequencies)

        if std_freq == 0:
            return {i: 1.0 for i in range(len(self.vocabulary_itw))}

        # Výpočet vah pro každé slovo
        word_weights = {}
        for word_idx in range(len(self.vocabulary_itw)):
            if word_idx < self.reserved_tokens_count:
                # Rezervované tokeny mají plnou váhu
                word_weights[word_idx] = 1.0
                continue

            freq = word_frequencies.get(word_idx, 0)
            z_score = (freq - mean_freq) / std_freq

            # Úprava váhy podle z-score
            if abs(z_score) <= self.gaussian_range:
                # Slova v běžném rozsahu četnosti mají sníženou váhu
                weight = 1.0 - (1.0 - abs(z_score) / self.gaussian_range) * 0.5
            else:
                # Slova mimo běžný rozsah mají plnou váhu
                weight = 1.0

            word_weights[word_idx] = weight

        return word_weights

    def create_text_attention(self, text_attention_span_length=3, text_attention_weight=0.1):
        """
        Vytváří attention vektory pro každé slovo v textu.

        Args:
            text_attention_span_length (int): Počet slov na každé straně od fokusovaného slova
            text_attention_weight (float): Základní váha pro attention mechanismus

        Returns:
            list: Seznam vět, kde každá věta obsahuje seznam dvojic (slova, váhy)
        """
        text_attention_sentences_local = []
        for sentence in self.sentence_index:
            text_attention_sentence_local = []
            text_attention_sentence_length = len(sentence)

            for pos in range(text_attention_sentence_length):
                # Inicializace polí pro slova a jejich váhy
                text_attention_words = [0] * (2 * text_attention_span_length + 1)
                text_attention_weights = [np.float64(0.0)] * (2 * text_attention_span_length + 1)

                # Naplnění polí pro aktuální pozici
                for offset in range(-text_attention_span_length, text_attention_span_length + 1):
                    array_pos = offset + text_attention_span_length
                    sentence_pos = pos + offset

                    if 0 <= sentence_pos < text_attention_sentence_length:
                        word_idx = sentence[sentence_pos]
                        text_attention_words[array_pos] = word_idx

                        # Výpočet základní váhy podle pozice
                        if offset > 0:  # Slova vpravo
                            base_weight = np.tanh(1.1 - (offset * text_attention_weight))
                        elif offset < 0:  # Slova vlevo
                            base_weight = np.tanh(offset * text_attention_weight)
                        else:  # Aktuální slovo
                            base_weight = 0.0

                        # Aplikace váhy slova z distribuce
                        text_attention_weights[array_pos] = base_weight * self.word_weights[word_idx]

                text_attention_sentence_local.append((text_attention_words, text_attention_weights))

            text_attention_sentences_local.append(text_attention_sentence_local)

        return text_attention_sentences_local

    def create_nlm_index(self, text_attention_sample):
        """
        Vytvoří jednorozměrné pole pro NLM (Natural Language Model) z attention vzorku.

        Args:
            text_attention_sample (tuple): Dvojice (pozice, hodnoty) z text attention

        Returns:
            list: Jednorozměrné pole s váhami na odpovídajících pozicích
        """
        nlm_index_count = len(self.vocabulary_itw)
        nlm_index_positions, nlm_index_values = text_attention_sample

        # Inicializace pole nulami
        nlm_index_array = [np.float64(0.0)] * nlm_index_count

        # Naplnění pole hodnotami na správných pozicích
        for position, value in zip(nlm_index_positions, nlm_index_values):
            if position < nlm_index_count:
                nlm_index_array[position] = value

        return nlm_index_array


# Příklad použití
if __name__ == "__main__":
    # Testovací text
    input_text = "This is a sample. Sample input text. Text which contains several words. This text is used for testing."

    # Vytvoření instance zpracovatele textu
    text_processor = TextProcessor(input_text)

    # Vytvoření attention vektorů
    text_attention = text_processor.create_text_attention(
        text_attention_span_length=3,
        text_attention_weight=0.1
    )
    print("\n[d] text attention index: ", text_attention)

    # Test vytvoření NLM indexu
    text_attention_sentence = 0
    text_attention_sentence_position = 0
    nlm_index_array = text_processor.create_nlm_index(
        text_attention[text_attention_sentence][text_attention_sentence_position]
    )
    print(nlm_index_array)