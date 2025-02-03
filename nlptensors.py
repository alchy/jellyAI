# nlptensors.py

import torch
import torch.nn as nn
import torch.optim as optim
from nlp import TextProcessor


class SimpleWordPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        """
        Jednoduchá neuronová síť pro predikci následujícího slova.

        Args:
            vocab_size (int): Velikost slovníku (počet všech možných slov)
            embedding_dim (int): Dimenze embedding vrstvy
            hidden_dim (int): Počet neuronů ve skryté vrstvě
        """
        super(SimpleWordPredictor, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """
        Dopředný průchod sítí.

        Args:
            x: Tensor obsahující sekvenci indexů slov

        Returns:
            Tensor s predikcemi pro každé možné následující slovo
        """
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        out = self.fc(lstm_out[:, -1, :])
        return out


class NLPTensorProcessor:
    def __init__(self, text_processor, embedding_dim=128, hidden_dim=256):
        """
        Třída pro zpracování textu pomocí neuronové sítě.

        Args:
            text_processor: Instance třídy TextProcessor
            embedding_dim (int): Dimenze embedding vrstvy
            hidden_dim (int): Počet neuronů ve skryté vrstvě
        """
        self.text_processor = text_processor
        self.vocab_size = len(text_processor.vocabulary_itw)
        self.model = SimpleWordPredictor(self.vocab_size, embedding_dim, hidden_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())

    def prepare_sequences(self, sequence_length=2):
        """
        Připraví trénovací sekvence ze zpracovaného textu.

        Args:
            sequence_length (int): Délka vstupní sekvence pro predikci

        Returns:
            list: Seznam dvojic (vstupní sekvence, cílové slovo)
        """
        sequences = []
        targets = []

        for sentence in self.text_processor.si:
            if len(sentence) > sequence_length:
                for i in range(len(sentence) - sequence_length):
                    seq = sentence[i:i + sequence_length]
                    target = sentence[i + sequence_length]
                    sequences.append(seq)
                    targets.append(target)

        for sequence, target in zip(sequences, targets):
            print("sequence, target:", sequence,target)

        return sequences, targets

    def train(self, num_epochs=100, sequence_length=3):
        """
        Trénuje model na připravených sekvencích.

        Args:
            num_epochs (int): Počet trénovacích epoch
            sequence_length (int): Délka vstupní sekvence pro predikci
        """
        sequences, targets = self.prepare_sequences(sequence_length)

        for epoch in range(num_epochs):
            total_loss = 0

            for seq, target in zip(sequences, targets):
                # Příprava dat
                x = torch.tensor([seq], dtype=torch.long)
                y = torch.tensor([target], dtype=torch.long)

                # Nulování gradientů
                self.optimizer.zero_grad()

                # Dopředný průchod
                output = self.model(x)

                # Výpočet chyby
                loss = self.criterion(output, y)

                # Zpětný průchod
                loss.backward()

                # Aktualizace vah
                self.optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(sequences):.4f}')

    def predict_next_word(self, sequence):
        """
        Predikuje následující slovo pro zadanou sekvenci.

        Args:
            sequence (list): Seznam indexů slov

        Returns:
            str: Predikované následující slovo
        """
        with torch.no_grad():
            x = torch.tensor([sequence], dtype=torch.long)
            output = self.model(x)
            predicted_idx = output.argmax().item()
            return self.text_processor.itw(predicted_idx)
