import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import os
import glob
import re


class NeuralNetwork:
    def __init__(self, input_layer_count, hidden_layer_counts, output_layer_count, save_dir):
        self.input_layer_count = input_layer_count
        self.hidden_layer_counts = hidden_layer_counts
        self.output_layer_count = output_layer_count
        self.save_dir = save_dir
        self.model = self.create_model()

        # Create the save directory if it does not exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Load the best model if it exists
        self.load_best_model()

    def create_model(self):
        model = Sequential()

        # Přidání vstupní vrstvy
        model.add(Dense(self.hidden_layer_counts[0], input_dim=self.input_layer_count, activation='relu'))

        # Přidání skrytých vrstev
        for neurons in self.hidden_layer_counts[1:]:
            model.add(Dense(neurons, activation='relu'))

        # Přidání výstupní vrstvy s lineární aktivací (bez aktivační funkce)
        model.add(Dense(self.output_layer_count, activation=None))

        # Kompilace modelu
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

        return model

    def train(self, X, y, batch_size=32, epochs=10):
        """
        Trénuje neuronovou síť pomocí mini-batchů a ukládá nejlepší model.

        :param X: List obsahující váhy jednotlivých vstupních neuronů (v první vrstvě)
        :param y: List obsahující výstupy ve výstupní vrstvě
        :param batch_size: Velikost batchů (default: 32)
        :param epochs: Počet epoch trénování (default: 10)
        """
        # Konvertování vstupů a výstupů na numpy array
        X = np.array(X)
        y = np.array(y)

        # Definování callbacku pro ukládání nejlepšího modelu
        checkpoint_filepath = os.path.join(self.save_dir, 'best_model_{epoch:02d}_{val_mae:.2f}.keras')
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='val_mae',
            mode='min',
            save_best_only=True,
            verbose=1  # Přidání výpisů během ukládání modelu
        )

        # Trénování modelu s validací
        self.model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.2,
                       callbacks=[checkpoint_callback])

    def load_best_model(self, specific_file=None):
        """
        Načte nejlepší model z uložených souborů, pokud nebyl specifikován konkrétní soubor.

        :param specific_file: Cesta k specifickému souboru modelu (default: None)
        """
        if specific_file:
            self.model.load_weights(specific_file)
        else:
            # Najít nejlepší model podle skóre v názvu souboru
            model_files = glob.glob(os.path.join(self.save_dir, 'best_model_*.keras'))
            if model_files:
                best_model_file = max(model_files, key=lambda x: float(re.search(r'_(\d+\.\d+)\.keras$', x).group(1)))
                self.model.load_weights(best_model_file)
                print(f"Best model loaded: {best_model_file}")
            else:
                print("No saved model found, training a new model.")

    def predict(self, X):
        """
        Provádí predikci pomocí natrénovaného modelu.

        :param X: List obsahující váhy jednotlivých vstupních neuronů (v první vrstvě)
        :return: Výstupy predikce
        """
        # Konvertování vstupů na numpy array
        X = np.array(X)

        # Provádění predikce a vrácení výsledků
        predictions = self.model.predict(X)
        return predictions

    def summary(self):
        self.model.summary()


if __name__ == "__main__":
    # Příklad použití:
    input_layer_count = 10
    hidden_layer_counts = [20, 15, 10]  # Tři skryté vrstvy s 20, 15 a 10 neurony
    output_layer_count = 3
    save_dir = "./model_checkpoints"

    nn = NeuralNetwork(input_layer_count, hidden_layer_counts, output_layer_count, save_dir)

    # Generování náhodných dat pro příklad
    X = np.random.rand(1000, input_layer_count)  # 1000 vzorků, každý s 10 vstupy
    y = np.random.rand(1000, output_layer_count) * 2 - 1  # Výstupy v rozmezí -1 až 1

    # Trénování modelu
    nn.train(X, y)
    nn.summary()

    # Provádění predikce
    test_X = np.random.rand(5, input_layer_count)  # 5 vzorků pro testování
    predictions = nn.predict(test_X)

    # Nastavení formátování pro zobrazení predikcí v běžném desítkovém formátu
    np.set_printoptions(suppress=True, precision=6)
    print("Predictions:", predictions)
