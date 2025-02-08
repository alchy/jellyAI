import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import os
import glob

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

        # Load the latest final model if it exists
        self.load_latest_model()

    def create_model(self):
        """
        Creates the neural network model.
        """
        model = Sequential()

        # Add input layer
        model.add(Dense(self.hidden_layer_counts[0], input_dim=self.input_layer_count, activation='relu'))

        # Add hidden layers
        for neurons in self.hidden_layer_counts[1:]:
            model.add(Dense(neurons, activation='relu'))

        # Add output layer with linear activation (no activation function)
        model.add(Dense(self.output_layer_count, activation=None))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

        return model

    def train(self, X, y, batch_size=32, epochs=10):
        """
        Trains the neural network using mini-batches and saves the final model with a sequence number.

        :param X: List containing the weights of the individual input neurons (in the first layer)
        :param y: List containing the outputs in the output layer
        :param batch_size: Size of the batches (default: 32)
        :param epochs: Number of training epochs (default: 10)
        """
        # Convert inputs and outputs to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Train the model
        self.model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.2)

        # Determine the latest model version number
        model_files = glob.glob(os.path.join(self.save_dir, 'model_*.keras'))
        if model_files:
            latest_version = max([int(f.split('_')[-1].split('.')[0]) for f in model_files])
        else:
            latest_version = 0

        # Save the final model with the next version number
        final_model_filepath = os.path.join(self.save_dir, f'model_{latest_version + 1}.keras')
        self.model.save(final_model_filepath)
        print(f"Final model saved: {final_model_filepath}")

    def load_latest_model(self):
        """
        Loads the latest model from saved files based on the sequence number.
        """
        model_files = glob.glob(os.path.join(self.save_dir, 'model_*.keras'))
        if model_files:
            latest_model_file = max(model_files, key=os.path.getctime)
            self.model.load_weights(latest_model_file)
            print(f"Latest model loaded: {latest_model_file}")
        else:
            print("No saved model found, training a new model.")

    def predict(self, X):
        """
        Performs prediction using the trained model.

        :param X: List containing the weights of the individual input neurons (in the first layer)
        :return: Prediction outputs
        """
        # Convert inputs to numpy array
        X = np.array(X)

        # Perform prediction and return results
        predictions = self.model.predict(X)
        return predictions

    def summary(self):
        """
        Prints a summary of the model architecture.
        """
        self.model.summary()


if __name__ == "__main__":
    # Example usage:
    input_layer_count = 10
    hidden_layer_counts = [20, 15, 10]  # Three hidden layers with 20, 15, and 10 neurons
    output_layer_count = 3
    save_dir = "./model_checkpoints"

    nn = NeuralNetwork(input_layer_count, hidden_layer_counts, output_layer_count, save_dir)

    # Generate random data for example
    X = np.random.rand(1000, input_layer_count)  # 1000 samples, each with 10 inputs
    y = np.random.rand(1000, output_layer_count) * 2 - 1  # Outputs in the range -1 to 1

    # Train the model
    nn.train(X, y)
    nn.summary()

    # Perform prediction
    test_X = np.random.rand(5, input_layer_count)  # 5 samples for testing
    predictions = nn.predict(test_X)

    # Set formatting for displaying predictions in standard decimal format
    np.set_printoptions(suppress=True, precision=6)
    print("Predictions:", predictions)