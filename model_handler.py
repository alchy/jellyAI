"""
Neural Network model handling and prediction functionality.
"""

import numpy as np
from typing import Tuple, Any, List

from nn.neuralbase import NeuralNetwork
from nlp import TextProcessor


class ModelHandler:
    def __init__(self, text_processor: TextProcessor, save_dir: str):
        """
        Initialize ModelHandler with text processor and model configuration.
        
        Args:
            text_processor: Initialized TextProcessor instance
            save_dir: Directory path for saving model checkpoints
        """
        self.text_processor = text_processor
        self.vocabulary_size = len(text_processor.vocabulary_itw)
        self.save_dir = save_dir
        
        # Neural network layer configuration
        self.input_layer_count = self.vocabulary_size
        self.hidden_layer_counts = [
            round(8),
            round(self.vocabulary_size),
            round(self.vocabulary_size),
            round(self.vocabulary_size),
            round(self.vocabulary_size)
        ]
        self.output_layer_count = self.vocabulary_size
        
        # Initialize neural network
        self.nn = NeuralNetwork(
            self.input_layer_count,
            self.hidden_layer_counts,
            self.output_layer_count,
            save_dir
        )

    def prepare_training_data(self, text_attention: List[List], span_length: int) -> Tuple[List, List]:
        """
        Prepare training data from text attention arrays.
        
        Args:
            text_attention: List of text attention data
            span_length: Length of text attention span
            
        Returns:
            Tuple containing input (X) and output (y) training data
        """
        X, y = [], []
        
        for text_attention_sample in text_attention:
            for text_attention_sample_variant in text_attention_sample:
                try:
                    # Create input array (focused on central word)
                    nlm_input_index_array = self.text_processor.create_nlm_index(
                        text_attention_sample=([
                            text_attention_sample_variant[0][span_length]
                        ], [np.float64(0.9)])
                    )
                    
                    # Create output array (full context)
                    nlm_output_index_array = self.text_processor.create_nlm_index(
                        text_attention_sample=text_attention_sample_variant
                    )
                    
                    X.append(nlm_input_index_array)
                    y.append(nlm_output_index_array)
                except IndexError as e:
                    print(f"Warning: Skipping sample due to index error: {e}")
                    continue
                
        return X, y

    def train(self, X: List, y: List, batch_size: int, epochs: int) -> None:
        """
        Train the neural network model.
        
        Args:
            X: Input training data
            y: Output training data
            batch_size: Size of training batches
            epochs: Number of training epochs
        """
        self.nn.train(X, y, batch_size=batch_size, epochs=epochs)
        self.nn.summary()

    def predict(self, word: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make prediction for a given word.
        
        Args:
            word: Input word for prediction
            
        Returns:
            Tuple containing input sample and prediction result
        """
        sample = self.text_processor.create_nlm_index(
            text_attention_sample=([[self.text_processor.wti(word)], [np.float64(0.9)]])
        )
        sample = np.array(sample).reshape(1, -1)
        return sample, self.nn.predict(sample)