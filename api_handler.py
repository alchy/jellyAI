"""
API interaction handling functionality.
"""

import numpy as np
from typing import Dict
from api.jellyAPI import JellyAPI
from helpers import return_aligned_vocabulary_and_array


class APIHandler:
    def __init__(self):
        """Initialize API handler with JellyAPI instance."""
        self.api = JellyAPI()
        self.api.clear_data()

    def process_prediction(self, vocabulary_itw: Dict, 
                         sample: np.ndarray, 
                         nn_result: np.ndarray,
                         send_input: bool = True,
                         threshold: float = 0.01) -> None:
        """
        Process and send prediction data to API.
        
        Args:
            vocabulary_itw: Vocabulary index-to-word mapping
            sample: Input sample data
            nn_result: Neural network prediction result
            send_input: Whether to send input data to API
            threshold: Threshold for filtering values
        """
        if send_input:
            # Send input sample data
            api_post_data = return_aligned_vocabulary_and_array(
                vocabulary_itw, sample[0], threshold=threshold
            )
            print("[d] API post data (input):", api_post_data)
            print("[i] API response (input):", 
                  self.api.add_data(api_post_data))

        # Send prediction result
        api_post_data = return_aligned_vocabulary_and_array(
            vocabulary_itw, nn_result[0], threshold=threshold
        )
        print("[d] API post data (prediction):", api_post_data)
        print("[i] API response (prediction):", 
              self.api.add_data(api_post_data))