"""
Main application script for text processing and prediction system.
"""

import numpy as np
from typing import Optional

from nlp import TextProcessor
from helpers import print_aligned_vocabulary_and_array_combo, return_aligned_vocabulary_and_array
from model_handler import ModelHandler
from api_handler import APIHandler
from input_text_handler import read_all_text_files
import config


def main():
    # Initialize text processor
    INPUT_TEXT = read_all_text_files(config.TEXT_DIRECTORY)
    text_processor = TextProcessor(INPUT_TEXT)
    
    # Create text attention arrays
    text_attention = text_processor.create_text_attention(
        text_attention_span_length=config.TEXT_ATTENTION_SPAN_LENGTH,
        text_attention_weight=config.TEXT_ATTENTION_WEIGHT
    )
    
    # Initialize model handler
    model_handler = ModelHandler(text_processor, config.SAVE_DIR)
    

    if config.TRAINING:
        # Prepare and train model (missing word)
        y, X = model_handler.prepare_training_data(text_attention, config.TEXT_ATTENTION_SPAN_LENGTH)

        print("\nSample input/output data visualization:")
        print_aligned_vocabulary_and_array_combo(
            text_processor.vocabulary_itw, X[0], y[0]
        )
        model_handler.train(X, y, config.BATCH_SIZE, config.EPOCHS)

        # Prepare and train model (nearby words)
        X, y = model_handler.prepare_training_data(text_attention, config.TEXT_ATTENTION_SPAN_LENGTH)
        
        print("\nSample input/output data visualization:")
        print_aligned_vocabulary_and_array_combo(
            text_processor.vocabulary_itw, X[0], y[0]
        )
        model_handler.train(X, y, config.BATCH_SIZE, config.EPOCHS)

    # Initialize API handler if needed
    api_handler = APIHandler() if config.USE_API else None
    
    # Interactive prediction loop
    last_output: Optional[str] = None
    while True:
        user_input = input("\nEnter a word (enter '.' to stop): ")
        
        if user_input == '.':
            print("Stopping the program.")
            break
            
        # Use last output if input is empty
        if user_input == '' and last_output is not None:
            user_input = last_output
            send_to_api = False
        else:
            send_to_api = True
            
        # Make prediction
        sample, nn_result = model_handler.predict(user_input)
        
        # Display results
        print("\nPrediction results:")
        print_aligned_vocabulary_and_array_combo(
            text_processor.vocabulary_itw, 
            sample[0], 
            nn_result.flatten()
        )
        
        # Handle API interactions
        if config.USE_API:
            api_handler.process_prediction(
                text_processor.vocabulary_itw,
                sample,
                nn_result,
                send_to_api
            )
        
        # Update last output
        values = return_aligned_vocabulary_and_array(
            text_processor.vocabulary_itw, 
            nn_result[0], 
            threshold=0.01
        )
        last_output = max(values, key=values.get)
        print("[d] prediction:", last_output)


if __name__ == "__main__":
    main()