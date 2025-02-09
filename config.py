"""
Configuration settings for the text processing and neural network application.
"""

# Program behavior flags
USE_API = True
TRAINING = True

# Neural Network configuration
TEXT_ATTENTION_SPAN_LENGTH = 7
TEXT_ATTENTION_WEIGHT = 0.2
SAVE_DIR = "./model_checkpoints"

# Training parameters
BATCH_SIZE = 32
EPOCHS = 8

# Path to the directory containing .txt files
TEXT_DIRECTORY = 'input_text'

# Sample input text
INPUT_TEXT = (
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