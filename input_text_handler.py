import os
import chardet
from config import TEXT_DIRECTORY

def read_all_text_files(directory):
    """
    Reads all .txt files in the specified directory with various encodings and concatenates their contents into a single string.

    :param directory: Path to the directory containing .txt files.
    :return: A single string containing the combined contents of all .txt files.
    """
    combined_text = ""

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                text = raw_data.decode(encoding)
                combined_text += text + "\n"  # Add a newline to separate file contents

    return combined_text

if __name__ == "__main__":
    # Example usage
    combined_text = read_all_text_files(TEXT_DIRECTORY)
    print(combined_text)  # Print the combined text