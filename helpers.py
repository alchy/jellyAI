def print_aligned_vocabulary_and_array(vocabulary, array):
    """
    Prints the vocabulary and the corresponding values in the array aligned side by side.

    Args:
        vocabulary (list): The list of vocabulary words.
        array (list): The list of corresponding values.
    """
    # Získání maximální délky slova ve slovníku pro zarovnání
    max_length = max(len(word) for word in vocabulary)

    # Tisk hlavičky
    print(f"{'Vocabulary':<{max_length}} | {'NLM Index Array'}")
    print("-" * (max_length + 3 + 15))

    # Tisk slovníku a odpovídajících hodnot z nlm_index_array
    for i, word in enumerate(vocabulary):
        value = array[i] if i < len(array) else 0.0
        print(f"{word:<{max_length}} | {value:.1f}")


def print_aligned_vocabulary_and_array_combo(vocabulary, input_array, output_array):
    """
    Prints the vocabulary and the corresponding values in the input and output arrays aligned side by side.

    Args:
        vocabulary (list): The list of vocabulary words.
        input_array (list): The list of corresponding input values.
        output_array (list): The list of corresponding output values.
    """
    # Získání maximální délky slova ve slovníku pro zarovnání
    max_length = max(len(word) for word in vocabulary)

    # Tisk hlavičky
    print(f"{'Vocabulary':<{max_length}} | {'Input Array':<12} | {'Output Array':<12}")
    print("-" * (max_length + 3 + 12 + 3 + 12))

    # Tisk slovníku a odpovídajících hodnot z input_array a output_array
    for i, word in enumerate(vocabulary):
        input_value = input_array[i] if i < len(input_array) else 0.0
        output_value = output_array[i] if i < len(output_array) else 0.0
        print(f"{word:<{max_length}} | {input_value:<12.1f} | {output_value:<12.1f}")
