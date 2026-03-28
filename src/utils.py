import json

import numpy as np


def make_noisy_state(state, noise_level, seed=0):
    """
    Adds random noise to a binary state by flipping bits with a given probability.

    This function introduces controlled noise to a binary state array by randomly
    flipping bits (0 to 1 or 1 to 0) based on the specified noise level. This is
    useful for testing the robustness of Hopfield networks and their ability to
    recover from corrupted inputs.

    :param state: The original binary state as a numpy array.
    :type state: np.ndarray
    :param noise_level: The probability of flipping each bit (0.0 to 1.0).
    :type noise_level: float
    :param seed: Random seed for reproducible noise generation. Defaults to 0.
    :type seed: int

    :return: A noisy version of the input state with bits flipped according to noise_level.
    :rtype: np.ndarray
    """
    rn_state = np.random.RandomState(seed)
    flip_mask = rn_state.rand(*state.shape) < noise_level
    return np.logical_xor(state, flip_mask).astype(int)


def load_letters_from_json(filename):
    """
    Loads letter representations from a JSON file.

    This function reads a JSON file containing binary representations of letters
    or characters, typically used for pattern recognition tasks in neural networks.

    :param filename: The path to the JSON file containing letter representations.
    :type filename: str

    :return: A dictionary mapping letters to their binary representations.
    :rtype: dict

    :raises FileNotFoundError: If the specified file cannot be found.
    :raises json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(filename, "r") as file:
        letters = json.load(file)
    return letters


def generate_binary_matrix(letter, letters_file="letters.json"):
    """
    Generates a binary matrix representation of a single letter.

    This function converts a letter character into its binary matrix representation
    by looking it up in a JSON file containing pre-defined letter patterns. The
    resulting matrix can be used as input for pattern recognition tasks.

    :param letter: The letter character to convert to binary matrix.
    :type letter: str
    :param letters_file: Path to the JSON file containing letter definitions.
                        Defaults to "letters.json".
    :type letters_file: str

    :return: A 2D numpy array representing the binary pattern of the letter.
    :rtype: np.ndarray

    :raises ValueError: If the specified letter is not found in the letters file.
    """
    # Load letters from JSON file
    letters = load_letters_from_json(letters_file)

    # Get the binary representation of the letter
    binary_representation = letters.get(letter.upper(), None)

    if binary_representation is None:
        raise ValueError(f"Letter {letter} not found in the letters file")

    # Convert binary representation to matrix
    binary_matrix = []
    for row in binary_representation:
        binary_row = [int(pixel) for pixel in row]
        binary_matrix.append(binary_row)

    return np.array(binary_matrix)


def string_to_matrix(string, letters_file="letters.json"):
    """
    Converts a string of characters into a concatenated binary matrix.

    This function takes a string and converts each character into its binary matrix
    representation, then concatenates all matrices horizontally with separators.
    Spaces in the string are replaced with underscores for lookup purposes.

    :param string: The input string to convert to binary matrix representation.
    :type string: str
    :param letters_file: Path to the JSON file containing letter definitions.
                        Defaults to "letters.json".
    :type letters_file: str

    :return: A 2D numpy array representing the concatenated binary patterns of all
            characters in the string, separated by columns of zeros.
    :rtype: np.ndarray

    :raises ValueError: If any character in the string is not found in the letters file.
    """
    matrices = [
        generate_binary_matrix(letter, letters_file)
        for letter in string.upper().replace(" ", "_")
    ]
    matrices = [
        np.concatenate((letter, np.zeros((letter.shape[0], 1))), axis=1)
        for letter in matrices
    ]
    return np.concatenate(matrices, axis=1)


def strings_to_matrix(strings, letters_file="letters.json"):
    """
    Converts multiple strings into a single concatenated binary matrix.

    This function takes a list of strings and converts each into its binary matrix
    representation, then stacks them vertically with separators. All matrices are
    padded to have the same width, and rows of zeros separate different strings.

    :param strings: A list of strings to convert to binary matrix representations.
    :type strings: list[str]
    :param letters_file: Path to the JSON file containing letter definitions.
                        Defaults to "letters.json".
    :type letters_file: str

    :return: A 2D numpy array representing all input strings stacked vertically,
            with zero-padding for uniform width and separator rows between strings.
    :rtype: np.ndarray

    :raises ValueError: If any character in any string is not found in the letters file.
    """
    mat_list = [string_to_matrix(s, letters_file) for s in strings]
    max_columns = max(m.shape[-1] for m in mat_list)
    mat_list_padded = []
    for m in mat_list:
        mat = np.zeros((m.shape[0], max_columns))
        mat[: m.shape[0], : m.shape[1]] = m
        mat_list_padded.append(mat)
        mat_list_padded.append(np.zeros((1, max_columns)))
    return np.concatenate(mat_list_padded, axis=0)
