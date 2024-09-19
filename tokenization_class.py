import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SmilesPE.pretokenizer import atomwise_tokenizer
from SmilesPE.pretokenizer import kmer_tokenizer


class MoleculeTokenizer:

    # Initialize the class variables
    def __init__(self, x_win, stride):
        self.x_win = x_win  # Sliding window size (subsequence length)
        self.stride = stride  # Step size for moving the center
        self.dict_subsequence = {}  # Single dictionary for all molecules
        self.final_list = []  # List of all subsequences

    # Extract subsequences (patches) from a molecule string with padding
    def window_extract(self, molecule, center_index):
        half_win = self.x_win // 2
        start_index = center_index - half_win
        end_index = center_index + half_win + (self.x_win % 2)

        # Adjust the start and end for padding
        subsequence = molecule[max(0, start_index):min(len(molecule), end_index)]

        # Pad zeros at the beginning if start_index is negative
        if start_index < 0:
            subsequence = '0' * abs(start_index) + subsequence

        # Pad zeros at the end if end_index exceeds molecule length
        if end_index > len(molecule):
            subsequence = subsequence + '0' * (end_index - len(molecule))

        return subsequence

    # Extract subsequences from a list of molecules with zero-padding
    def slide_window(self, molecule_list):
        for molecule in molecule_list:
            molecule_len = len(molecule)
            center_index = 0
            subsequence_list = []  # To store subsequences for the current molecule

            while center_index < molecule_len:
                # Extract subsequence centered on the current position
                subsequence = self.window_extract(molecule, center_index)
                subsequence = subsequence.rstrip('0').lstrip('0')

                # Add subsequence to the list if it's non-empty
                if subsequence:
                    subsequence_list.append(subsequence)

                # Move the sliding window center by the stride
                center_index += self.stride

            # Add the molecule and its subsequences to the single dictionary
            self.dict_subsequence[molecule] = subsequence_list
            self.final_list.append(subsequence_list)

        return self.dict_subsequence, self.final_list
    
    def spe_atomwise(self, molecule_list):
        # Create a list of tokenized molecules
        tokenized_molecules = [atomwise_tokenizer(molecule) for molecule in molecule_list]
        return tokenized_molecules
    
    def spe_kmer(self, molecule_list, ngram):
        # Create a list of tokenized molecules
        tokenized_molecules = [kmer_tokenizer(molecule,ngram) for molecule in molecule_list]
        return tokenized_molecules
