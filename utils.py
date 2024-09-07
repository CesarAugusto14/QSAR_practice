"""
author: @cesarasa

This file contains utility functions to use for QSAR modeling. 
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

def get_similarity(smiles : list,
                   similarity_metric : str = 'Tanimoto') -> tuple:
    """
    Function that computes the similarity matrix of 
    a list of smiles. 
    
    Inputs:
    - smiles : list of SMILES strings.
    - similarity_metric : string with the similarity metric to use, it
                            can be 'Tanimoto' or 'Dice'.
    
    Outputs:
    - similarity_matrix : numpy array with the similarity matrix
    - fingerprints      : list of RDKit fingerprints
    - mols              : list of RDKit molecules
    
    TODO: 
    - Add more similarity metrics.
    - Add different ways to create the fingerprints. 
    """
    # Convert SMILES to RDKit format:
    mols = [Chem.MolFromSmiles(x) for x in smiles]

    # Get Morgan Fingerprints: radius 2 for diameter 4 (ECFP4)
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol=x,
                                                          radius=2,
                                                          nBits=1024) for x in mols]

    # Define the similarity metric
    if similarity_metric == 'Tanimoto':
        similarity_function = DataStructs.TanimotoSimilarity
    elif similarity_metric == 'Dice':
        similarity_function = DataStructs.DiceSimilarity
    else:
        raise ValueError('Similarity metric not recognized. Please use Tanimoto or Dice')

    # Create a similarity matrix
    num_fingerprints = len(fingerprints)
    # We know that the similarity of a molecule with itself is 1, so we can
    # initialize the matrix with ones in the diagonal.
    similarity_matrix = np.eye((num_fingerprints, num_fingerprints))

    # This is a symmetric matrix, it is not necessary to compute all the values,
    # so we can iterate over the upper triangular part of the matrix, and then
    # copy the values to the lower triangular part.
    #
    # This is a common trick to avoid computing the same value twice.
    for i in range(num_fingerprints):
        # We start from i+1 to avoid computing the similarity of a molecule with itself
        for j in range(i+1, num_fingerprints):
            similarity = DataStructs.FingerprintSimilarity(fingerprints[i],
                                                           fingerprints[j],
                                                           metric=similarity_function)
            # We copy the value to the upper triangular part of the matrix
            similarity_matrix[i][j] = similarity
            # We copy the value to the lower triangular part of the matrix
            similarity_matrix[j][i] = similarity

    return similarity_matrix, fingerprints, mols
