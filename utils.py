"""
author: @cesarasa

This file contains utility functions to use for QSAR modeling. 
"""
import os
import deepchem as dc
import numpy as np
import pandas as pd
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
    similarity_matrix = np.eye(num_fingerprints)

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

def load_bace(featurizer : int = 'ECFP',
              splitter = dc.splits.ScaffoldSplitter,
              random_seed : int = 1301) -> tuple:
    """
    Function that loads the BACE dataset from DeepChem and saves it in three different csv files.
    """
    _, bace_datasets, _ = dc.molnet.load_bace_regression(featurizer=featurizer, splitter=splitter, random_seed=random_seed)
    data_path = './data_BACE'
    data_csvs = ['scaffold_bace_train.csv',
                 'scaffold_bace_valid.csv',
                 'scaffold_bace_test.csv']

    # Create a directory to store the data
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Create a list of dataframes:
    dfs = []
    # Save the data
    for i, dataset in enumerate(bace_datasets):
        smiles = dataset.ids
        targets = dataset.y
        fingerprints = dataset.X
        df_fp = pd.DataFrame(fingerprints)
        df_fp.columns = ['fp_%d' % i for i in range(df_fp.shape[1])]
        df_fp['targets'] = targets
        df_fp['smiles'] = smiles

        # Move smiles to the first column
        cols = df_fp.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df_fp = df_fp[cols]

        dfs.append(df_fp)
        # Save the three datasets:
        df_fp.to_csv(os.path.join(data_path, data_csvs[i]), index=False)
    return dfs
