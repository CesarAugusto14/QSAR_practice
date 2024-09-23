import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SmilesPE.pretokenizer import atomwise_tokenizer
from SmilesPE.pretokenizer import kmer_tokenizer
import os
import sys
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import tokenization_class as tc
#import vectorization_class as vc
from measurements import nRMSE
from gensim.models import Word2Vec
from sklearn.linear_model import LinearRegression
#from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import hashlib
import torch 
import torch.nn as nn

# loading the data
#
def data_load(data_path):
    # Load the data
    data = pd.read_csv(data_path)
    X=data['smiles'].values
    y=data['target'].values
    return X,y

# Tokenization
#
def tokenization(X, tokenization_type,x_wind):
    # Tokenization object
    #
    token = tc.MoleculeTokenizer(x_win=x_wind, stride=1)
    # using SPE atomwise tokenization
    #
    if tokenization_type=='atomwise':
        token_list= token.spe_atomwise(X)
    # using SPE kmer tokenization
    #
    elif tokenization_type=='kmer':
        token_list = token.spe_kmer(X,ngram=6)
    # using slide window tokenization
    #
    elif tokenization_type=='swindow':
        _,token_list = token.slide_window(X)
    return token_list

# finding unique tokens to create dictionary
#
def uniq_values(token_list):
    u=set()
    for x in token_list:
        for y in x:
            u.add(y)
    return u
# creating hashing dictionary and considerin parts of it that is between 0 and 100
#
def hashing_func(string, used_hashes=set()):
    hash_item =hashlib.md5(string.encode()).hexdigest()
    # considering the int part of hashing function
    #
    numeric_hash = int(hash_item, 16) % 100
    # creating unique hashing function
    #
    while numeric_hash in used_hashes:
        numeric_hash = (numeric_hash + 1) % 100
    used_hashes.add(numeric_hash)
    return numeric_hash
# creating dictionary of hashed tokens
#
def dict_hash(token_list):
    unique_values = uniq_values(token_list)
    dict_hash = {}
    for x in unique_values:
        dict_hash[x] = hashing_func(x)
    return dict_hash
# vectorization of tokens
#
def my_vectorization(token_list,dict_hash):
    list_whole=[]
    for element in token_list:
        l1=[]
        for i in range(len(element)):
            #  doing normalization based on the lenght of the tokenized list for each chemical compound
            #
            element[i] = dict_hash[element[i]]/len(element)
            l1.append(element[i])
        list_whole.append(l1)
    return list_whole
# doing zero padding befor embedding
#
def padding(list_whole):
    max_len = max([len(x) for x in list_whole])
    padded_vectors = [x + [0]*(max_len-len(x)) for x in list_whole]
    return padded_vectors

# embedding the padded vectors
#
def embeding(padded_vectors, num_embedings, embeding_dim):
    embedings = nn.Embedding(num_embedings, embeding_dim, padding_idx=0)
    input = torch.LongTensor(padded_vectors)
    out = embedings(input)
    out_array = out.detach().numpy()
    out_put = out_array.reshape(-1, out_array.shape[-1])
    return out_put
# splitting the data into train and test
#
def train_test(X,y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

def main(argv):
    # data 
    sys.arg1=argv[1]
    # type of tokenization
    sys.arg2=argv[2]
    # loading the data
    X,y= data_load(sys.arg1)
    # Tokenization
    X_token = tokenization(X, sys.arg2,5)
    # saved the tokenized data
    with open('tokenization.txt', 'w') as f:
        for sublist in X_token:
            f.write(','.join(sublist) + '\n')
    # creating the dictionary of hashed
    dict_h = dict_hash(X_token)
    # vectorization
    X_vector = my_vectorization(X_token,dict_h)
    # padding
    X_padded = padding(X_vector)
    # daved the padded data
    with open('vect.txt', 'w') as f:
        for sublist in X_padded:
            f.write(','.join(sublist) + '\n')
    # embedding
    X_embeded = embeding(X_padded, 100, 10)
    # splitting the data
    x_train, x_test, y_train, y_test = train_test(X_embeded, y)

    
    modellr=linear_model.Lasso(alpha=0.000001)
    modellr.fit(np.array(x_train), y_train)
    y_p_train = modellr.predict(x_train)
    print(y_p_train.shape)
    print(y_train.shape)
    y_p_test = modellr.predict(x_test)
    # # Calculate the NRMSE
    nrmse_tr_lr, r2_tr_lr, mse_tr_lr, rmse_tr_lr = nRMSE(y_train, y_p_train)
    nrmse_te_lr, r2_te_lr, mse_te_lr, rmse_te_lr = nRMSE(y_test, y_p_test)
    print(f'Training NRMSE_lr: {nrmse_tr_lr}, Test NRMSE_lr: {nrmse_te_lr}')
    print(f'Training R2_lr: {r2_tr_lr}, Test R2_lr: {r2_te_lr}')
    print(f'Training MSE_lr: {mse_tr_lr}, Test MSE_lr: {mse_te_lr}')
    print(f'Training RMSE_lr: {rmse_tr_lr}, Test RMSE_lr: {rmse_te_lr}')
    # Random Forest
    modelrf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=0)
    modelrf.fit(x_train, y_train)
    y_p_train_rf = modelrf.predict(x_train)
    y_p_test_rf = modelrf.predict(x_test)
    # Calculate the NRMSE
    nrmse_tr_rf, r2_tr_rf, mse_tr_rf, rmse_tr_rf = nRMSE(y_train, y_p_train_rf)
    nrmse_te_rf, r2_te_rf, mse_te_rf, rmse_te_rf = nRMSE(y_test, y_p_test_rf)
    print(f'Training NRMSE_rf: {nrmse_tr_rf}, Test NRMSE_rf: {nrmse_te_rf}')
    print(f'Training R2_rf: {r2_tr_rf}, Test R2_rf: {r2_te_rf}')
    print(f'Training MSE_rf: {mse_tr_rf}, Test MSE_rf: {mse_te_rf}')
    print(f'Training RMSE_rf: {rmse_tr_rf}, Test RMSE_rf: {rmse_te_rf}')
    return True

if __name__ == '__main__':
    main(sys.argv)




