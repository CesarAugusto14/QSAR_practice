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


def data_load(data_path):
    # Load the data
    data = pd.read_csv(data_path)
    X=data['SMILES'].values
    y=data['docking score'].values
    return X,y

def tokenization(X, tokenization_type):
    # Tokenization
    token = tc.MoleculeTokenizer(x_win=5, stride=1)
    if tokenization_type=='atomwise':
        token_list= token.atomwise_tokenizer(X)
    elif tokenization_type=='kmer':
        token_list = token.kmer_tokenizer(X)
    elif tokenization_type=='swindow':
        _,token_list = token.slide_window(X)
    return token_list

def vectorization(token_list):
    model = Word2Vec(token_list, vector_size=200, window=5, min_count=1, workers=4)
    model.save('word2vec.model')
    return model
def sentencevec(sentence, model):
        """Convert a sentence to a vector by averaging word vectors."""
        word_vectors = [model.wv[word] for word in sentence if word in model.wv]
        if len(word_vectors) == 0:
            return np.zeros(model.vector_size)  # Return zero vector if no words are in the model
        return np.mean(word_vectors, axis=0)
    
def sentence_vectorizer(token_list, model):
    sentence_vectors = [sentencevec(sentence, model) for sentence in token_list]
    return sentence_vectors

def main(argv):
    # data train
    sys.arg1=argv[1]
    # data test
    sys.arg2=argv[2]
    # Tokenization type
    sys.arg3=argv[3]
    x_train, y_train = data_load(sys.arg1)
    x_test, y_test = data_load(sys.arg2)
    # Tokenization
    x_train_token = tokenization(x_train, sys.arg3)
    
    x_test_token = tokenization(x_test, sys.arg3)
    # Vectorization
    modelv = vectorization(x_train_token)
    x_train_vector = sentence_vectorizer(x_train_token, modelv)
    x_test_vector = sentence_vectorizer(x_test_token, modelv)
    # Linear Regression
    modellr = LinearRegression()
    modellr.fit(np.array(x_train_vector), y_train)
    y_p_train = modellr.predict(np.array(x_train_vector))
    print(y_p_train.shape)
    print(y_train.shape)
    y_p_test = modellr.predict(np.array(x_test_vector))
    # # Calculate the NRMSE
    nrmse_tr_lr, r2_tr_lr, mse_tr_lr, rmse_tr_lr = nRMSE(y_train, y_p_train)
    nrmse_te_lr, r2_te_lr, mse_te_lr, rmse_te_lr = nRMSE(y_test, y_p_test)
    print(f'Training NRMSE_lr: {nrmse_tr_lr}, Test NRMSE_lr: {nrmse_te_lr}')
    print(f'Training R2_lr: {r2_tr_lr}, Test R2_lr: {r2_te_lr}')
    print(f'Training MSE_lr: {mse_tr_lr}, Test MSE_lr: {mse_te_lr}')
    print(f'Training RMSE_lr: {rmse_tr_lr}, Test RMSE_lr: {rmse_te_lr}')
    # Random Forest
    modelrf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0)
    modelrf.fit(np.array(x_train_vector), y_train)
    y_p_train_rf = modelrf.predict(np.array(x_train_vector))
    y_p_test_rf = modelrf.predict(np.array(x_test_vector))
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




