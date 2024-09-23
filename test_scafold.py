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


# def data_load(data_path):
#     # Load the data
#     data = pd.read_csv(data_path)
#     X=data['SMILES'].values
#     y=data['docking score'].values
#     short = len(min(X,key=len))
#     long = len(max(X,key=len))
#     print(short)
#     print(long)
#     return X,y,short,long

def data_load(data_path):
    # Load the data
    data = pd.read_csv(data_path)
    X=data['smiles'].values
    y=data['target'].values
    #short = len(min(X,key=len))
    #long = len(max(X,key=len))
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    short_t = len(min(x_train,key=len))
    long_t = len(max(x_train,key=len))
    short_v = len(min(x_test,key=len))
    long_v = len(max(x_test,key=len))
    return x_train, y_train, x_test, y_test,short_t,long_t,short_v,long_v
def tokenization(X, tokenization_type,x_wind):
    # Tokenization
    token = tc.MoleculeTokenizer(x_win=x_wind, stride=1)
    if tokenization_type=='atomwise':
        token_list= token.spe_atomwise(X)
    elif tokenization_type=='kmer':
        token_list = token.spe_kmer(X,ngram=6)
    elif tokenization_type=='swindow':
        _,token_list = token.slide_window(X)
    return token_list

def vectorization(token_list):
    model = Word2Vec(vector_size=200, window=5, min_count=1, workers=4)
    model.build_vocab(token_list)  # Build the vocabulary
    model.train(token_list, total_examples=len(token_list), epochs=model.epochs)  # Train the model
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
    #sys.arg3=argv[3]
    x_train, y_train,x_test,y_test,short_t,long_t,short_v,long_v = data_load(sys.arg1)
    #x_test, y_test,short_v,long_v = data_load(sys.arg2)
    # Tokenization
    x_train_token = tokenization(x_train, sys.arg2,3)
    
    x_test_token = tokenization(x_test, sys.arg2,3)
    # Vectorization
    modelv = vectorization(x_train_token)
    x_train_vector = sentence_vectorizer(x_train_token, modelv)
    x_test_vector = sentence_vectorizer(x_test_token, modelv)
    # Linear Regression
    #modellr = LinearRegression()
    modellr=linear_model.Lasso(alpha=0.000001)
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
    modelrf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=0)
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




