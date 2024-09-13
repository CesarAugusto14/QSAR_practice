import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SmilesPE.pretokenizer import atomwise_tokenizer
from SmilesPE.pretokenizer import kmer_tokenizer
from SmilesPE.pretokenizer import atomwise_tokenizer
from SmilesPE.pretokenizer import kmer_tokenizer
import pandas as pd
import numpy as np
import os
import sys
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import tokenization_class as tc
import vectorization_class as vc
from sklearn.model_selection import train_test_split

def data_load(data_path):
    # Load the data
    data = pd.read_csv(data_path)
    X=data['smiles']
    y=data['target']
    return X,y

def random_spliting_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def main(argv):
    # file path
    arg1=argv[1]
    #tokenization_type
    arg2=argv[2]
    #vectorization_type
    arg3=argv[3]

    # Load the data
    X,y=data_load(arg1)
    #split the data
    X_train, X_test, y_train, y_test=random_spliting_data(X,y)
    # Tokenization
    token = tc.MoleculeTokenizer(x_win=5, stride=1)
    if arg2=='atomwise':
        token_list_train= token.atomwise_tokenizer(X_train)
        token_list_test = token.atomwise_tokenizer(X_test)

    elif arg2=='kmer':
        token_list_train = token.kmer_tokenizer(X_train)
        token_list_test = token.kmer_tokenizer(X_test)
    elif arg2=='swindow':
        _,token_list_train = token.slide_window(X_train)
        _,token_list_test = token.slide_window(X_test)


    # Vectorization
    vector = vc.MoleculeVectorizer(tokenizer_object=token,tokenizer_type=arg2)
    if arg3=='word2vec':
        model = vector.wordvec(token_list_train, vector_size=100, window=5, min_count=1, workers=4, tokenizer_type=arg2)
        
        train_vec = np.array(vector.sentence_vectorizer(token_list_train, model))
        test_vec = np.array(np.array(vector.sentence_vectorizer(token_list_test, model)))
    elif arg3=='doc2vec':
        #print(token_list_train)
        model = vector.docvec(token_list_train, vector_size=100, window=5, min_count=1, workers=4, tokenizer_type=arg2)
        train_vec=np.array(vector.vector_doc(model, token_list_train))
        test_vec=np.array(vector.vector_doc(model, token_list_test))
    print(train_vec.shape)
    print(test_vec.shape)
    return True

if __name__ == "__main__":
    main(sys.argv)