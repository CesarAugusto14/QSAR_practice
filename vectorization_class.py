from SmilesPE.pretokenizer import atomwise_tokenizer
from SmilesPE.pretokenizer import kmer_tokenizer
import pandas as pd
import numpy as np
import os
import sys
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import tokenization_class as tc

class MoleculeVectorizer:
    def __init__(self,tokenizer_object,tokenizer_type):

        self.tokenizer_type = tokenizer_type
        self.tokenizer_object = tokenizer_object

    def wordvec(self,tokenized_sentences, vector_size, window, min_count, workers, tokenizer_type):
        modelv = Word2Vec(tokenized_sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
        modelv.save(f'{tokenizer_type}_word2vec.model')
        return modelv

    def sentencevec(self,sentence, modelv):
        """Convert a sentence to a vector by averaging word vectors."""
        word_vectors = [modelv.wv[word] for word in sentence if word in modelv.wv]
        if len(word_vectors) == 0:
            return np.zeros(modelv.vector_size)  # Return zero vector if no words are in the model
        return np.mean(word_vectors, axis=0)
    
    def sentence_vectorizer(self,tokenized_sentences, modelv):
        sentence_vectors = [self.sentencevec(sentence, modelv) for sentence in tokenized_sentences]
        return sentence_vectors

    def docvec(self,tokenized_sentences, vector_size, window, min_count, workers, tokenizer_type):
        tagged_data = [TaggedDocument(doc, [i]) for i, doc in enumerate(tokenized_sentences)]
        modeldoc = Doc2Vec(tagged_data, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
        modeldoc.build_vocab(tagged_data)
        modeldoc.train(tagged_data, total_examples=modeldoc.corpus_count, epochs=modeldoc.epochs)
        modeldoc.save(f'{tokenizer_type}_doc2vec.model')
        return modeldoc
    
    def vector_doc(self,modeldoc,tokenized_sentences):
            vector_doc = [modeldoc.dv[i] for i in range(len(tokenized_sentences))]
            return vector_doc


