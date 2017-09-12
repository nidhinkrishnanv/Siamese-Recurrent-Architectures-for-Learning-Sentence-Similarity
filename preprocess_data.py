import os
import time
import glob
import sys
import pickle
import numpy as np
import gensim

import torch
import torch.optim as O
import torch.nn as nn
import json
from nltk.corpus import stopwords

stopWords = set(stopwords.words('english'))

datapath = '../data/SICK/'


def save_data(is_train_gensim=False):
    labelDict = {'neutral':0, 'entailment':1, 'contradiction':2, '-':0}
    dset_types = ['TRAIN', 'TRIAL', 'TEST']
    vocab = set()
    gensim_train_data = []
        
    data = {dset_type:[] for dset_type in dset_types}
    print ('preprossing ' + 'SICK...')
    fpr = open(datapath + 'SICK.txt', 'r')
    fpr.readline()
    count = 0
    for line in fpr:
        if count >= 4:
            break
        sentences = line.strip().split('\t')
        tokens = [[token for token in sentences[x].split(' ') if token != '(' and token != ')'] for x in [1, 2]]
        #and token not in stopWords] for x in [1, 2]
        
        #For training gensim model
        if is_train_gensim:
            gensim_train_data.extend([tokens[0], tokens[1]])

        vocab.update(tokens[0]); vocab.update(tokens[1])

        #check for empty strings
        if len(tokens[0]) != 0 and len(tokens[1]) != 0:
            data[sentences[11]].append(([tokens[0], tokens[1]], float(sentences[4])))

        count += 1
    fpr.close()
    # print(vocab)

    #create word_to_idx and save     
    word_to_idx = {word:i for i, word in enumerate(vocab, 1)}
    word_to_idx["[<pad>]"] = 0
    with open('data/word_to_idx.json', 'w') as f:
        json.dump(word_to_idx, f)
    print("Vocab size : ", len(vocab))


    #Saving dataset
    for dset_type in dset_types:
        print(dset_type)
        print(data[dset_type][:4])
        data[dset_type] = convert_data_to_word_to_idx(data[dset_type])
        with open('data/' + dset_type + '.pkl', 'wb') as f:
            pickle.dump(data[dset_type], f)
    
    if is_train_gensim:
        model = gensim.models.Word2Vec(gensim_train_data, size=300, window=1, min_count=1, workers=4)
        model.save('model/word2vec_snli.model')

    print("Vocab size : ", len(word_to_idx))

def get_data(dset_type):
    with open('data/' + dset_type + '.pkl', 'rb') as f:
        return pickle.load(f)

def convert_data_to_word_to_idx(data):
    data_to_word_to_idx = []
    word_to_idx = get_word_to_ix()
    for sentences, score in data:
        for i in range(2):
            sentences[i] = [word_to_idx[w] for w in sentences[i]]

        data_to_word_to_idx.append((np.array(sentences[0], dtype=np.int64),
            np.array(sentences[1], dtype=np.int64),
            np.array([score], dtype=np.float64)))

    return data_to_word_to_idx


    
def get_word_to_ix():
    with open('data/word_to_idx.json') as f:
        return json.load(f)

if __name__ == "__main__":
    if not os.path.isdir('data'):
        os.mkdir('data')
    # vocab()
    save_data()