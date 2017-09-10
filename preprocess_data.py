import os
import time
import glob
import sys
import pickle

import torch
import torch.optim as O
import torch.nn as nn
import json


def vocab():
    labelDict = {'neutral':0, 'entailment':1, 'contradiction':2, '-':0}
    vocab = set()
    max_len = 0
    print ('preprossing ' + 'SICK' + '...')
    fpr = open('data/SICK/'+'SICK.txt', 'r')
    count = 0
    fpr.readline()
    for line in fpr:
        # if count > 1:
        #     break
        sentences = line.strip().split('\t')
        tokens = [token for token in sentences[1].split(' ') if token != '(' and token != ')']
        tokens += [token for token in sentences[2].split(' ') if token != '(' and token != ')' ]
        max_len = max([max_len, len(tokens)])
        vocab.update(tokens)
        count += 1       
    fpr.close()
    word_to_idx = {word:i for i, word in enumerate(vocab, 1)}
    word_to_idx["[<pad>]"] = 0
    print("Vocab size : ", len(word_to_idx))
    print("max_len " ,max_len)
    
    with open('data/word_to_idx.json', 'w') as f:
        json.dump(word_to_idx, f)
    with open('data/max_len.pkl', 'wb') as f:
        pickle.dump(max_len, f)

    print ('SICK vocab preprossing finished!')

    
def get_word_to_ix():
    with open('data/word_to_idx.json') as f:
        return json.load(f)

def get_max_len():
    with open('data/max_len.pkl', 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   vocab()