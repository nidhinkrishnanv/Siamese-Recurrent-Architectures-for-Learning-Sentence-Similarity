import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from create_word_to_ix import get_word_to_ix, get_max_len

import numpy as np

def packed_collate_fn(data):

    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    enumerated_data = [[idx, x[0], x[1], x[2]] for idx, x in enumerate(data)]
    
    #sort for sentence1 and create batch
    enumerated_data.sort(key=lambda sent: len(sent[1]), reverse=True)
    sent1_order, sent1_seqs, _, _ = zip(*enumerated_data)
    sent1_seqs, sent1_lengths = merge(sent1_seqs)

    #sort for sentence2 and create batch
    enumerated_data.sort(key=lambda sent: len(sent[2]), reverse=True)
    sent2_order, _, sent2_seqs, _ = zip(*enumerated_data)
    sent2_seqs, sent2_lengths = merge(sent2_seqs)

    score = [x[2].numpy() for x in data]
    score = torch.Tensor(score)

    return sent1_seqs, sent1_lengths, sent1_order, sent2_seqs, sent2_lengths, sent2_order, score

class SICKDataset(Dataset):
    def __init__(self, dset_type, transform=None):
        self.data = []
        self.transform = transform
        self.word_to_idx = get_word_to_ix()
        self.max_len = get_max_len()
        self.read_file(dset_type)

    def read_file(self, dset_type):
        max_list = []
        # dataset = {'train':[], 'test':[], 'dev':[]}
        data = []
        print ('preprossing ' + 'SICK' + '...')
        fpr = open('data/SICK/'+'SICK.txt', 'r')
        fpr.readline()
        count = 0
        for line in fpr:
            if count > 3:
                break
            sentences = line.strip().split('\t')
            if dset_type != sentences[11]:
                continue
            tokens = [[token for token in sentences[x].split(' ') if token != '(' and token != ')'] for x in [1, 2]]
            data.append(([tokens[0], tokens[1]], float(sentences[4])))
            count += 1
        fpr.close()
        print(data)
        # print ('SICK preprossing ' + dset_type + ' finished!')
        # print("Vocab size : ", len(self.word_to_idx))
        self.data = self.convert_data_to_word_to_idx(data)
        

    def convert_data_to_word_to_idx(self, data):
        data_to_word_to_idx = []

        for sentences, score in data:
            # sentences_pad = [np.zeros(self.max_len) for i in range(2)]
            for i in range(2):
                sentences[i] = [self.word_to_idx[w] for w in sentences[i]]
                # sentences_pad[i][:len(sentences[i])] = sentences[i]

            data_to_word_to_idx.append((np.array(sentences[0], dtype=np.int64),
                np.array(sentences[1], dtype=np.int64),
                # np.array([len(sentences[0])], dtype=np.int64),
                # np.array([len(sentences[1])], dtype=np.int64),
                np.array([score], dtype=np.float64)))

        # print('data_to_word_to_idx', data_to_word_to_idx[0])
        return data_to_word_to_idx

    def len_of_sentence(self, input_tuple):
        sentence, _ = input_tuple
        return len(sentence)

    def __len__(self):
        return(len(self.data))

    def len_vocab(self):
        return len(self.word_to_idx)

    def __getitem__(self, idx):
        sample = {'sentence1': self.data[idx][0],
            'sentence2': self.data[idx][1],
            # 'len_sent1' : self.data[idx][2],
            # 'len_sent2' : self.data[idx][3],
            'score' : self.data[idx][2]}
        if self.transform:
            sample = self.transform(sample)
        # print('sample')
        # print(sample)
        return sample

class ToTensor(object):
    """Covert ndarray in sample to Tensors."""
    def __call__(self, sample):
        sentence1, sentence2, score = sample['sentence1'], sample['sentence2'], sample['score']
        # len_sent1, len_sent2 = sample['len_sent1'], sample['len_sent2']
        # return {"sentence1" : torch.from_numpy(sentence1),
        #         "sentence2" : torch.from_numpy(sentence2),
        #         'score': torch.from_numpy(score)}

        return [torch.from_numpy(sentence1), torch.from_numpy(sentence2), 
            # torch.from_numpy(len_sent1), torch.from_numpy(len_sent2), 
            torch.from_numpy(score)]



if __name__ == "__main__":
    dataset = SICKDataset('TRAIN', transform=transforms.Compose([ToTensor()]))

    # for i in range(2):
    #     sample = dataset[i]
    #     print(sample)

    dataloader = DataLoader(dataset, batch_size=4, num_workers=4, collate_fn=packed_collate_fn)

    # print("dataloader size")
    # print(len(dataloader))

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched,)
        if (i_batch == 5):
            break