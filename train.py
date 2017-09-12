import torch
import torch.optim as optim
import torch.autograd as autograd
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import numpy as np

import time
import copy
import pickle

from model import MaLSTM
from sick_dataset import SICKDataset, ToTensor, packed_collate_sort


torch.manual_seed(1)

EMBEDDING_DIM = 300
EPOCH = 3
BATCH_SIZE = 4

dset_types = ['TRAIN', 'TRIAL', 'TEST']

dataset = {x : SICKDataset(x, transform=transforms.Compose([ToTensor()]))
            for x in dset_types}
dset_loader = {x : DataLoader(dataset[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=packed_collate_sort)
                for x in dset_types}
dset_sizes = {x : len(dataset[x]) for x in dset_types}

print(dset_sizes)


def train_model(model, loss_function, optimizer, lr_scheduler=None, num_epochs=5):
    since = time.time()

    best_model = model
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        #Each epoch has a training and validation phase
        for phase in ['TRAIN', 'TRIAL']:
            if phase == 'TRAIN':
                if lr_scheduler:
                    optimizer = lr_scheduler(optimizer, epoch)
                model.train()
            else:
                model.eval()
            running_loss = 0.0

            # Iterate ove data
            for data in dset_loader[phase]:
                #get the inputs
                sentences = []; sent_lengths = []; 
                orders = []; model.hidden = []
                for i in range(2):
                    sentences.append(Variable(data['sent'][i].cuda()))
                    sent_lengths.append(data['sent_length'][i])
                    orders.append (Variable(torch.LongTensor(data['order'][i]).cuda()))
                    model.hidden.append(model.init_hidden(data['sent'][i].size()[0]))

                scores = Variable(data['scores'].cuda())

                optimizer.zero_grad()

                outputs = model(sentences, sent_lengths, orders)
                loss = loss_function(outputs, scores.view(-1))

                if phase == 'TRAIN':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]
                # running_corrects += torch.sum(preds == scores.data.view(-1))

                # print(running_corrects)

            epoch_loss = running_loss / dset_sizes[phase]
            # epoch_acc = running_corrects / dset_sizes[phase]

            print('{} Loss: {:.4f} '.format(
                phase, epoch_loss))

            #deep copy the model
            if phase == 'TRIAL' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model = copy.deepcopy(model)
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    return best_model, best_loss

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


weight = 10**(-7*np.random.rand(15))
dropout = np.random.rand(15)
# com = [(x,y) for x in weight for y in dropout]


loss_function = nn.MSELoss()


def hyperparm_tune():
    best_loss = float('inf')
    best_w = 0
    best_d = 0
    for i, (w, d) in enumerate(zip(weight, dropout)):
        print()
        print('{} Weight: {} Dropout: {}'.format(i, w, d))
        model = MaLSTM(dataset['TRAIN'].len_vocab(), EMBEDDING_DIM, 3, d)
        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=w)
        model_ft, acc = train_model(model, loss_function, optimizer, exp_lr_scheduler, EPOCH)
        if acc > best_acc:
            best_acc = acc
            best_w = w
            best_d = d
    print('best_w: {:.4f} best_d: {:.4f} best_loss: {:.4f}'.format(
        best_w, best_d, best_acc))

if __name__ == "__main__":
    model = MaLSTM(dataset['TRAIN'].len_vocab(), EMBEDDING_DIM)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1.287051167874816e-06)
    # model_ft, acc = train_model(model, loss_function, optimizer, exp_lr_scheduler, 5)
    model_ft, acc = train_model(model, loss_function, optimizer, num_epochs=EPOCH)
