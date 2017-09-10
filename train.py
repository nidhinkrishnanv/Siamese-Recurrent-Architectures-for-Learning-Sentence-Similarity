import torch
import torch.optim as optim
import torch.autograd as autograd

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import numpy as np

import time
import copy
import pickle

from model import MaLSTM
from sick_dataloader import SICKDataset, ToTensor


EMBEDDING_DIM = 100

dataset = {x : SICKDataset(x, transform=transforms.Compose([ToTensor()]))
            for x in ['TRAIN', 'TRIAL']}
dset_loader = {x : DataLoader(dataset[x], batch_size=512, shuffle=True, num_workers=4)
                for x in ['TRAIN', 'TRIAL']}
dset_sizes = {x : len(dataset[x]) for x in ['TRAIN', 'TRIAL'] }

print(dset_sizes)


def train_model(model, loss_function, optimizer, lr_scheduler=None, num_epochs=5):
    since = time.time()

    best_model = model
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        #Each epoch has a training and validation phase
        for phase in ['TRAIN', 'TRIAL']:
            if phase == 'TRAIN':
                optimizer = lr_scheduler(optimizer, epoch)
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            # Iterate ove data
            for data in dset_loader[phase]:
                #get the inputs
                sentences = Variable(data['sentence'].cuda())
                scores = Variable(data['label'].cuda())
                # print(model.embeddings.weight.data[0])

                optimizer.zero_grad()

                outputs = model(sentences)
                _, preds = torch.max(outputs.data, 1)
                loss = loss_function(outputs, scores.view(-1))

                if phase == 'TRAIN':
                    loss.backward()
                    optimizer.step()

                # print(preds)
                # print(scores.data)
                # print(preds == scores.data)
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == scores.data.view(-1))

                # print(running_corrects)

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            #deep copy the model
            if phase == 'TRIAL' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model, best_acc

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


loss_function = nn.NLLLoss()


def hyperparm_tune():
    best_acc = 0
    best_w = 0
    best_d = 0
    for i, (w, d) in enumerate(zip(weight, dropout)):
        print()
        print('{} Weight: {} Dropout: {}'.format(i, w, d))
        model = ClassifySentence(dataset['TRAIN'].len_vocab(), EMBEDDING_DIM, 3, d)
        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=w)
        model_ft, acc = train_model(model, loss_function, optimizer, exp_lr_scheduler, 25)
        if acc > best_acc:
            best_acc = acc
            best_w = w
            best_d = d
    print('best_w: {:.4f} best_d: {:.4f} best_acc: {:.4f}'.format(
        best_w, best_d, best_acc))

model = ClassifySentence(dataset['TRAIN'].len_vocab(), EMBEDDING_DIM)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1.287051167874816e-06)
model_ft, acc = train_model(model, loss_function, optimizer, exp_lr_scheduler, 5)