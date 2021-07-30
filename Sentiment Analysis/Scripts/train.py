from dataloader import CreateDataset
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = CreateDataset('../input/')
train_iterator, valid_iterator, test_iterator = dataset.getData()
pretrained_embeddings = dataset.getEmbeddings()
pretrained_embeddings.to(device)

from Kimcnn import KimCNN
input_dim = dataset.lengthVocab()[0]
embedding_dim = 300
n_filters = 100
filters = [3,4,5]
output_dim = 2
model = KimCNN(input_dim, embedding_dim, n_filters, filters, output_dim, pretrained_embeddings)
model.to(device)

# uncomment below to use RNN/LSTM
'''
from rnn import *
input_dim = dataset.lengthVocab()[0]
embedding_dim = 300
hidden_dim = 256
output_dim = 2
num_layers = 2
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNN(input_dim, embedding_dim, hidden_dim, output_dim, pretrained_embeddings)
#model = LSTM(input_dim, embedding_dim, num_layers, hidden_dim, pretrained_embeddings, bidirectional = True)
model.to(device)
'''

import torch.optim as optim
import torch.nn as nn

optimizer = optim.SGD(model.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss()
model = model.to(device)
criterion = criterion.to(device)

import torch.nn.functional as F

def accuracy(preds, y):

    preds, ind= torch.max(F.softmax(preds, dim=-1), 1)
    correct = (ind == y).float()
    acc = correct.sum()/float(len(correct))
    return acc

import pyprind

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    bar = pyprind.ProgBar(len(iterator), bar_char='█')
    for batch in iterator:
        
        optimizer.zero_grad()
                
        predictions = model(batch.Text).squeeze(0)

        loss = criterion(predictions, batch.Label)

        acc = accuracy(predictions, batch.Label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        bar.update()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        #bar = pyprind.ProgBar(len(iterator), bar_char='█')
        for batch in iterator:

            predictions = model(batch.Text).squeeze(0)
            
            loss = criterion(predictions, batch.Label)
            
            acc = accuracy(predictions, batch.Label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            #bar.update()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

epochs = 20
best_acc = 0
for epoch in range(epochs):

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    if valid_acc > best_acc:
        torch.save(model.state_dict(), 'weights_kim_sentiment.pth')
    print(f'Epoch: {epoch+1} \t Train Loss: {train_loss:.3f}  \t Train Acc: {train_acc*100:.2f}% \nVal. Loss: {valid_loss:.3f} \t Val. Acc: {valid_acc*100:.2f}% ')


# TESTING
test_loss, test_acc = evaluate(model, test_iterator, criterion)
print(f'test. Loss: {test_loss:.3f} \t test. Acc: {test_acc*100:.2f}% ')


# Checking for a random comment 
def tokenize_en(sentence):
    return [tok.text for tok in dataset.spacy.tokenizer(sentence)]
review = 'It was a very interesting movie. One of the best movies I have ever seen. I would recodmend everyone to watch it'
a = tokenize_en(review)
inputs = [dataset.TEXT.vocab.stoi[word] for word in a]
inputs = torch.tensor(inputs)
inputs = inputs.unsqueeze(1)
inputs = inputs.to(device)
preds = model(inputs)
preds, ind= torch.max(F.softmax(preds, dim=-1), 1)
if (ind == 0 ):
    print('negative')
else:
    print('positive')