import numpy as np
import pandas as pd
import torch
import torchtext
import spacy

class CreateDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, batch_size=32):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.spacy = spacy.load("en_core_web_sm")

        self.TEXT = torchtext.data.Field(sequential=True, tokenize="spacy")
        self.LABEL = torchtext.data.LabelField(dtype=torch.long, sequential=False)

        self.initData()
        self.initEmbed()

        self.makeData()

    def initData(self):
        
        df_path = self.root_dir + 'imdb-dataset-sentiment-analysis-in-csv-format'

        self.train_data, self.valid_data, self.test_data = torchtext.data.TabularDataset.splits(
                        path=df_path, 
                        train="Train.csv", validation="Valid.csv", test="Test.csv", 
                        format="csv", 
                        skip_header=True, 
                        fields=[('Text', self.TEXT), ('Label', self.LABEL)])

    def initEmbed(self):
        
        embed_path = self.root_dir + 'glove6b300dtxt/glove.6B.300d.txt'

        self.TEXT.build_vocab(self.train_data,
                         vectors=torchtext.vocab.Vectors(embed_path), 
                         max_size=20000, 
                         min_freq=10)
        self.LABEL.build_vocab(self.train_data)

    def makeData(self):
        self.train_iterator, self.valid_iterator, self.test_iterator = torchtext.data.BucketIterator.splits(
                        (self.train_data, self.valid_data, self.test_data), 
                        sort_key=lambda x: len(x.Text), 
                        batch_size=self.batch_size,
                        device=self.device)

    def lengthData(self):
        return len(self.train_data), len(self.valid_data), len(self.test_data)
    
    def lengthVocab(self):
        return len(self.TEXT.vocab), len(self.LABEL.vocab)

    def freqLABEL(self):
        return self.LABEL.vocab.freqs

    def getData(self):
        return self.train_iterator, self.valid_iterator, self.test_iterator

    def getEmbeddings(self):
        return self.TEXT.vocab.vectors