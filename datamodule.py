# extends Lightning Data Module
import numpy as np
from json import decoder
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from kobert_transformers import get_tokenizer

class MyDataModule(pl.LightningDataModule):
    def __init__(self, 
                 train_file,
                 test_file, 
                 max_len=512,
                 batch_size=10,
                 num_workers=5):
        super().__init__()
        self.batch_size = batch_size
        self.max_len = max_len
        self.train_file_path = train_file
        self.test_file_path = test_file
        self.num_workers = num_workers

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        # split dataset
        self.trainDataset = MyDataset(self.train_file_path,
                                 self.max_len,
                                 pad_index=1)
        self.testDataset = MyDataset(self.test_file_path,
                                self.max_len,
                                pad_index=1)

    def train_dataloader(self):
        train = DataLoader(self.trainDataset,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers, 
                           shuffle=True)
        return train

    def val_dataloader(self):
        val = DataLoader(self.testDataset,
                         batch_size=self.batch_size,
                         num_workers=self.num_workers, 
                         shuffle=False)
        return val

    def test_dataloader(self):
        test = DataLoader(self.testDataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, 
                          shuffle=False)
        return test



class MyDataset(Dataset):
    def __init__(self, file, max_len, pad_index = 1, ignore_index=1):
        super().__init__()
        self.max_len = max_len
        self.docs = pd.read_csv(file, sep='\t')
        self.len = self.docs.shape[0]
        self.pad_index = pad_index
        self.ignore_index = ignore_index
        self.tokenizer = get_tokenizer()

    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = []
            for i in range(self.max_len - len(inputs)):
                pad.append(self.pad_index)
            inputs = inputs + pad
        else:
            inputs = inputs[:self.max_len]

        return inputs
    
    def __getitem__(self, idx):
        instance = self.docs.iloc[idx]

        input_ids = self.tokenizer.encode(instance['decision'], max_length=self.max_len, truncation=True)
        input_ids = self.add_padding_data(input_ids)

        label_ids = self.tokenizer.encode(instance['issue'], max_length=self.max_len, truncation=True)
        label_ids = self.add_padding_data(label_ids)
        
        return {'input_ids': np.array(input_ids, dtype=np.int_), 
                'labels': np.array(label_ids, dtype=np.int_)}
    
    def __len__(self):
        return self.len
