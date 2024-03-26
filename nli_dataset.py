from torch.utils.data import Dataset
from transformers import RobertaTokenizer
import pandas as pd


class NliDataset(Dataset):
    def __init__(self, csv_file, max_length=256):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        premise = self.data['premise'][idx]
        hypothesis = self.data['hypothesis'][idx]
        label = int(self.data['label'][idx])
        
        # encoding = self.tokenizer(premise, hypothesis, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        # Tokenize premise-hypothesis pairs
        encoded_premise = self.tokenizer(premise,
                                        hypothesis, 
                                         padding='max_length', 
                                         truncation=True, 
                                         max_length=self.max_length, 
                                         return_tensors='pt')

        premise_ids = encoded_premise['input_ids'].squeeze()
        premise_mask = encoded_premise['attention_mask'].squeeze()
        
        return {
            'input_ids': premise_ids,
            'attention_mask': premise_mask,
            'label': label
        }


