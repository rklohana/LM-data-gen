import pandas as pd
import torch
from sklearn import preprocessing
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
import numpy as np

class DataLoader(Dataset):
    def __init__(self, dataframe, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.original_data = dataframe
        self.preprocessed_data = self.preprocess_dataframe(dataframe)

    def preprocess_dataframe(self, dataframe):
        # Remove null values
        dataframe = dataframe.dropna()

        # Remove duplicate rows
        dataframe = dataframe.drop_duplicates()

        # Normalize numeric columns
        dataframe = self.normalize_dataframe(dataframe)

        # Encode string columns
        dataframe = self.encode_dataframe(dataframe)

        # Tokenize text columns
        text_columns = dataframe.select_dtypes(include=['object']).columns.tolist()
        dataframe = self.tokenize_dataframe(dataframe, text_columns)
        return dataframe

    def normalize_dataframe(self, dataframe):
        numeric_columns = dataframe.select_dtypes(include=['int64', 'float64']).columns
        dataframe[numeric_columns] = preprocessing.MinMaxScaler().fit_transform(dataframe[numeric_columns])
        return dataframe

    def encode_dataframe(self, dataframe):
        categorical_columns = dataframe.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            if dataframe[column].nunique() < 10:  # or some other threshold
                dataframe[column] = preprocessing.LabelEncoder().fit_transform(dataframe[column])
        return dataframe

    def tokenize_dataframe(self, dataframe, text_columns):
        for column in text_columns:
            texts = dataframe[column].tolist()
            encodings = self.tokenizer(texts, truncation=True, padding=True)
            dataframe[column + '_input_ids'] = [ids for ids in encodings['input_ids']]
            dataframe[column + '_attention_mask'] = [mask for mask in encodings['attention_mask']]
        dataframe = dataframe.drop(columns=text_columns)
        return dataframe

    def __len__(self):
        return len(self.preprocessed_data)

    def __getitem__(self, idx):
        item = {}
        for column in self.preprocessed_data.columns:
            if column.endswith('_input_ids') or column.endswith('_attention_mask'):
                item[column] = torch.tensor(self.preprocessed_data[column].iloc[idx], dtype=torch.long)
            else:
                item[column] = torch.tensor(self.preprocessed_data[column].iloc[idx])
        return item
