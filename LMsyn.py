import os
import numpy as np
import pandas as pd
import random
import torch
import json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
from dataclasses import dataclass


def get_col_dist(df, col):
    if df[col].dtype != "float":
        col_dist = df[col].value_counts(1).to_dict()
    else:
        col_dist = df[col].to_list()
    return col_dist



def txt_to_tab_data_conversion(txt, gen_df):
    cols = gen_df.columns.to_list()
    for t in txt:
        samples = t.split(",")
        td = dict.fromkeys(cols)
        for s in samples:
            vals = s.strip().split(" is ")
            if vals[0] in cols and not td[vals[0]]:
                try:
                    td[vals[0]] = [vals[1]]
                except IndexError:
                    pass
        gen_df = pd.concat([gen_df, pd.DataFrame(td)], ignore_index=True, axis=0)
    return gen_df


def token_to_txt_conversion(toks, tokenizer):
    txt_data = [tokenizer.decode(t) for t in toks]
    txt_data = [d.replace("<|endoftext|>", "") for d in txt_data]
    txt_data = [d.replace("\r", "") for d in txt_data]
    txt_data = [d.replace("\n", " ") for d in txt_data]
    return txt_data


def pad_toks(toks):
    max_len = len(max(toks, key=len))
    padded_toks = [pad_tok(t, max_len) for t in toks]
    return padded_toks


def pad_tok(x, length, pad_val=50256):
    padding = [pad_val] * (length - len(x))
    padded_token = padding + x
    return padded_token


class LMmodelStart:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_start_toks(self, n_samples):
        raise NotImplementedError("This method must be overridden by the subclasses")


class ContStart(LMmodelStart):
    def __init__(self, tokenizer, start_col, start_col_dist, decimal_places=5, noise=.01):
        super().__init__(tokenizer)
        self.start_col = start_col
        self.start_col_dist = start_col_dist
        self.decimal_places = decimal_places
        self.noise = noise

    def get_start_toks(self, n_samples):
        words = random.choices(self.start_col_dist, k=n_samples)
        txt = [self.start_col + " is " + format(s, f".{self.decimal_places}f") + "," for s in words]
        toks = pad_toks(self.tokenizer(txt)["input_ids"])
        return toks


class CatStart(LMmodelStart):
    def __init__(self, tokenizer, start_col, start_col_dist):
        super().__init__(tokenizer)
        self.start_col = start_col
        self.weights = list(start_col_dist.values())
        self.population = list(start_col_dist.keys())

    def get_start_toks(self, n_samples):
        words = random.choices(self.population, self.weights, k=n_samples)
        txt = [self.start_col + " is " + str(s) + "," for s in words]
        toks = pad_toks(self.tokenizer(txt)["input_ids"])
        return toks


class RandStart(LMmodelStart):
    def __init__(self, tokenizer, all_columns):
        super().__init__(tokenizer)
        self.all_columns = all_columns

    def get_start_toks(self, n_samples):
        words = random.choices(self.all_columns, k=n_samples)
        txt = [s + " is " for s in words]
        toks = pad_toks(self.tokenizer(txt)["input_ids"])
        return toks


class LMmodelDataset(Dataset):
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def _getitem(self, key, decoded=True, **kwargs):
        row = self._data.fast_slice(key, 1)
        shuffle_index = list(range(row.num_columns))
        random.shuffle(shuffle_index)
        shuffled_txt_parts = []
        for i in shuffle_index:
            column_name = row.column_names[i]
            column_value = str(row.columns[i].to_pylist()[0]).strip()
            shuffled_txt_parts.append("{} is {}".format(column_name, column_value))
        shuffled_txt = ", ".join(shuffled_txt_parts)
        tokenized_txt = self.tokenizer(shuffled_txt)
        return tokenized_txt

    def __getitems__(self, keys):
        if not isinstance(keys, list):
            return self._getitem(keys)
        return [self._getitem(key) for key in keys]


@dataclass
class LMmodelDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch




class LMmodelTrainer(Trainer):
    def _init_worker(self,_):
        working_seed = torch.initial_seed() % 2**32
        random.seed(working_seed)
        np.random.seed(working_seed)
        torch.manual_seed(working_seed)
        torch.cuda.manual_seed_all(working_seed)

    def get_train_dataloader(self):
      
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        train_sampler = self._get_train_sampler()
        return DataLoader(
            train_dataset,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=self._init_worker,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator
        )


class LMmodel:
    def __init__(self, directory_path="trainer_lmmodel", batch_size=8, epochs=100, **train_kwargs):
        
        self.directory_path = directory_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_hyperparameters = train_kwargs
        self.columns = None
        self.num_cols = None
        self.cond_col = None
        self.cond_col_dist = None
        self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-125M')

    
    def update_col_info(self, df):
        self.columns = df.columns.to_list()
        self.num_cols = df.select_dtypes(include=np.number).columns.to_list()

    def update_cond_info(self, df, cond_col=None):
        self.cond_col = cond_col if cond_col else df.columns[-1]
        self.cond_col_dist = get_col_dist(df, self.cond_col)


    def fit(self, data, cond_col=None):
        df = data
        self.update_col_info(df)
        self.update_cond_info(df, cond_col)
        lmmodel_ds = LMmodelDataset.from_pandas(df)
        lmmodel_ds.set_tokenizer(self.tokenizer)
        training_args = TrainingArguments(self.directory_path, per_device_train_batch_size=self.batch_size,num_train_epochs=self.epochs,  **self.train_hyperparameters)
        lmmodel_trainer = LMmodelTrainer(self.model, training_args, train_dataset=lmmodel_ds, tokenizer=self.tokenizer,data_collator=LMmodelDataCollator(self.tokenizer))

        lmmodel_trainer.train()
        return lmmodel_trainer

    
    def get_vocab_generator(self, start_col, start_col_dist):
        start_col = start_col if start_col else self.cond_col
        start_col_dist = start_col_dist if start_col_dist else self.cond_col_dist

        if isinstance(start_col_dist, list):
            obj=ContStart(self.tokenizer, start_col, start_col_dist)
            return obj
        elif isinstance(start_col_dist, dict):
            obj= CatStart(self.tokenizer, start_col, start_col_dist)
            return obj
        else:
            obj=RandStart(self.tokenizer, self.columns)

            return obj
    
    def sample(self, n_samples,cond_col="",cond=None, temp=0.7):
        self.model.to("cuda")
        
        lmmodel_vocab = self.get_vocab_generator(cond_col, cond)
        df = pd.DataFrame(columns=self.columns) 
        
        while n_samples > df.shape[0]:
            id=50256
            start_vocab = torch.tensor(lmmodel_vocab.get_start_toks(100)).to("cuda")
            toks = self.model.generate(input_ids=start_vocab, max_length=3000, do_sample=True, temperature=temp, pad_token_id=id)
            text_data = token_to_txt_conversion(toks, self.tokenizer)
            df = txt_to_tab_data_conversion(text_data, df)
            df[self.num_cols] = df[self.num_cols].apply(pd.to_numeric, errors='coerce')
            df.dropna(axis=0, how='any', inplace=True)
            df[self.num_cols] = df[self.num_cols].astype(float)

        df = df.reset_index(drop=True)
        return df.head(n_samples)


    def save(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        with open(path + "/config.json", "w") as f:
            attr = self.__dict__.copy()
            attr.pop("tokenizer")
            attr.pop("model")
            if isinstance(attr["cond_col_dist"], np.ndarray):
                attr["cond_col_dist"] = list(attr["cond_col_dist"])
            json.dump(attr, f)
        torch.save(self.model.state_dict(), path + "/model.pt")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    