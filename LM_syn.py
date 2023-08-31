"""
This python script contains several utility functions and classes to generate synthetic data using the GPT-NEO model. The script has been developed to be utilized in a way that resembles scikit-learn's fit-transform methodology. 

Before diving into the details of the functions and classes, let's understand the workflow:

1. Preprocess your data with the `preprocess` method, which converts categorical columns into numerical format using label encoding.
2. Fit your preprocessed data to the `LMmodel` instance using the `fit` method. This trains the language model to understand the underlying data structure.
3. Generate synthetic samples using the `sample` method.

Now let's go into details:

`get_col_dist`: Given a dataframe and a column name, this function returns a dictionary where the keys are unique values in the column and the values are their corresponding frequencies. If the column is of type float, the function returns the column as a list.

`stringtotab`: This function converts the generated tokenized text back into a tabular format. 

`removeextratokens`: This function removes rows from the generated dataframe that contain extra unneccessary tokens.

`postprocess`: This function postprocesses the generated tokenized text. It converts the tokenized text into a dataframe, converts the numerical columns into numeric datatype, and removes rows with any missing values.

`is_cat`: This function checks if a column in the dataframe is categorical or not.

`pad_toks` and `pad_tok`: These functions add padding to the tokenized text so that all sequences are of the same length.

`LMmodelStart`, `ContStart`, `CatStart`, `RandStart`: These are classes for generating the start tokens for the GPT-NEO model based on the starting column and its distribution. The `LMmodelStart` class is the parent class, and the other three are child classes for different scenarios.

`LMmodelDataset`: This class represents the dataset object for the language model. It shuffles the data and tokenizes it.

`LMmodelDataCollator`: This class collates the data into a format suitable for feeding into the language model.

`LMmodelTrainer`: This class is used to train the language model on the dataset. It inherits from the `Trainer` class in the `transformers` library.

`LMmodel`: This is the main class that contains methods for training the language model and generating synthetic data. It also contains methods for saving and loading the model. 

Please make sure to have the required packages installed and the GPT-NEO model available in the specified path.

"""
import os
import numpy as np
import pandas as pd
import random
import torch
import json
import pickle
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, GPTNeoConfig,DataCollatorWithPadding, GPTNeoForCausalLM
from datasets import Dataset
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder as le

def get_col_dist(df, col):
    if df[col].dtype != "float":
        col_dist = df[col].value_counts(1).to_dict()
    else:
        col_dist = df[col].to_list()
    return col_dist





def stringtotab(toks, tokenizer,gen_df):
    txt_data = [tokenizer.decode(t) for t in toks]
    txt_data = [d.replace("<|endoftext|>", "") for d in txt_data]
    txt_data = [d.replace("\r", "") for d in txt_data]
    txt_data = [d.replace("\n", " ") for d in txt_data]
    
    cols = gen_df.columns.to_list()
    for t in txt_data:
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
def removeextratokens(gen_df,orig_df,cols):
    
    uncom={}
    for column in cols:
        uncom[column]=[]
        for val in gen_df[column].unique():
            if val not in orig_df[column].unique():
                uncom[column].append(val)
    temp=gen_df.copy()
    for i in uncom.keys():
        if len(uncom[i])>0:
            for j in uncom[i]:
                temp = temp.drop(temp[temp[i] == j].index)
    print(uncom)
    return temp

def postprocess(out,df,orig,num_cols,tokenizer,enc):
    df = stringtotab(out, tokenizer,df)
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(axis=0, how='any', inplace=True)
    df[num_cols] = df[num_cols].astype(float)
  
    return df

def is_cat(column):
    categorical_dtypes = ['object', 'category', 'bool']
    if column.dtype.name in categorical_dtypes:
        return True
    else:
        return False   

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
        # print(txt)
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
    def __init__(self, directory_path="trainer_lmmodel", **train_kwargs):
        
        self.directory_path = directory_path
        self.train_hyperparameters = train_kwargs
        self.columns = None
        self.num_cols = None
        self.cond_col = None
        self.cond_col_dist = None
        self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-125M')
        self.label_encoders={}
        self.condflag=False


    def preprocess(self,df):
        temp=df.copy()
        for i in df.columns:
            if  is_cat(df[i]):
                enc=le()
                temp[i]=enc.fit_transform(df[i])
                self.label_encoders[i]=enc
        return temp
                  
    

    def fit(self, data,batch_size=8,epoch=100, cond_col=None):
        if cond_col:
            self.condflag=True
        df = self.preprocess( data)
        self.data=df.copy()
        self.columns = df.columns.to_list()
        self.num_cols = df.select_dtypes(include=np.number).columns.to_list()
        self.cond_col = cond_col if cond_col else df.columns[-1]
        self.cond_col_dist = get_col_dist(df, self.cond_col)

    
        lmmodel_ds = LMmodelDataset.from_pandas(df)
        lmmodel_ds.set_tokenizer(self.tokenizer)
        training_args = TrainingArguments(self.directory_path, per_device_train_batch_size=batch_size,num_train_epochs=epoch, optim="adafactor", fp16=True,gradient_accumulation_steps=4,  gradient_checkpointing=True,**self.train_hyperparameters)
        lmmodel_trainer = LMmodelTrainer(self.model, training_args, train_dataset=lmmodel_ds, tokenizer=self.tokenizer,data_collator=LMmodelDataCollator(self.tokenizer))

        lmmodel_trainer.train()
        return lmmodel_trainer

    def set_metadata(self, data, cond_col=None):
        if cond_col:
            self.condflag=True
        df = self.preprocess(data)
        self.data=df.copy()
        self.columns = df.columns.to_list()
        self.num_cols = df.select_dtypes(include=np.number).columns.to_list()
        self.cond_col = cond_col if cond_col else df.columns[-1]
        self.cond_col_dist = get_col_dist(df, self.cond_col)

    
    
    def get_vocab_generator(self, start_col, start_col_dist):
        start_col = start_col if start_col else self.cond_col
        start_col_dist = start_col_dist if start_col_dist else self.cond_col_dist

        if isinstance(start_col_dist, list):
            obj=ContStart(self.tokenizer, start_col, start_col_dist)
            return obj
        elif isinstance(start_col_dist, dict):
            if self.condflag:
                temp={}
                for i in start_col_dist.keys():
                    temp[self.label_encoders[start_col].transform([i])]=start_col_dist[i]
                start_col_dist=temp
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
            
            out = self.model.generate(input_ids=torch.tensor(lmmodel_vocab.get_start_toks(100)).to("cuda"), max_length=1000, do_sample=True, temperature=temp, pad_token_id=50256)
            df  = postprocess(out,df,self.data,self.num_cols,self.tokenizer,self.label_encoders)
            print("generated: ",df.shape[0],"/",n_samples)
            if df.shape[0]>=n_samples:
                df=removeextratokens(df,self.data,self.label_encoders.keys())
        for i in self.label_encoders.keys():
            df[i]=self.label_encoders[i].inverse_transform(pd.to_numeric(df[i], downcast='integer'))
        df = df.reset_index(drop=True)
        return df.head(n_samples)


    def save(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        # with open(path + "/config.json", "w") as f:
        #     attr = self.__dict__.copy()
        #     attr.pop("tokenizer")
        #     attr.pop("model")
        #     if isinstance(attr["cond_col_dist"], np.ndarray):
        #         attr["cond_col_dist"] = list(attr["cond_col_dist"])
        #     json.dump(attr, f)
        torch.save(self.model.state_dict(), path + "/model.pt")
        # output = open(path+'/label_encoder.pkl', 'wb')
        # pickle.dump(self.label_encoders, output)
        # output.close()

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        # pkl_file = open(path+'/label_encoder.pkl', 'rb')
        # self.label_encoders = pickle.load(pkl_file) 
        # pkl_file.close()


    