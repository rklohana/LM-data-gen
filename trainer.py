import numpy as nump
import torch
from torch.utils.data import DataLoader
from transformers import Trainer
from trl import SFTTrainer
import random


def initialize_worker(_):
    seed_value = torch.initial_seed() % 2 ** 32
    random.seed(seed_value)
    nump.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


class Trainer(SFTTrainer):

    def fetch_training_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("A training dataset is required for training.")

        collation_data = self.data_collator
        dataset_for_training = self.train_dataset
        sampler_for_training = self._get_train_sampler()

        return DataLoader(
            dataset_for_training,
            batch_size=self._train_batch_size,
            sampler=sampler_for_training,
            collate_fn=collation_data,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=initialize_worker,
        )
