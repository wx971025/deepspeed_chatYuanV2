from torch.utils.data import Dataset
import torch
from transformers import T5Tokenizer
import pandas as pd
import os

class mydata(object):
    def __init__(self, config):
        self.data_dir = 'UsedData'
        self.config = config

    def _getdata(self, filename):  # 加载每行数据及其标签
        path = os.path.join(self.data_dir, filename)
        df = pd.read_csv(path)
        df = df.sample(frac=self.config.DATA_SIZE)
        df = df[[self.config.SOURCE_TEXT, self.config.TARGET_TEXT]]

        train_dataset = df.sample(frac=self.config.TRAIN_SIZE, random_state=self.config.SEED)
        val_dataset = df.drop(train_dataset.index).reset_index(drop=True)
        train_dataset = train_dataset.reset_index(drop=True)

        return train_dataset, val_dataset

    def load_data(self):  # 加载数据
        return self._getdata('belle2M.csv')

class DataSet(Dataset):
    """
    创建一个自定义的数据集，用于训练，必须包括两个字段：输入(如source_text)、输出（如target_text）
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(self, dataframe, config):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.config = config
        self.tokenizer = T5Tokenizer.from_pretrained(config.MODEL)
        self.data = dataframe
        self.source_len = config.MAX_SOURCE_TEXT_LENGTH
        self.summ_len = config.MAX_TARGET_TEXT_LENGTH
        self.target_text = self.data[config.TARGET_TEXT]
        self.source_text = self.data[config.SOURCE_TEXT]

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }