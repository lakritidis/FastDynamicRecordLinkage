import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BatchEncoding


class SentencePairDataset:
    def __init__(self, tokenizer, batch_size : int=16, max_sequence_length : int=128) -> None:
        """
        Initializes a SentencePairDataset object

        Args:
            tokenizer (transformers.BertTokenizer): Tokenizer object.
            batch_size (int, optional): Batch size. Defaults to 16.
            max_sequence_length (int): Maximum sequence length. Defaults to 128.
        """
        self._batch_size = batch_size
        self._max_seq_length = max_sequence_length
        self._tokenizer = tokenizer

    def tokenize_function(self, df : pd.DataFrame) -> BatchEncoding:
        """
        Tokenizes a dataframe with two columns: sentence and label
.
        Args:
            df (pd.DataFrame): The dataframe to tokenize.

        Returns:
            A BertTokenizer object.
        """
        return self._tokenizer(df['t1'].tolist(), df['t2'].tolist(),
                               padding=True, truncation=True, max_length=self._max_seq_length, return_tensors="pt")

    def create_data_loader(self, df : pd.DataFrame, sort_col : int=None, label_col : str=None) -> DataLoader:
        """
        Creates DataLoader object from an input DataFrame.

        Args:
            df (pd.DataFrame): The dataframe from which to create DataLoader
            sort_col:
            label_col:

        Returns:

        """
        if label_col is None:
            print("No label column was specified, aborting")
            exit()

        if sort_col is not None:
            df.sort_values(by=sort_col, inplace=True)

        label_tensor = torch.tensor(df[label_col].values)
        encodings = self.tokenize_function(df)
        dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], label_tensor)

        dataloader = DataLoader(dataset, batch_size=self._batch_size)

        return dataloader
