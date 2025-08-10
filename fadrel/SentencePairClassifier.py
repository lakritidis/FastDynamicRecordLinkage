import pandas as pd

import torch

from tqdm import tqdm

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_scheduler
from transformers import logging

from SentencePairDataset import SentencePairDataset

logging.set_verbosity_error()


class SentencePairClassifier:
    def __init__(self, model_name, model_path : str, max_emb_length : int, batch_size : int, epochs : int) -> None:
        """
        Initialize the SentencePair Classifier of FaDReL. Typically, this is a ``BertForSequenceClassification`` model.

        Args:
            model_name (str): The name of the classification model.
            model_path (str): The path to save the fine-tuned model.
            max_emb_length (int): The maximum embedding length of the BERT Classifier model.
            batch_size (int): The batch size to be used during training.
            epochs (int): The number of training epochs
        """
        self._model_name = model_name

        self.tokenizer = BertTokenizer.from_pretrained(self._model_name, do_lower_case=True)
        self.model = BertForSequenceClassification.from_pretrained(self._model_name, num_labels=2)

        self.model_path = model_path
        self._max_length = max_emb_length
        self._batch_size = batch_size
        self._epochs = epochs
        self._learning_rate = 2e-5

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self._device)

    def train(self, training_pairs: pd.DataFrame) -> None:
        """
        Train the SentencePair Classifier.

        Args:
            training_pairs (pd.DataFrame): A dataframe containing the training pairs.
        """

        train_dataset = SentencePairDataset(self.tokenizer, self._batch_size, self._max_length)
        train_dataloader = train_dataset.create_data_loader(training_pairs, sort_col=None, label_col='y')

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self._learning_rate)

        num_training_steps = self._epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        self.model.train()
        for epoch in range(self._epochs):
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
            for batch in progress_bar:
                # Move batch to device (GPU or CPU)
                input_ids, attention_mask, labels = [b.to(self._device) for b in batch]

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                # Backward pass
                loss.backward()

                # Update model parameters
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.set_postfix(loss=loss.item())

        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)

        # Clear the GPU cache and reload the model from checkpoint. In this way, only the model resides in VideoRAM
        if self._device == 'cuda':
            with torch.no_grad():
                torch.cuda.empty_cache()

        print("Model Training Completed")
