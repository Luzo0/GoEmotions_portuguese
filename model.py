import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import pytorch_lightning as pl
from utils import class_balanced_loss
from torch.utils.data import DataLoader


class EmotionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_token_len
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        text = data_row['text']
        labels = data_row['labels']
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return dict(
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.FloatTensor(labels)
        )


class EmotionDataModule(pl.LightningDataModule):
    def __init__(self, df_train, df_validation, df_test, tokenizer, batch_size, max_token_len):
        super().__init__()
        self.batch_size = batch_size
        self.df_train = df_train
        self.df_validation = df_validation
        self.df_test = df_test
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def setup(self, stage=None):
        self.train_dataset = EmotionDataset(
            self.df_train,
            self.tokenizer,
            self.max_token_len
        )
        self.validation_dataset = EmotionDataset(
            self.df_validation,
            self.tokenizer,
            self.max_token_len
        )
        self.test_dataset = EmotionDataset(
            self.df_test,
            self.tokenizer,
            self.max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=torch.get_num_threads()
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            num_workers=torch.get_num_threads()
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=torch.get_num_threads()
        )


class EmotionTagger(pl.LightningModule):
    def __init__(
            self,
            n_classes,
            model_name,
            samples_per_classes,
            beta,
            n_training_steps,
            n_warmup_steps,
            no_cuda
    ):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_classes)
        self.samples_per_classes = samples_per_classes
        self.beta = beta
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.n_classes = n_classes
        self.no_cuda = no_cuda

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        logits = output.logits
        loss = 0
        if labels is not None:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                input=logits,
                target=labels,
                weight=class_balanced_loss(
                    n_classes=self.n_classes,
                    samples_per_classes=self.samples_per_classes,
                    b_labels=labels,
                    beta=self.beta,
                    no_cuda=self.no_cuda
                )
            )
        return loss, logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, logits = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss, "predictions": logits, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, logits = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, logits = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps
        )
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )
