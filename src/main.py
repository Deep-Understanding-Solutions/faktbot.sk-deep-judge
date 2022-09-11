from transformers import RobertaTokenizer, RobertaModel
from torch import nn
from torch import tensor
from torch.optim import Adam
import torch
import pandas as pd
import numpy as np
import config as cfg
from model.DeepJudge import DeepJudge
import random

# Parse csv data.
csv_data = pd.read_csv("data/train.csv")
csv_data = csv_data.dropna().head(cfg.csv_rows_limit)

articles = csv_data[cfg.article_csv_selector].values
articles = np.reshape(articles, (-1, cfg.batch_size)).tolist()
tokenized_articles = np.empty((np.shape(articles)[0]), dtype=object)

labels = csv_data[cfg.label_csv_selector].values
labels = np.reshape(labels, (-1, cfg.batch_size))

# Get ready BERT and encode input data.
tokenizer = RobertaTokenizer.from_pretrained("gerulata/slovakbert")
roberta_model = RobertaModel.from_pretrained("gerulata/slovakbert")

# Tokenize article batches by iterating over them.
for batch_idx, articles_batch in enumerate(articles):
    tokenized_articles_batch = tokenizer(articles[batch_idx], return_tensors="pt", padding=True, truncation=True)
    tokenized_articles[batch_idx] = tokenized_articles_batch

tokenized_articles = tokenized_articles.tolist()
labels = labels.tolist()


def train(model, xs, ys, learning_rate, epochs):
    """Method invoking main training cycle."""
    # We will use categorical cross-entropy with Adam activation.
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # This is for demonstrational purposes, it needs to be integrated to DataLoader and stuff.
    for epoch in range(epochs):
        for batch_index in range(len(tokenized_articles)):
            x = xs[batch_index]
            y = ys[batch_index]

            output = model(x)

            loss = criterion(output, tensor(y))
            print(f"Epoch: {epoch + 1}/{epochs}, batch: {batch_index + 1}/{len(tokenized_articles)}, loss: {loss}")
            model.zero_grad()
            loss.backward()
            optimizer.step()


deep_judge = DeepJudge(roberta_model=roberta_model)
train(deep_judge, tokenized_articles, labels, cfg.learning_rate, cfg.epochs)
torch.save(deep_judge.state_dict(), "trained_model/deep_judge.pth")
