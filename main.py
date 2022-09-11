from transformers import RobertaTokenizer, RobertaModel
from torch import nn
from torch import tensor
from torch.optim import Adam
import pandas as pd
import numpy as np

# System constants.
batch_size = 2
num_batches = 2
article_csv_selector = "text"
label_csv_selector = "label"
title_csv_selector = "title"
bert_output_dim = 768
csv_rows_limit = batch_size * num_batches

# Parse csv data.
csv_data = pd.read_csv("train.csv")
csv_data = csv_data.dropna().head(csv_rows_limit)

articles = csv_data[article_csv_selector].values.tolist()
articles = np.reshape(articles, (-1, batch_size)).tolist()
tokenized_articles = np.empty((np.shape(articles)[0]), dtype=object)

labels = csv_data[label_csv_selector].values.tolist()
labels = list(map(lambda value: [0., 1.] if str(1) else [1., 0.], labels))
labels = np.reshape(labels, (-1, batch_size, 2))

# Get ready BERT and encode input data.
tokenizer = RobertaTokenizer.from_pretrained("gerulata/slovakbert")
roberta_model = RobertaModel.from_pretrained("gerulata/slovakbert")

# Tokenize article batches by iterating over them.
for batch_idx, articles_batch in enumerate(articles):
    tokenized_articles_batch = tokenizer(articles[batch_idx], return_tensors="pt", padding=True, truncation=True)
    tokenized_articles[batch_idx] = tokenized_articles_batch

tokenized_articles = tokenized_articles.tolist()
labels = labels.tolist()

class DeepJudge(nn.Module):
    """This model uses pooled output of BERT and then runs it through
    linear layers to generate two classes - one for realness and one
    for fakeness."""

    def __init__(self, dropout=.5):
        """Initialize model layers."""
        super(DeepJudge, self).__init__()

        self.roberta_model = roberta_model
        self.dropout_layer = nn.Dropout(dropout)

        self.linear_layer_1 = nn.Linear(bert_output_dim, 50)
        self.relu_layer_1 = nn.ReLU()

        self.linear_layer_2 = nn.Linear(50, 25)
        self.relu_layer_2 = nn.ReLU()

        self.linear_layer_3 = nn.Linear(25, 2)
        # Currently not used, but might be in the future.
        self.relu_layer_3 = nn.ReLU()

    def forward(self, x):
        """Execute the model architecture."""
        pooled_output = roberta_model(**x).pooler_output
        dropout_layer = self.dropout_layer(pooled_output)

        linear_layer_1 = self.linear_layer_1(dropout_layer)
        relu_layer_1 = self.relu_layer_1(linear_layer_1)

        linear_layer_2 = self.linear_layer_2(relu_layer_1)
        relu_layer_2 = self.relu_layer_2(linear_layer_2)

        final_layer = self.linear_layer_3(relu_layer_2)

        return final_layer


def train(model, train_data, val_data, learning_rate, epochs):
    """Method invoking main training cycle."""
    # We will use categorical cross-entropy with Adam activation.
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # This is for demonstrational purposes, it needs to be integrated to DataLoader and stuff.
    # But it works.
    # @TODO
    articles_batch = tokenized_articles[0]
    labels_batch = labels[0]
    output = model(articles_batch)

    loss = criterion(output, tensor(labels_batch))
    model.zero_grad()
    loss.backward()
    optimizer.step()


deepJudgeModel = DeepJudge()
train(deepJudgeModel, None, None, .1, None)
