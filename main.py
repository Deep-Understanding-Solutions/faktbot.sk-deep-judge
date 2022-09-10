from transformers import RobertaTokenizer, RobertaModel
from torch import nn
from torch import tensor
from torch.optim import Adam
import pandas as pd

# System constants.
article_csv_selector = "text"
title_csv_selector = "title"
bert_output_dim = 768

# Parse csv data.
csv_data = pd.read_csv("train.csv")
articles = csv_data[article_csv_selector].values.tolist()

# Get ready BERT and encode input data.
tokenizer = RobertaTokenizer.from_pretrained("gerulata/slovakbert")
roberta_model = RobertaModel.from_pretrained("gerulata/slovakbert")
encoded_articles = tokenizer(articles, return_tensors="pt", padding=True, truncation=True)


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
    output = model(encoded_articles)
    loss = criterion(output, tensor([[0., 1.], [0., 1.]]))
    model.zero_grad()
    loss.backward()
    optimizer.step()


deepJudgeModel = DeepJudge()
train(deepJudgeModel, None, None, .1, None)