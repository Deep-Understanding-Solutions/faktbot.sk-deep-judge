from torch import nn
from src import config as cfg


class DeepJudge(nn.Module):
    """This model uses pooled output of BERT and then runs it through
    linear layers to generate two classes - one for realness and one
    for fakeness."""

    def __init__(self, roberta_model, dropout=.5):
        """Initialize model layers."""
        super(DeepJudge, self).__init__()

        self.roberta_model = roberta_model
        self.dropout_layer = nn.Dropout(dropout)

        self.linear_layer_1 = nn.Linear(cfg.bert_output_dim, 50)
        self.relu_layer_1 = nn.ReLU()

        self.linear_layer_2 = nn.Linear(50, 25)
        self.relu_layer_2 = nn.ReLU()

        self.linear_layer_3 = nn.Linear(25, 2)
        # Currently not used, but might be in the future.
        self.relu_layer_3 = nn.ReLU()

    def forward(self, x):
        """Execute the model architecture."""
        pooled_output = self.roberta_model(**x).pooler_output
        dropout_layer = self.dropout_layer(pooled_output)

        linear_layer_1 = self.linear_layer_1(dropout_layer)
        relu_layer_1 = self.relu_layer_1(linear_layer_1)

        linear_layer_2 = self.linear_layer_2(relu_layer_1)
        relu_layer_2 = self.relu_layer_2(linear_layer_2)

        final_layer = self.linear_layer_3(relu_layer_2)

        return final_layer