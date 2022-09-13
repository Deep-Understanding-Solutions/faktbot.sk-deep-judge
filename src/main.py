from transformers import RobertaTokenizer, RobertaModel
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import torch
import pandas as pd
import config as cfg
from model.DeepJudge import DeepJudge
from data.Dataset import Dataset

# Parse csv data.
csv_data = pd.read_csv("data/train.csv")
csv_data = csv_data.dropna().head(cfg.csv_rows_limit)

# Convert x and ys to list.
articles = csv_data[cfg.article_csv_selector].values
labels = csv_data[cfg.label_csv_selector].values

# Get ready BERT and encode input data.
tokenizer = RobertaTokenizer.from_pretrained("gerulata/slovakbert")
roberta_model = RobertaModel.from_pretrained("gerulata/slovakbert")

# Create the dataset and the data loader.
train = Dataset(articles, labels, tokenizer)
train_data_loader = torch.utils.data.DataLoader(train, batch_size=cfg.batch_size, shuffle=True)


def evaluateConfidence(predicted_value):
    """
    Evaluate confidence of prediction.
    :param predicted_value: Tensor predicted by DeepJudge model.
    :return: Tuple (predicted_result_index, predicted_result_confidence).
    """
    max_index = torch.argmax(predicted_value)
    min_index = 0 if max_index == 1 else 1

    confidence = torch.tensor(100) if predicted_value[min_index] == 0 else predicted_value[max_index] / predicted_value[min_index]
    return max_index.item(), confidence.item()


def train(model, learning_rate, epochs):
    """
    Method invoking main training cycle.
    """
    # We will use categorical cross-entropy with Adam activation.
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # This is for demonstrational purposes, it needs to be integrated to DataLoader and stuff.
    for epoch in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_data_loader):
            # [batch_size, 1, seq_length] - remove the 1.
            input_ids = train_input.input_ids.squeeze(1)
            input_masks = train_input.attention_mask

            output = model(input_ids, input_masks)

            loss = criterion(output, train_label)

            total_loss_train += loss.item()
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epochs: {epoch + 1}, Train Loss: {total_loss_train / len(labels)}, Train Accuracy: {total_acc_train/len(labels)}')


deep_judge = DeepJudge(roberta_model=roberta_model)
train(deep_judge, cfg.learning_rate, cfg.epochs)
torch.save(deep_judge.state_dict(), f"trained_model/{cfg.model_name}.pth")
