<p align="center">
  <img src="https://i.postimg.cc/d06RCLJW/deep-judge-modified.png" align="center" width="200" style="border-radius: 20px;"/>
</p>

# faktbot.sk-deep-judge
This project implements model and training API for fake news
detection in Slovak language.

## Code
This project is written in PyTorch (Python language).

## Dataset
Model will be trained on carefully collected and curated list of provably accurate and inaccurate  articles.

## Model architecture
Our model utilizes internal states produced by `gerulata/slovakbert` model (RoBERTa architecture) and attempts to learn article sentiment from them. Internal
states are passed into feedforward DNN that learns to classify text.
