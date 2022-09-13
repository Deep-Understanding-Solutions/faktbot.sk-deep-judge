<p align="center">
  <img src="https://gcdnb.pbrd.co/images/adEnsuA1zDRa.png" align="center" width="200" style="border-radius: 20px;"/>
</p>

# faktbot.sk-deep-judge
This project implements model and training API for fake news detection in Slovak language by utilizing <strong>B</strong>ilinear <strong>E</strong>ncoder <strong>R</strong>epresentations <strong>F</strong>rom <strong>T</strong>ransformers (BERT).

## Code
This project is written in PyTorch (Python language).

## Dataset
Model will be trained on carefully collected and curated list of provably accurate and inaccurate  articles.

## Model architecture
Our model utilizes internal states produced by `gerulata/slovakbert` model (BERT architecture) and attempts to learn article truthfulness/realiability from them. Internal
states are passed into feedforward DNN that learns to classify text.

<p align="center"><img src="https://gcdnb.pbrd.co/images/nMTMAtce86yC.png"/></p>

## Model output
Model output is constructed by using Linear layer with 2 nodes, activated by ReLu to trim the values. This layer generates two arbitrary values (without metrics), which score given class to be the one of the article.

## Labels
Labels consist of 0s and 1s shaped (batch_size). Each label will be used to maximize label-th node value of the last linear layer. Therefore higher score on first node will mean that article belongs to class 0 and higher score on the 2nd will mean that article belongs to class 1.
