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

<p align="center"><img width="600" src="https://gcdnb.pbrd.co/images/goKt1pev0sb4.png"/></p>

## Model output
Model output is constructed by using Linear layer with 2 nodes, activated by ReLu to trim the values. This layer generates two arbitrary values (without metrics), which score given class to be the one of the article.

## Labels
Labels consist of 0s and 1s shaped (batch_size). Each label will be used to maximize label-th node value of the last linear layer. Therefore higher score on first node will mean that article belongs to class 0 and higher score on the 2nd will mean that article belongs to class 1.

## Training
Model training is generally highly stable and converges fast towards the goal. Experiments proved batch sizes up to 12 to work the best along with learning rate ~10e-6. Learning rate is very important in achieving the goal, because larger values make training stagnate because of vanishing gradient.

Collab training url: https://colab.research.google.com/drive/193kVNejCzooPE_-bkueZ9cwRa5t1pVOr?usp=sharing
