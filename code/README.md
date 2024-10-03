# VQA
Pytorch implementation and improvement of the paper - VQA: Visual Question Answering (https://arxiv.org/pdf/1505.00468.pdf).


## Usage 

#### 1. Dataset
Since the image dataset is too big, we deleted them when we uploaded the project. If you want to see the dataset, please visit the website https://visualqa.org/download.html.


#### 2. Models
We did several different experiments and test their performance. "model" is the original one, using VGG-19 and LSTM, which serves as the base line. "model_2" replaces VGG-19 by ResNet-18. "model_3" uses Resnet-18 and BERT. "model_4" adds Attention mechanism in image encoder based on "model_2".


