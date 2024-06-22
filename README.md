# Introduction

This is a character-level seq2seq LSTM diacritization model with the below archeticture.

```
EncoderRNN(
  (embedding): Embedding(55, 128)
  (lstm): LSTM(128, 64, num_layers=2, batch_first=True, dropout=0.1, bidirectional=True)
  (dropout): Dropout(p=0.1, inplace=False)
)
DecoderRNN(
  (embedding): Embedding(13, 128)
  (lstm): LSTM(128, 64, batch_first=True)
  (out): Linear(in_features=64, out_features=13, bias=True)
)
```

# Installation


- Create virtual env:

```
virtualenv -p $(which python3) .tashkeel
source .tashkeel/bin/activate 
pip install -r requirements.txt
```

- Inference:
