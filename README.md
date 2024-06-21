# Introduction

This is a character-level seq2seq LSTM diacritization model with the below archeticture.

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


# Usage

```
from train import inference
from model import EncoderRNN, DecoderRNN
from tokenizer import Tokenizer, SOS_TOKEN, EOS_TOKEN
import torch
from utils import shakkel
import json

# load the tokenizer
tokenizer = Tokenizer(character_level=True)
tokenizer.load("tokenizer.pkl")

# load params
with open('params.json', 'r') as f:
    params = json.load(f)

# choose device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params['device'] = device

# load model
encoder = EncoderRNN(params).to(device)
decoder = DecoderRNN(params).to(device)
encoder.load_state_dict(torch.load('encoder.pth'))
decoder.load_state_dict(torch.load('decoder.pth'))

# inference
input_text = "رسول"
input_tensor = tokenizer.encode(input_text)
input_tensor = torch.LongTensor(input_tensor).reshape(1, -1).to(device) # to make batch first
output = inference(encoder, decoder, input_tensor)[0]

# to remove SOS and EOS tokens
output = output[output != SOS_TOKEN]
output = output[output != EOS_TOKEN]

# decode into strings
harakat = tokenizer.decode(output.tolist(), is_harakat=True)

# merge harakat with input text
print(shakkel(input_text, harakat))

```
