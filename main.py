from train import inference
from model import EncoderRNN, DecoderRNN
from tokenizer import Tokenizer, SOS_TOKEN, EOS_TOKEN
import torch
from utils import shakkel
import json
import argparse
import numpy as np

def main(args):
    # load the tokenizer
    tokenizer = Tokenizer(character_level=True)
    tokenizer.load("pickels/tokenizer.pkl")

    # load the model
    device = torch.device("cpu")

    with open('pickels/params.json', 'r') as f:
        params = json.load(f)

    # choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['device'] = device

    encoder = EncoderRNN(params).to(device)
    decoder = DecoderRNN(params).to(device)
    encoder.load_state_dict(torch.load('pickels/encoder.pth'))
    decoder.load_state_dict(torch.load('pickels/decoder.pth'))

    # read from arguments the arabic text

    input_text = args.input
    input_encoded = tokenizer.encode(input_text)
    input_ids = np.zeros((1, params['max_input']), dtype=np.int32)
    input_ids[0, :len(input_encoded)] = input_encoded
    input_tensor = torch.LongTensor(input_ids).reshape(1, -1).to(device) # to make batch first
    output = inference(encoder, decoder, input_tensor)[0]
    # to remove SOS and EOS tokens
    output = output[output != SOS_TOKEN]
    output = output[output != EOS_TOKEN]
    # decode into strings
    harakat = tokenizer.decode(output.tolist(), is_harakat=True)
    print(shakkel(input_text, harakat))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="input arabic text")
    args = parser.parse_args()

    if args.input is None:
        print("Please provide an input text")
    else: 
        main(args)
