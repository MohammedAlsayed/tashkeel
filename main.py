from model import EncoderRNN, DecoderRNN
from tokenizer import Tokenizer, SOS_TOKEN, EOS_TOKEN
import torch
from utils import *
import json
import argparse
import numpy as np
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import math
from train import *
from tokenizer import Tokenizer
from sklearn.metrics import classification_report
import sys
import json
import pickle

def predict(args):
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
    
    if len(input_encoded) > params['max_input']:
        input_ids[0, :params['max_input']] = input_encoded
        print("Input text longer then {}, input will be trimmed.".format(params['max_input']))
    else:
        input_ids[0, :len(input_encoded)] = input_encoded

    input_tensor = torch.LongTensor(input_ids).reshape(1, -1).to(device) # to make batch first
    output = inference(encoder, decoder, input_tensor)[0]
    # to remove SOS and EOS tokens
    output = output[output != SOS_TOKEN]
    output = output[output != EOS_TOKEN]
    # decode into strings
    harakat = tokenizer.decode(output.tolist(), is_harakat=True)
    print(shakkel(input_text, harakat))


# Calculates the length of words in a text (tokens)
def custom_length(text: list[str]) -> int:
    return len(text.split())

def main(args):
    print("reading data...")
    files = get_file_list('data/texts.txt/')
    keys = list(files.keys())
    # read files into Document objects
    docs = []
    # for i in range(len(files[keys[0]])):
    for i in tqdm(range(1)): # reading a sample because of limited computational resources
        with open(files[keys[0]][i], 'r') as f:
            lines = f.readlines()
            lines = ' '.join(lines)
            file = files[keys[0]][i].split('/')[-1]
            metadata = {"source": file}
            doc = Document(page_content=lines, metadata=metadata)
            docs.append(doc)

    train_len = math.ceil(len(docs)*0.8)
    valid_len = math.floor(len(docs)*0.1)
    test_len = math.ceil(len(docs)*0.1)

    # Split the documents into chunks of words
    print("splitting docs into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1,
                chunk_overlap=0,
                length_function=custom_length,
                separators=["\n\n", "\n", ".", " "])

    splitted_docs = splitter.split_documents(docs)

    print("cleaning the data ...")
    splitted_docs = clean_data(splitted_docs)

    train_len = math.ceil(len(splitted_docs)*0.8) # 80% training
    valid_len = math.floor(len(splitted_docs)*0.1) # 10% validation
    test_len = math.ceil(len(splitted_docs)*0.1) # 10% testing

    training_docs = splitted_docs[:train_len]
    validation_docs = splitted_docs[train_len:train_len+valid_len]
    testing_docs = splitted_docs[train_len+valid_len:]

    # # save Document objects to disk
    # with open('pickels/training_docs.pkl', 'wb') as f:
    #     pickle.dump(training_docs, f)
    # with open('pickels/validation_docs.pkl', 'wb') as f:
    #     pickle.dump(validation_docs, f)
    # with open('pickels/testing_docs.pkl', 'wb') as f:
    #     pickle.dump(testing_docs, f)

    print(f"training_docs: {len(training_docs)}, validation_docs: {len(validation_docs)}, testing_docs: {len(testing_docs)}")
                    

    print("building tokenizer")
    tokenizer = Tokenizer(character_level=True)
    tokenizer.build_tokenizer_table(training_docs)

    print("preparing train data")
    train_pairs, max_input, max_target, avg_input, avg_target = prepare_data(training_docs[:20000], tokenizer)

    avg_target = int(math.ceil(avg_target))
    avg_input = int(math.ceil(avg_input))

    print("preparing validation data")
    validate_pairs, _, _, _, _ = prepare_data(validation_docs[:10000], tokenizer)

    print(f"input token length: {max_input}, target output length: {max_target}, average input length: {avg_input}, average target length: {avg_target}")
    device = torch.device("cuda")
    
    print(f"device being used: {device}")

    print("loading data into dataloaders")
    train_dataloader = get_dataloader(train_pairs, max_input, max_target, 32, device)
    validate_dataloader = get_dataloader(validate_pairs, max_input, max_target, 32, device)

    params = {
    'input_size': tokenizer.get_input_size(), # num of (character or word) tokens
    'embedding_size': 256, # size of embedding
    'hidden_size': 128, # size of embedding
    'output_size': tokenizer.get_output_size(), # harakat size
    'enc_n_layers': 3, # num of layers in the encoder
    'dec_n_layers': 1, # num of layers in the decoder
    'dropout': 0.1,
    'max_length': max_target,  # max length of output sequence (harakat sequence)
    'device': device,
    'SOS_TOKEN': SOS_TOKEN, # first input to the decoder
    'max_input': max_input, # for inference use
    }

    tokenizer.save("pickels/tokenizer.pkl")
    # dump the params
    with open('pickels/params.json', 'w') as f:
        if 'device' in params:
            params.pop('device')
        json.dump(params, f)
    params['device'] = device

    encoder = EncoderRNN(params).to(device)
    decoder = DecoderRNN(params).to(device)

    print(f"Start training with the following params: {params}")
    train(train_dataloader, validate_dataloader, encoder, decoder, args.epochs, print_every=5)
    
    print("load best model and test on validation data")

    encoder.load_state_dict(torch.load('pickels/encoder.pth'))
    decoder.load_state_dict(torch.load('pickels/decoder.pth'))
    equal_list, y_true, y_pred = test_result(validate_dataloader, encoder, decoder)
    # iterate over the first 12 keys in the tokenizer table to get the labels
    labels = [get_name(k) for k in list(tokenizer.token2index.keys())[:12]]
    print(classification_report(y_true, y_pred,  target_names=labels))

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-2da89cbd-fd3d-fc9c-6d9d-92c1db379af3"
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, help="training epochs")
    parser.add_argument("--predict", type=bool, help="inference")
    parser.add_argument("--input", type=str)
    args = parser.parse_args()
    if args.predict and args.input == None:
        print("to predict need input text")
        sys.exit(0)
    elif args.predict and args.input:
        predict(args)
    else:
        main(args)
