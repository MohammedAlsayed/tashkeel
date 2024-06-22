import time
import math
import torch.nn as nn
from torch import optim
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import numpy as np
from utils import clean_sentence, has_any_diacritics
from langchain.docstore.document import Document

def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):

    total_loss = 0

    encoder.train()
    decoder.train()

    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        batch_size = input_tensor.size(0)
        hidden = encoder.init_hidden(batch_size)
        encoder_outputs, encoder_hidden = encoder(input_tensor, hidden)
        decoder_outputs, _ = decoder(encoder_outputs, encoder_hidden, None)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def validate(dataloader, encoder, decoder, criterion):
    total_loss = 0

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for data in dataloader:
            input_tensor, target_tensor = data

            batch_size = input_tensor.size(0)
            hidden = encoder.init_hidden(batch_size)
            encoder_outputs, encoder_hidden = encoder(input_tensor, hidden)
            decoder_outputs, _ = decoder(encoder_outputs, encoder_hidden, None)

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )

            total_loss += loss.item()

    return total_loss / len(dataloader)


def inference(encoder, decoder, input_tensor):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        batch_size = input_tensor.size(0)
        hidden = encoder.init_hidden(batch_size)
        encoder_outputs, encoder_hidden = encoder(input_tensor, hidden)
        decoder_outputs, _ = decoder(encoder_outputs, encoder_hidden)

    return decoder_outputs.argmax(dim=-1)

def test_result(dataloader, encoder, decoder):
    encoder.eval()
    decoder.eval()

    total = 0
    equal_list = []
    for data in dataloader:
        input_tensor, target_tensor = data
        output_tensor = inference(encoder, decoder, input_tensor)
        # average accuracy of the batch
        for idx, (predict,target) in enumerate(zip(output_tensor, target_tensor)):
            # count matching numbers in the two tensors using torch function
            equal = torch.eq(predict, target).sum()
            if equal == target_tensor.size(-1):
                equal_list.append([input_tensor[idx], predict, target])
            total += equal.item()

    print(f'accuracy: {total / (len(dataloader.dataset) * target_tensor.size(-1))}')
    return equal_list

def prepare_data(docs, tokenizer):
    max_input = 0  
    max_target = 0
    average_target = 0
    average_input = 0
    pairs = []
    for doc in tqdm(docs):
        try:
            input_ids, target_ids = tokenizer.get_pair(doc.page_content, encoded=True)
        except:
            # print("exception: ", doc.page_content)
            continue
        pairs.append((input_ids, target_ids))
        max_input = max(max_input, len(input_ids))
        max_target = max(max_target, len(target_ids))
        average_target += len(target_ids)
        average_input += len(input_ids)

    average_target /= len(pairs)
    average_input /= len(pairs)
    
    return pairs, max_input, max_target, average_input, average_target

def get_dataloader(pairs, max_input, max_target, batch_size, device, truncate=True):
    n = len(pairs)
    input_ids = np.zeros((n, max_input), dtype=np.int32)
    target_ids = np.zeros((n, max_target), dtype=np.int32)

    for idx, (inp_ids, tgt_ids) in tqdm(enumerate(pairs)):
        if truncate:
            inp_ids = inp_ids[:max_input]
            tgt_ids = tgt_ids[:max_target]
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader

# clean the data
def clean_data(docs):
    cleaned_docs = []
    for doc in tqdm(docs):
        clean = clean_sentence(doc.page_content)
        if len(clean) == 0:
            continue
        if not has_any_diacritics(clean):
            continue
        cleaned_doc = Document(page_content=clean, metadata=doc.metadata)
        cleaned_docs.append(cleaned_doc)
    return cleaned_docs

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def main(train_dataloader, validate_dataloader, encoder, decoder, n_epochs, learning_rate=0.001, print_every=100):
    start = time.time()
    print_loss_total = 0  
    plot_loss_total = 0 

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            validate_loss = validate(validate_dataloader, encoder, decoder, criterion)
            print(f'time (time left): {timeSince(start, epoch / n_epochs)}\nepoch:{epoch}/{n_epochs}\ntrain loss: {print_loss_avg: 0.3f}, validate loss: {validate_loss:0.3f}')
