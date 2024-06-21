import time
import math
import torch.nn as nn
from torch import optim
import torch


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
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _ = decoder(encoder_outputs, encoder_hidden)

    return decoder_outputs.argmax(dim=-1)

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
