import torch.nn as nn
import torch 
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, params):
        super(EncoderRNN, self).__init__()
        self.hidden_size = params.hidden_size

        self.embedding = nn.Embedding(params.input_size, params.hidden_size)
        self.gru = nn.GRU(params.hidden_size, params.hidden_size, dropout=params.dropout, batch_first=True)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)

        return output, hidden
    

class DecoderRNN(nn.Module):
    def __init__(self, params):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(params.output_size, params.hidden_size)
        self.gru = nn.GRU(params.hidden_size, params.hidden_size, batch_first=True)
        self.out = nn.Linear(params.hidden_size, params.output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden