import torch
from torch import nn


class IronyDection(nn.Module):

    def __init__(self, embedding_vectors, embedding_dim, out_features, hidden_size, dropout, bidirectional, num_layers,
                 rnn):
        super().__init__()
        self.embeddings = nn.Embedding.from_pretrained(embedding_vectors, freeze=False)
        if rnn == 'LSTM':
            self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                               bidirectional=bidirectional, dropout=dropout)
        else:
            self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                              bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # self.ui = nn.Linear(in_features=2 * hidden_size, out_features=150)
        # self.uw = nn.Parameter(torch.randn(150))

        self.fc = nn.Linear(in_features=hidden_size * 2 if bidirectional else hidden_size, out_features=32)
        self.fc1 = nn.Linear(in_features=32, out_features=out_features)

    def forward(self, text, text_length):
        # text = [sent len, batch size]
        embedded = self.embeddings(text)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_length.to('cpu'))
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        # hidden = [num layers * num direc, bacth size, hid dim]

        # enc, _ = nn.utils.rnn.pad_packed_sequence(packed_output)

        # u_it = torch.tanh(self.ui(enc))
        # weights = torch.softmax(u_it.matmul(self.uw), dim=1).unsqueeze(1)
        # sent = torch.sum(weights.matmul(enc), dim=1)
        # logits = self.fc(sent)
        # print(logits)
        # return logits
        if self.rnn.bidirectional:
            concat = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
            # print(concat.size())
            # avg_pool = torch.mean(hidden, dim=0)
            # print(hidden)
            # print(hidden.size())
            # print(avg_pool)
            # print(avg_pool.size())
            # max_pool, _ = torch.max(hidden, dim=0)
            # print(max_pool)
            # print(max_pool.size())
            # concat = torch.cat((avg_pool, max_pool))
            # print(concat)
            # print(concat.size())
            concat = self.relu(self.fc(concat))
            # concat = self.dropout(concat)
        else:
            concat = self.dropout(hidden[-1, :, :])
        # out = self.relu(self.fc(concat))
        out = self.fc1(concat)
        return out
        # print(logits)
        # print(out.squeeze(0))
        # exit()
        # return self.fc(attn_output)
        # return self.fc1(self.dropout(out))
