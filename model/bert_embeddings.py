import torch
import torch.nn as nn


class BertEmbeddings(nn.Module):

    def __init__(self, bert, hidden_dim, output_dim, n_layers, bidirectional, dropout, rnn):
        super().__init__()

        self.bert = bert
        embedding_dim = self.bert.config.to_dict()['hidden_size']
        if rnn == 'LSTM':
            self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=bidirectional,
                               num_layers=n_layers, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=bidirectional,
                              num_layers=n_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features=hidden_dim * 2 if bidirectional else hidden_dim, out_features=output_dim)

    def forward(self, text):
        with torch.no_grad():
            embedded = self.bert(text)[0]
        output, (hidden, cell) = self.rnn(embedded)
        if self.rnn.bidirectional:
            concat = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            concat = self.dropout(hidden[-1, :, :])
        return self.fc(concat)
