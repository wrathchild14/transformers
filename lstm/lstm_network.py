import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.W_c = nn.Linear(input_dim, hidden_dim)
        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.U_i = nn.Linear(hidden_dim, hidden_dim)
        self.U_f = nn.Linear(hidden_dim, hidden_dim)
        self.U_c = nn.Linear(hidden_dim, hidden_dim)
        self.U_o = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, c, h):
        # x - batch of encoded characters
        # C - Cell state of the previous iteration
        # h - Hidden state of the previous iteration

        i = torch.sigmoid(self.W_i(x) + self.U_i(h))
        f = torch.sigmoid(self.W_f(x) + self.U_f(h))
        o = torch.sigmoid(self.W_o(x) + self.U_o(h))
        c_candidate = torch.tanh(self.W_c(x) + self.U_c(h))

        c_out = f * c + i * c_candidate
        h_out = o * torch.tanh(c_out)

        return c_out, h_out


class LSTMSimple(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMSimple, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.lstm_cell = LSTMCell(self.input_dim, self.hidden_dim, self.output_dim)
        self.proj = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        # output_dim = x.shape[2]
        c = torch.zeros(batch_size, self.hidden_dim).cuda()
        h = torch.zeros(batch_size, self.hidden_dim).cuda()
        outputs = torch.zeros((batch_size, seq_length, self.output_dim)).cuda()

        for i in range(seq_length):
            x_t = x[:, i, :]
            c, h = self.lstm_cell(x_t, c, h)
            y_t = self.proj(h)
            outputs[:, i, :] = y_t

        return outputs, (c, h)
