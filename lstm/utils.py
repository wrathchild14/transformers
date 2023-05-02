import numpy as np
import unidecode
import string
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset


class LSTMDataset(Dataset):
    def __init__(self, chunk_len=200, padded_chunks=False):
        # Character based dataset
        dataset_path = "../data/input.txt"
        # The tokens in the vocabulary (all_characters)
        # are just the printable characters of the string class
        self.all_characters = string.printable
        self.n_characters = len(self.all_characters)
        # Maps characters to indices
        self.char_dict = {x: i for i, x in enumerate(self.all_characters)}
        self.file, self.file_len = self.read_file(dataset_path)
        # Sequence length of the input
        self.chunk_len = chunk_len

    def read_file(self, filename):
        file = unidecode.unidecode(open(filename).read())
        return file, len(file)

    def char_tensor(self, in_str):
        # in_str - input sequence - String
        # Return one-hot encoded characters of in_str
        tensor = torch.zeros(len(in_str), self.n_characters).long()
        char_ind = [self.char_dict[c] for c in in_str]
        tensor[torch.arange(tensor.shape[0]), char_ind] = 1
        return tensor

    def __getitem__(self, idx):
        inp, target = self.get_random_text()
        return {"input": inp, "target": target}

    def __len__(self):
        return 10000

    def get_random_text(self):
        # Pick a random string of length self.chunk_len from the dataset
        start_index = np.random.randint(0, self.file_len - self.chunk_len)
        end_index = start_index + self.chunk_len + 1
        chunk = self.file[start_index:end_index]
        # One-hot encode the chosen string
        inp = self.char_tensor(chunk[:-1])
        # The target string is the same as the
        # input string but shifted by 1 character
        target = self.char_tensor(chunk[1:])
        inp = Variable(inp).cuda()
        target = Variable(target).cuda()
        return inp, target


def greedy_sampling_lstm(lstm, x, num_chars):
    # x -- b x onehot_char
    outputs = torch.zeros((1, num_chars, x.shape[2]))
    t_outputs, (cell_state, hidden) = lstm(x.float())
    for c in range(num_chars):
        output_tmp = torch.softmax(lstm.proj(hidden), dim=1)
        top_ind = torch.argmax(output_tmp, dim=1)[0]
        tmp = torch.zeros_like(x[:, 0, :]).cuda()
        tmp[:, top_ind] = 1
        outputs[:, c] = tmp

        cell_state, hidden = lstm.lstm_cell(tmp, cell_state, hidden)
    return outputs


def topk_sampling_lstm(lstm, x, num_chars):
    # x -- b x onehot_char
    outputs = torch.zeros((1, num_chars, x.shape[2]))
    t_outputs, (cell_state, hidden) = lstm(x.float())
    for c in range(num_chars):
        output_vals, output_ind = torch.topk(lstm.proj(hidden), 5, dim=1)
        output_tmp = torch.softmax(output_vals, dim=1)
        top_ind = torch.multinomial(output_tmp[0], 1)[0]
        tmp = torch.zeros_like(x[:, 0, :]).cuda()
        tmp[:, output_ind[0, top_ind]] = 1
        outputs[:, c] = tmp

        cell_state, hidden = lstm.lstm_cell(tmp, cell_state, hidden)

    return outputs
