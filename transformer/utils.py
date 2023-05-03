import numpy as np
import torch
import unidecode
import string
from torch.utils.data import Dataset


class TextDataset(Dataset):
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
        self.encoded_file = [self.char_dict[x] for x in self.file]

    def read_file(self, filename):
        file = unidecode.unidecode(open(filename).read())
        return file, len(file)

    def encode_text(self, in_str):
        # in_str - input sequence - String
        # Returns - in_str mapped to tokens in char_dict
        tensor = torch.LongTensor([self.char_dict[x] for x in in_str])
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
        chunk = self.encoded_file[start_index:end_index]
        # input_tokens - random sequence of tokens from the dataset
        input_tokens = torch.LongTensor(chunk[:-1])
        # target - input token sequence shifted by 1
        # the idea is to predict next token for each token in the input sequence
        # therefore if the input is [1,2,3,4] the target is [2,3,4,5]
        target = torch.LongTensor(chunk[1:])
        input_tokens = input_tokens.cuda()
        target = target.cuda()
        return input_tokens, target


def topk_sampling_iter_transformer(model, x, num_chars, chunk_len, output_token):
    # x -- b x onehot_char
    # x = b x l
    outputs = torch.zeros((1, num_chars))
    inp = x

    for t in range(num_chars):
        # b x onehot_char
        output = model(inp.long())[0, -1:]
        # output = torch.softmax(output, dim=1)
        # b x 3
        output_vals, output_ind = torch.topk(output, 5, dim=1)
        # 3 -> int
        output_vals = torch.softmax(output_vals, dim=1)
        top_ind = torch.multinomial(output_vals[0], 1)[0]
        # int
        out_char_index = output_ind[0, top_ind]
        # int -> 1
        out_char_index = torch.ones(1).cuda() * out_char_index

        outputs[:, t] = out_char_index.item()
        if inp.shape[1] > chunk_len:
            inp = torch.cat((inp[:, 1:], out_char_index.unsqueeze(0)), dim=1)
        else:
            inp = torch.cat((inp, out_char_index.unsqueeze(0)), dim=1)

    return outputs


def greedy_sampling_iter_transformer(model, x, num_chars, chunk_len, output_token):
    # x -- shape (batch, tokens in x)
    outputs = torch.zeros((1, num_chars))
    inp = x

    for t in range(num_chars):
        # b x l x onehot_char
        output = model(inp.long())[0, -1:]
        output = torch.softmax(output, dim=1)
        out_char_index = torch.argmax(output, dim=1)
        outputs[:, t] = out_char_index.item()
        if inp.shape[1] > chunk_len:
            inp = torch.cat((inp[:, 1:], out_char_index.unsqueeze(0)), dim=1)
        else:
            inp = torch.cat((inp, out_char_index.unsqueeze(0)), dim=1)

    return outputs
