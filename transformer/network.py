import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        # Positional encoding adds the positional information to the
        # embedding. Without it the model would not be able to differentiate
        # between different characters orders such as between "dog" and "god".
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = 10000.0 ** (torch.arange(0, d_model, 2).float() / d_model)
        print(div_term.shape)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.cuda()
        self.pe.requires_grad = False

    def forward(self, x):
        p = self.pe[:, :x.size(1)]
        return p


class AttentionMasking(nn.Module):
    def __init__(self, max_len):
        super(AttentionMasking, self).__init__()
        self.register_buffer("mask", torch.tril(torch.ones(max_len, max_len))
                             .view(1, 1, max_len, max_len))

    def forward(self, x):
        length = x.shape[-1]
        out = x.masked_fill(self.mask[:, :, :length, :length] == 0, float('-inf'))
        return out


class ScaledDotProductAttention(nn.Module):
    def __init__(self, max_len):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        # Multiply with an upper triangular
        # matrix of dimensions (length x length) after the scale operation
        # in Figure 2 of the paper.
        self.mask_opt = AttentionMasking(max_len)

    def forward(self, q, k, v):
        # length = number of input tokens
        num_neuron = k.size(3)
        scores = q @ k.transpose(-1, -2) / np.sqrt(num_neuron)
        scores = self.mask_opt(scores)
        attention_weights = self.softmax(scores)
        output = attention_weights @ v
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model, num_neuron, n_head, max_len):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.num_neuron = num_neuron

        self.sdpa = ScaledDotProductAttention(max_len)

        self.w_q = nn.Linear(dim_model, n_head * num_neuron)
        self.w_k = nn.Linear(dim_model, n_head * num_neuron)
        self.w_v = nn.Linear(dim_model, n_head * num_neuron)

        self.linear = nn.Linear(n_head * num_neuron, dim_model)

    def split(self, tensor):
        batch_size, length, total_dim = tensor.size()
        # Reshape the tensor to enable the use in
        # the ScaledDotProductAttention module
        split_tensor = tensor.view(batch_size, length, self.n_head, self.num_neuron).transpose(1, 2)
        return split_tensor

    def concat(self, tensor):
        batch_size, num_heads, length, num_neuron = tensor.size()
        # Reshape the tensor to its original size before the split operation.
        concat_tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, self.n_head * self.num_neuron)
        return concat_tensor

    def forward(self, q, k, v):
        batch_size = q.size(0)

        q = self.w_q(q).view(batch_size, -1, self.n_head, self.num_neuron).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_head, self.num_neuron).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_head, self.num_neuron).transpose(1, 2)
        x = self.sdpa(q, k, v)
        x = self.concat(x)
        x = self.linear(x)

        return x


class PositionFeedForwardNet(nn.Module):
    def __init__(self, dim_model):
        super(PositionFeedForwardNet, self).__init__()
        self.ff_net1 = nn.Linear(dim_model, dim_model * 4)
        self.ff_net2 = nn.Linear(dim_model * 4, dim_model)

    def forward(self, x):
        ff_out = self.ff_net1(x)
        ff_out = torch.nn.functional.relu(ff_out)
        ff_out = self.ff_net2(ff_out)
        return ff_out


class TransformerBlock(nn.Module):
    def __init__(self, dim_model, num_neuron, n_head, max_len):
        super(TransformerBlock, self).__init__()
        self.mha = MultiHeadAttention(dim_model, num_neuron, n_head, max_len)
        self.l_norm = torch.nn.LayerNorm(dim_model)
        self.l_norm2 = torch.nn.LayerNorm(dim_model)
        self.ff_net = PositionFeedForwardNet(dim_model)
        # b, len_seq, n_head, num_neuron

    def forward(self, x):
        # A Transformer block as described in the
        # Attention is all you need paper. In Figure 1 the transformer
        # block is marked with a gray rectangle right of the text "Nx"
        _x = x
        mha1 = self.mha(x, x, x)
        lnorm = self.l_norm(_x + mha1)
        _x = lnorm
        ff_out = self.ff_net(lnorm)
        out = self.l_norm2(ff_out + _x)

        return out


class TransformerSimple(nn.Module):
    def __init__(self, seq_length, input_dim, output_dim,
                 batch_size):
        super(TransformerSimple, self).__init__()
        num_neuron = 64
        n_head = 8
        dim_model = 256
        max_len = 512
        self.start_embedding = nn.Embedding(input_dim, dim_model)

        self.pos_embedding = PositionalEncoding(dim_model)

        # b x l x c*n_head
        self.t_block1 = TransformerBlock(dim_model, num_neuron, n_head, max_len)
        self.t_block2 = TransformerBlock(dim_model, num_neuron, n_head, max_len)
        self.t_block3 = TransformerBlock(dim_model, num_neuron, n_head, max_len)
        self.t_block4 = TransformerBlock(dim_model, num_neuron, n_head, max_len)
        self.t_block5 = TransformerBlock(dim_model, num_neuron, n_head, max_len)

        # self.out_layer_1 = nn.Linear(dim_model, dim_model)
        self.output_layer = nn.Linear(dim_model, output_dim)

    def forward(self, x):
        # x - Tensor - (b, seq_len)
        # Embeds the input tensor from tokens to features
        s_emb = self.start_embedding(x)
        # Adds positional embeddings
        p_emb = self.pos_embedding(s_emb)
        b_out = p_emb + s_emb
        # Transformer blocks - You can experiment with varying depth
        # For example GPT uses 12 blocks but this might be a bit memory intensive
        b_out = self.t_block1(b_out)
        b_out = self.t_block2(b_out)
        b_out = self.t_block3(b_out)
        b_out = self.t_block4(b_out)
        b_out = self.t_block5(b_out)

        # Output mapping to a classification of output tokens
        # For each token the network tries to predict the next token
        # based only on the previous tokens.
        # Output shape: (b x seq_len x vocabulary_size)
        out = self.output_layer(b_out)

        return out
