from tkinter import Variable

import torch
from tqdm import tqdm
import torch.optim as optim

from lstm.lstm_network import LSTMSimple
from lstm.utils import LSTMDataset, topk_sampling_lstm, greedy_sampling_lstm

batch_size = 256
chunk_len = 128
model_name = "LSTM"
train_dataset = LSTMDataset(chunk_len=chunk_len)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0)

# Sample parameters, use whatever you see fit.
input_dim = train_dataset.n_characters
hidden_dim = 256
output_dim = train_dataset.n_characters
learning_rate = 0.005
model = LSTMSimple(chunk_len, input_dim, hidden_dim, output_dim, batch_size)
model.train()
model.cuda()

criterion = torch.nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

epochs = 2

for epoch in range(epochs):
    with tqdm(total=len(trainloader.dataset), desc='Training - Epoch: ' + str(epoch) + "/" + str(epochs),
              unit='chunks') as prog_bar:
        for i, data in enumerate(trainloader, 0):
            inputs = data['input'].float()
            labels = data['target'].float()
            # b x chunk_len x len(dataset.all_characters)
            target = torch.argmax(labels, dim=2)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs.view(inputs.shape[0] * inputs.shape[1], -1),
                             target.view(labels.shape[0] * labels.shape[1]))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=10.0)
            optimizer.step()
            prog_bar.set_postfix(**{'run:': model_name, 'lr': learning_rate,
                                    'loss': loss.item()
                                    })
            prog_bar.update(batch_size)
        # Intermediate output
        sample_text = "O Romeo, wherefore art thou"
        inp = train_dataset.char_tensor(sample_text)
        sample_input = Variable(inp).cuda().unsqueeze(0).float()
        out_test = topk_sampling_lstm(model, sample_input, 300)[0]
        out_char_index = torch.argmax(out_test, dim=1).detach().cpu().numpy()
        out_chars = sample_text + "".join([train_dataset.all_characters[i] for i in out_char_index])
        print("Top-K sampling -----------------")
        print(out_chars)

        out_test = greedy_sampling_lstm(model, sample_input, 300)[0]
        out_char_index = torch.argmax(out_test, dim=1).detach().cpu().numpy()
        out_chars = sample_text + "".join([train_dataset.all_characters[i] for i in out_char_index])
        print("Greedy sampling ----------------")
        print(out_chars)
