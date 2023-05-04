import torch
from tqdm import tqdm
import torch.optim as optim

from transformer.network import TransformerSimple
from transformer.utils import TextDataset, topk_sampling_iter_transformer

# Sample parameters, use whatever you see fit.
batch_size = 256
chunk_len = 128
train_dataset = TextDataset(chunk_len=chunk_len)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0)

input_dim = train_dataset.n_characters
output_dim = train_dataset.n_characters
learning_rate = 0.0006

model = TransformerSimple(chunk_len, input_dim, output_dim, batch_size)
model.train()
model.cuda()

criterion = torch.nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

epochs = 50

for epoch in range(epochs):
    with tqdm(total=len(trainloader.dataset), desc='Training - Epoch: ' + str(epoch) + "/" + str(epochs),
              unit='chunks') as prog_bar:
        for i, data in enumerate(trainloader, 0):
            # inputs - shape (batch_size, chunk_len) - Tensor of vocabulary tokens
            inputs = data['input'].long()
            # labels - shape (batch_size, chunk_len) - Tensor of vocabulary tokens
            labels = data['target'].long()

            optimizer.zero_grad()
            outputs = model(inputs)
            target_t = labels
            loss = criterion(outputs.view(inputs.shape[0] * inputs.shape[1], -1),
                             target_t.view(labels.shape[0] * labels.shape[1]))
            loss.backward()
            optimizer.step()
            prog_bar.set_postfix(**{'run:': "Transformer", 'lr': learning_rate,
                                    'loss': loss.item()
                                    })
            prog_bar.update(batch_size)

        # Intermediate text output
        sample_texts = ["What authority surfeits on",
                        "I say unto you, what he hath done famously, he did it to that end:",
                        "That in submission will return to us: And then, as we have ta'en the sacrament,"]
        output_token = torch.zeros(1, 1).cuda()
        output_token[0, 0] = train_dataset.n_characters - 1
        print("Top-K sampling")
        for sample_text in sample_texts:
            sample_encoding = train_dataset.encode_text(sample_text)
            sample_input = sample_encoding.cuda().unsqueeze(0).long()

            # out_test= greedy_sampling_iter_transformer(model, sample_input, 400, chunk_len, output_token)[0]
            out_test = topk_sampling_iter_transformer(model, sample_input, 400, chunk_len, output_token)[0]
            out_char_index = out_test.long().detach().cpu().numpy()
            out_chars = sample_text + " " + "".join([train_dataset.all_characters[i] for i in out_char_index])
            print("----------------------------------------")
            print(out_chars)

''' Text sampling
sample_text = "Here's to my love! O true apothecary! Thy drugs are quick."
sample_encoding = train_dataset.encode_text(sample_text)
sample_input = sample_encoding.cuda().unsqueeze(0).long()

#out_test= greedy_sampling_iter_transformer(model, sample_input, 400, chunk_len, output_token)[0]
out_test= topk_sampling_iter_transformer(model, sample_input, 400, chunk_len, output_token)[0]
out_char_index = out_test.long().detach().cpu().numpy()
out_chars = sample_text+" "+"".join([train_dataset.all_characters[i] for i in out_char_index])
print("----------------------------------------")
print(out_chars)
'''
