from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence

class TextDataset(Dataset):

    def __init__(self, text, vocab, seq_len=64):

        tokens = vocab.encode(text)

        self.inputs = []
        self.targets = []

        for i in range(len(tokens) - seq_len):

            x = tokens[i:i+seq_len]
            y = tokens[i+1:i+seq_len+1]

            self.inputs.append(torch.tensor(x))
            self.targets.append(torch.tensor(y))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):

        return self.inputs[index], self.targets[index]


def collate_fn(batch):

    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]

    xs = pad_sequence(xs, batch_first=True, padding_value=0)
    ys = pad_sequence(ys, batch_first=True, padding_value=0)

    return xs, ys