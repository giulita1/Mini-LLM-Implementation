import torch
from torch.utils.data import DataLoader, random_split
from src_utils.tokenizer import Vocab
from src_utils.dataset import TextDataset, collate_fn
from models.transformer import LLM
import torch.nn as nn
from tqdm import tqdm

#hiperparametros y config
num_epochs = 10
batch_size = 32
learning_rate = 3e-4
max_seq_len = 64

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#cargar datos y vocab

print("Cargando datos...")
with open("data/dataset.txt", "r", encoding="utf-8") as f:
    text = f.read(10000)
print(f"Datos cargados. Tamaño del texto: {len(text)} caracteres.")

vocab = Vocab()
vocab_size = vocab.vocab_size
pad_idx = vocab.eot_token

dataset = TextDataset(text, vocab, seq_len=64)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    collate_fn=collate_fn
)

#modelo
model = LLM(
    vocab_size=vocab_size,
    embed_size=128,
    num_layers=4,
    heads=4,
    device=device,
    max_length=64,
    dropout=0.1,
    forward_expansion=4
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx) #ignorar el error en los pad

for epoch in tqdm(range(num_epochs)):
    
    #training
    model.train()
    train_loss = 0 

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        seq_length = data.shape[1]

        mask = torch.triu(torch.ones(seq_length, seq_length) * float("-inf"), diagonal=1).to(device)
        
        #forward
        scores = model(data, mask)

        #el modelo devuelve N, seq_length, vocab size
        #aplanamos para ccompararlo con las palabras reales
        loss = criterion(scores.view(-1, vocab_size), targets.view(-1))
        train_loss += loss.item()

        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()

    #evaluacion 

    model.eval()
    test_loss = 0

    with torch.no_grad():
        for data, targets in test_loader:
            scores = model(data, mask)
            loss = criterion(scores.view(-1, vocab_size), targets.view(-1))
            test_loss += loss.item()

    import math

    #calcular promedios
    train_loss_avg = train_loss / len(train_loader)
    test_loss_avg = test_loss / len(test_loader)

    #calcular perplexity
    train_perplexity = math.exp(train_loss_avg)
    test_perplexity = math.exp(test_loss_avg)

    print(f"epoch: {epoch}")
    print(f"train Loss: {train_loss_avg:.4f} | train perplexity: {train_perplexity:.2f}")
    print(f"test Loss: {test_loss_avg:.4f}  | test perplexity: {test_perplexity:.2f}")

torch.save(model.state_dict(), "checkpoints/model.pt")
print("Modelo guardado en checkpoints/model.pt")