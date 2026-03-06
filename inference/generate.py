import torch
from src_utils.tokenizer import Vocab
from models.transformer import LLM

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# tokenizer
vocab = Vocab()

# cargar modelo
model = LLM(
    vocab_size=vocab.vocab_size,
    embed_size=128,
    num_layers=4,
    heads=4,
    device=device,
    max_length=64,
    dropout=0.1,
    forward_expansion=4
).to(device)

model.load_state_dict(torch.load("checkpoints/model.pt", map_location=device))

model.eval()

def generate_text(prompt, max_new_tokens=20, temperature=0.8, top_k=40):
    tokens = vocab.encode(prompt)
    tokens = torch.tensor(tokens).unsqueeze(0).to(device).long()

    for _ in range(max_new_tokens):
        seq_len = tokens.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1).to(device)

        with torch.no_grad():
            logits = model(tokens, mask)
        
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            values, indices = torch.topk(logits, top_k)
            logits_filtered = torch.full_like(logits, float("-inf"))
            logits_filtered.scatter_(1, indices, values)
            logits = logits_filtered

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).long()
        tokens = torch.cat((tokens, next_token), dim=1)

    return vocab.decode(tokens.squeeze().tolist())

if __name__ == "__main__":
    while True:
        prompt = input("Prompt: ")
        print(generate_text(prompt))