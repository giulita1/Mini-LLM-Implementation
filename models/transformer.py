import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()

        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, batch_first=True)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size), #recibe el vector original, lo expande
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size) #recibe el vector expandido, sale el dim original
        )
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x, mask):
        #attention + conexion residual
        attention_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(attention_output + x)

        forward_output = self.feed_forward(x)
        out = self.norm2(forward_output + x)

        return self.dropout(out)
    

class LLM(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, device, max_length, dropout, forward_expansion):
        super(LLM, self).__init__()
        self.device = device

        self.word_embedding = nn.Embedding(vocab_size, embed_size) #tokens, dimensiones
        self.position_embedding = nn.Embedding(max_length, embed_size) #frases, dimensiones

        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout=dropout, forward_expansion=forward_expansion)
             for _ in range(num_layers)]
        )

        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        #sumar el significado de las palabras x (convertidos a indices) + su posicion (se suman para ahorrrar memoria)
        out = self.word_embedding(x) + self.position_embedding(positions)

        for layer in self.layers:
            out = layer(out, mask)

        return self.fc_out(out)
    
