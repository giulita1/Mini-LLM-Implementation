import tiktoken

class Vocab:
    def __init__(self):
        self.encoder = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.encoder.n_vocab
        
        self.eot_token = self.encoder.eot_token 

    def encode(self, text):
        return self.encoder.encode(text)

    def decode(self, ids):
        return self.encoder.decode(ids)