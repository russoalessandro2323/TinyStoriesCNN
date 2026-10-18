import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, fan_out):

        """
            fan_out: embedding dim.
        """

        super().__init__()
        self.embedding_matrix = nn.Parameter(torch.randn(vocab_size, fan_out) / fan_out)

    
    def forward(self, x):
        """
            x: embedded batch, shape = (batch_size, sequence_length)
        """
        return self.embedding_matrix[x]

class WeightNormLayer(nn.Module):

    def __init__(self, fan_in, fan_out, k):

        """
            It reparametrizes each weight as w = g * v / ||v||. The direction is v and the magnitude is g.
        """

        super().__init__()

        self.v = nn.Parameter(torch.randn(fan_out, fan_in, k) / fan_in)
        
        with torch.no_grad():
            norm = torch.linalg.vector_norm(self.v, dim=(1, 2), keepdim=True)

        self.g = nn.Parameter(norm) 

    
    def forward(self):
        return self.g * (self.v / torch.linalg.vector_norm(self.v, dim=(1, 2), keepdim=True))

class DilatedCausalConvolutionalLayer(nn.Module):

    def __init__(self, fan_in, fan_out, k, d):
        """
            fan_in:     embedding dim      \n
            fan_out:    number of kernels  \n
            k:          kernel size        \n
            d:          dilation factor
        """
        
        super().__init__()

        if (d < 1):
            raise ValueError("Dilatation factor has to be >= 1.")

        self.weight_norm = WeightNormLayer(fan_in, fan_out, k)
        self.d = d
        self.k = k
    
    def forward(self, x):
        """
            X: input -> shape (Batch_size, fan_in, L)
            K:                (fan_out, fan_in, k)
            Y:                (Batch_size, fan_out, L)  
        """
        w = self.weight_norm()
        padding = (self.k - 1) * self.d
        x_padded = nn.functional.pad(x,(padding, 0) , "constant", 0)

        return nn.functional.conv1d(x_padded, w, dilation=self.d) 

class ResidualBlock(nn.Module):
    
    def __init__(self, emb_dim, k, d, dropout_prob = 0.1):
        super().__init__()

        self.nn = nn.Sequential(
            DilatedCausalConvolutionalLayer(emb_dim, emb_dim, k, d),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            DilatedCausalConvolutionalLayer(emb_dim, emb_dim, k, d * 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
    
    def forward(self, x):
        return x + self.nn(x)

class CNN(nn.Module):

    def __init__(self, vocab_size, emb_dim, n_res_blocks = 3):
        super().__init__()

        self.embedding = EmbeddingLayer(vocab_size, emb_dim)

        self.blocks = nn.ModuleList(
            [ ResidualBlock(emb_dim, 3, 2 ** i) for i in range(n_res_blocks) ]
        )

        self.conv = nn.Conv1d(in_channels=emb_dim, out_channels=vocab_size, kernel_size=1)

    def forward(self, x):
        x = self.embedding(x)       # out (B, L, emb_dim)
        x = x.transpose(1, 2)   # out (B, emb_dim, L)
        
        for block in self.blocks:
            x = block(x)            # out (B, emb_dim, L)

        return self.conv(x)        # out (B, vocab_size, L)

    def generate_story(self, prompt_text, t, L, device, top_k = 40, temperature = 0.8):
        self.eval()

        formatted_text = "<NWS> " + t.replace_punctuation(prompt_text)
        ids = [t.word2idx.get(tok, t.word2idx["<UNK>"]) for tok in formatted_text.split()]
        idx = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        start = t.word2idx["<NWS>"]

        with torch.no_grad():
            for _ in range(L):
                logits = self.forward((idx[:, -L:]))[0, :, -1].float() / temperature

                logits[[0, 2]] = float("-inf")                                  # no <PAD> or <UNK>
                top_k_th = torch.topk(logits, top_k).values[-1]
                logits = logits.masked_fill(logits < top_k_th, float("-inf"))   # select only top_k  

                nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
                
                if nxt.item() == start:                             
                    break
                
                idx = torch.cat((idx, nxt.unsqueeze(0)), dim=1)

        story = t.decode(idx[0].tolist()).replace("<NWL>", "\n")
        print(f"\n{story}\n")




            
